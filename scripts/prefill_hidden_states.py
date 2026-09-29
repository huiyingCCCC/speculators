#!/usr/bin/env python3
"""
Async Hidden-States Prefill Worker
"""

import argparse
import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path

import openai
from datasets import load_from_disk

from speculators.data_generation.disk_queue import DiskQueueManager
from speculators.data_generation.vllm_client import wait_for_lock_async
from speculators.train.data import build_client_item

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Asynchronously prefill hidden-state files into a bounded disk queue"
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        default="http://localhost:8000/v1",
        help="vLLM endpoint configured for hidden-states extraction.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model ID (default: auto-select first model at endpoint).",
    )
    parser.add_argument(
        "--preprocessed-data",
        type=str,
        required=True,
        help="Preprocessed dataset (as produced by prepare_data.py).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Directory that hosts the bounded disk queue (hs_{idx}.safetensors + manifest + consumed markers). "
             "Must match the trainer's --hidden-states-path.",
    )
    parser.add_argument(
        "--disk-quota-gib",
        type=float,
        default=40.0,
        help="Maximum on-disk footprint for the hidden-state window in GiB.",
    )
    parser.add_argument(
        "--eviction-ratio",
        type=float,
        default=0.7,
        help="Evict consumed files until usage drops to [quota x ration] before generating more.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=32,
        help="Number of concurrent vLLM requests to keep in flight.",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=900.0,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=2,
        help="Retries per request on failure.",
    )
    parser.add_argument(
        "--prefill-order",
        type=str,
        default=None,
        help="Optional path to a JSON array (or newline-separated text file) of global dataset indices in the "
             "exact order training will consume them. When omitted, sequential global order 0..N-1 is used. Used a "
             "shared pre-shuffle order file to align a random-access trainer with the sliding window.",
    )
    return parser.parse_args()


def _load_prefill_order(path: str | None, n: int) -> lit[int]:
    if path is None:
        order = list(range(n))
    else:
        p = Path(path)
        raw = p.read_text()
        try:
            order = json.loads(raw)
        except json.JSONDecodeError:
            order = [int(line) for line in raw.split() if line.strip()]
    if not order:
        raise SystemError(f"Empty prefill order {path}")
    return [int(i) for i in order if 0 <= int(i) < n]


def _estimated_bytes(queue: DiskQueueManager) -> int:
    return queue.estimated_bytes()


async def _await_capacity(queue: DiskQueueManager, committed_bytes: int=0) -> bool:
    """Return True once the window accepts another sample.

    ``committed_bytes`` is the estimated on-disk bytes of requests that are already in flight (task launched but file
    not yet landed_. Accounting for it prevents the disk foorptint from overshooting the quota when many concurrent
    requests land around the same time.

    Tries evicting consumed files (FIFO) first; if still not enough room the caller must wait (backpressure).
    """

    est = _estimated_bytes(queue)
    fits = (
        queue.can_enqueue(est)
        and queue.current_usage_bytes() + committed_bytes + est <= queue.quota_bytes
    )
    if fits:
        return True
    freed = queue.evict_consumed()
    if freed:
        logger.info("Evicted consumed files, freed %.2f GiB", freed / 1024**3)
    return (
        queue.can_enqueue(est)
        and queue.current_usage_bytes() + committed_bytes + est <= queue.quota_bytes
    )

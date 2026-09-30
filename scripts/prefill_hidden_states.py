#!/usr/bin/env python3
"""
Async Hidden-States Prefill Worker: Async Prefill + Sliding Window queue

A dedicated asyncio process keeps 32-64 requests resident in the vLLM waiting queue (saturating continuous batching)
and writes completed hidden-state files into a *bounded* disk queue, decoupled from the trainer. The trainer runs with
``--on-missing wait --on-generate noop`` and consumes files as they become ready; this worker owns eviction (FIFO
window slide) so training never waits on inference and inference never idles waiting on training.

Coordination is lock-free over the shared filesystem (see ``disk_queue.py``): the worker is the single manifest writer
the trainer signales consumption by creating passive ``hs_{idx}.consumed`` markers, and the worker evicts consumed
files to free window slots.

Usage:
    python scripts/prefill_hidden_states.py \
        --endpoint http://127.0.0.1:8007/v1 \
        --model XXXX \
        --preprocessed-data /data/.../dataset \
        --output /data/.../hs_queue \
        --disk-quota-gib 40 \
        --oncurrency 32 \
        [--prefill-order order.json]
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


async def prefill_loop(args, dataset, queue: DiskQueueManager):
    semaphore = asyncio.Semaphore(args.concurrency)
    max_inflight = args.concurrency * 2

    async with openai.AsyncOpenAI(
        base_url=args.endpoint, api_key="EMPTY", max_retries=0
    ) as client:
        list_models = await client.models.list()
        model_id = list_models.data[0].id
        if args.model and args.model != model_id:
            raise ValueError(f"Explicit --model {args.model} does not match endpoint model {model_id}")

        order = _load_prefill_order(args.prefill_order, len(dataset))
        if args.max_samples is not None:
            order = order[:args.max_samples]
        logger.info(
            "Prefilling %d samples (concurrency=%d, quota=%.0f GiB) at %s",
            len(order), args.concurrency, args.disk_quota_gib, args.endpoint,
        )

        states = {"ok": 0, "err": 0, "start": time.perf_counter()}
        active: set[asyncio.Task] = set()
        reserved_bytes = 0

        async def worker(idx: int):
            nonlocal reserved_bytes
            async with semaphore:
                try:
                    dataset_item = dataset[idx]
                    client_item = build_client_item(dataset_item)
                    handle = await _generate(client, model_id, client_item, args)
                    target = queue.sample_path(idx)
                    await _move_into_queue(handle, target)
                    queue.mark_ready(idx, os.path.getsize(target))
                    states["ok"] += 1
                except Exception as e: # noqa: BLE001
                    logger.warning("Prefill failed for sample %d: %s", idx, e)
                    queue.drop(idx)
                    states["err"] += 1
                finally:
                    reserved_bytes -= _estimated_bytes(queue)
                    active.discard(asyncio.current_task())

        for idx in order:
            # Bound in-flight work: wait for a slot (frees quota / a task).
            while (
                len(active) >= max_inflight
                or not await _await_capacity(queue, reserved_bytes)
            ):
                if active:
                    done, _ = await asyncio.wait(
                        active, return_when=asyncio.FIRST_COMPLETED
                    )
                    for t in done:
                        t.result()
                else:
                    # Nothing to wait on but the window is full: evict and retry
                    queue.evict_consumed()
                    await asyncio.sleep(0.05)
            queue.mark_in_progress(idx)
            reserved_bytes += _estimated_bytes(queue)
            active.add(asyncio.create_task(worker(idx)))

        if active:
            await asyncio.wait(active)
            for t in active:
                t.result()

    elapsed = time.perf_counter() - states["start"]
    logger.info(
        "Prefill done: ok=%d err=%d elapsed=%.1fs rate=%.2fsamples/s",
        states["ok"], states["err"], elapsed, states["ok"] / elapsed if elapsed > 0 else 0.0,
    )


async def _generate(client, model_id, client_item, args):
    from spelulators.data_generation.vllm_client import generate_hidden_states_async

    return await generate_hidden_states_async(
        client, model_id, client_item, timeout=args.request_timeout, max_retries=args.max_retries,
    )


async def _move_into_queue(handle: str, target: Path) -> None:
    lock_path = handle + ".lock"
    if Path(lock_path).exists(): # noqa: ASYNC240
        await wait_for_lock_async(lock_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    await asyncio.to_thread(shutil.move, handle, target)


def main():
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    queue = DiskQueueManager(
        Path(args.output), args.dick_quota_gib, args.eviction_ratio,
    )
    dataset = load_from_disk(args.preprocessed_data)
    try:
        asyncio.run(prefill_loop(args, dataset, queue))
    except KeyboardInterrupt:
        logger.info("Prefill interrupted")
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Callable

import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from hs_connectors import HiddenStatesTransfer
from hs_connectors.transfer import adxl_proxy_requested
from speculators.train.data import (
    ArrowDataset,
    BaseDataset,
    CollateFn,
)
from speculators.train.distributed import (
    get_dp_rank,
    get_dp_size,
    get_local_rank,
    get_rank,
)
from speculators.train.distributed_batch_sampler import (
    MultipackDistributedBatchSamplerV2,
)
from speculators.train.noise_transforms import AddUniformNoise

logger = logging.getLogger(__name__)

BatchType = dict[str, Any]


def _limit_worker_threads() -> None:
    """Limit per-worker thread pools to avoid thread exhaustion.

    With ``multiprocessing_context='spawn'``, each worker is a full process
    that re-imports numpy (OpenBLAS) and torch, each creating thread pools
    sized to the core count.  DataLoader workers only do I/O and tensor
    slicing — they don't benefit from intra-op parallelism.

    The env vars must be set before numpy/torch are imported to take effect
    on OpenBLAS/OMP.  Call this at the top of the training entry point,
    before DataLoader construction.
    """
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")


def _worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    torch.set_num_threads(1)


def _adxl_proxy_mode_enabled() -> bool:
    """Return a process-group-wide proxy mode flag."""
    requested = adxl_proxy_requested()
    if not dist.is_available() or not dist.is_initialized():
        return requested

    accelerator = torch.accelerator.current_accelerator()
    device = (
        torch.device(accelerator.type, get_local_rank())
        if accelerator is not None
        else torch.device("cpu")
    )
    requested_tensor = torch.tensor(
        [int(requested)], dtype=torch.int64, device=device
    )
    dist.all_reduce(requested_tensor, op=dist.ReduceOp.MAX)
    return bool(requested_tensor.item())


def _setup_dataloader(
    dataset: BaseDataset,
    total_seq_len: int,
    hidden_size: int,
    num_workers: int = 12,
    num_target_layers: int = 3,
    prefetch_factor: int | None = 4,
    preprocess: Callable[[BatchType], BatchType] | None = None,
    pin_memory: bool = True,
    single_data_source: bool = False,
) -> DataLoader:
    batch_sampler = MultipackDistributedBatchSamplerV2(
        batch_max_length=total_seq_len,
        lengths=dataset.approx_lengths,
        num_replicas=1 if single_data_source else get_dp_size(),
        rank=0 if single_data_source else get_dp_rank(),
    )
    use_workers = num_workers > 0
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if use_workers else None,
        pin_memory=pin_memory,
        collate_fn=CollateFn(
            total_seq_len,
            hidden_size,
            num_target_layers=num_target_layers,
            dtype=dataset.hidden_states_dtype,
            preprocess=preprocess,
        ),
        persistent_workers=use_workers,
        multiprocessing_context="spawn" if use_workers else None,
        worker_init_fn=_worker_init_fn if use_workers else None,
    )


def create_train_val_loaders(
    *,
    data_path: str,
    total_seq_len: int,
    hidden_states_dtype: torch.dtype,
    noise_std: float,
    transfer: HiddenStatesTransfer | None = None,
    vllm_endpoint: str,
    on_missing: Literal["generate", "skip", "warn", "raise"],
    on_generate: Literal["cache", "delete"],
    verifier_name_or_path: str,
    request_timeout: float | None,
    max_retries: int,
    hidden_size: int,
    num_target_layers: int,
    num_workers: int,
    prefetch_factor: int,
    preprocess: Callable[[BatchType], BatchType] | None,
    train_data_ratio: float = 0.9,
) -> tuple[DataLoader, DataLoader]:
    """Create training and validation DataLoaders.

    Non-data SP ranks get lightweight loaders with no workers (they receive
    batches via scatter).  Reads DP/SP topology from
    :mod:`speculators.train.distributed`.
    """
    _limit_worker_threads()
    noise_transform = AddUniformNoise(std=noise_std)
    proxy_mode = _adxl_proxy_mode_enabled()
    proxy = proxy_mode and get_rank() != 0
    loader_workers = 0 if proxy_mode else num_workers
    if proxy_mode and num_workers > 0 and get_rank() == 0:
        logger.info(
            "ADXL proxy mode uses the training process as the single Mooncake "
            "consumer; disabling DataLoader workers"
        )

    if not (0.0 < train_data_ratio < 1.0):
        raise ValueError(f"train_data_ratio must be in (0, 1), got {train_data_ratio}")

    train_dataset: BaseDataset = ArrowDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transfer=transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        transform=noise_transform,
        train_ratio=train_data_ratio,
        split="train",
        model=verifier_name_or_path,
        hidden_states_dtype=hidden_states_dtype,
        request_timeout=request_timeout,
        max_retries=max_retries,
        should_generate=not proxy,
    )
    val_dataset: BaseDataset = ArrowDataset(
        datapath=data_path,
        max_len=total_seq_len,
        transfer=transfer,
        vllm_endpoint=vllm_endpoint,
        on_missing=on_missing,
        on_generate=on_generate,
        train_ratio=train_data_ratio,
        split="val",
        model=verifier_name_or_path,
        hidden_states_dtype=hidden_states_dtype,
        request_timeout=request_timeout,
        max_retries=max_retries,
        should_generate=not proxy,
    )

    train_loader = _setup_dataloader(
        train_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        pin_memory=not proxy,
        single_data_source=proxy_mode,
    )
    val_loader = _setup_dataloader(
        val_dataset,
        total_seq_len,
        hidden_size,
        num_target_layers=num_target_layers,
        num_workers=loader_workers,
        prefetch_factor=prefetch_factor,
        preprocess=preprocess,
        pin_memory=not proxy,
        single_data_source=proxy_mode,
    )

    return train_loader, val_loader

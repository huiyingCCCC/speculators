"""Abstraction for hidden-states transfer between vLLM and the trainer."""

from __future__ import annotations

import dataclasses
import fcntl
import os
import shutil
import socket
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

import torch
from safetensors.torch import load_file

from hs_connectors.mooncake_store import MooncakeHiddenStatesStore, MooncakeStoreConfig

ADXL_PROXY_ENV = "MOONCAKE_ADXL_PROXY"


def adxl_proxy_requested() -> bool:
    """Return whether the launch requested ADXL proxy coordination."""
    return os.environ.get(ADXL_PROXY_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def adxl_proxy_enabled() -> bool:
    """Return whether this process is a non-producing ADXL proxy.

    The environment variable may be exported for the whole torchrun launch.
    Rank zero remains the producer in that case; only non-zero ranks disable
    vLLM and Mooncake setup.
    """
    if not adxl_proxy_requested():
        return False
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() != 0
    try:
        return int(os.environ.get("RANK", "0")) != 0
    except ValueError:
        return False


if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable


def wait_for_lock(lock_path: str, timeout: float = 10.0, poll_interval: float = 0.1):
    fd = os.open(lock_path, os.O_RDWR)
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"Timed out waiting for lock: {lock_path}"
                    ) from None
                time.sleep(poll_interval)
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    os.remove(lock_path)


class HiddenStatesTransfer(ABC):
    """Interface for reading hidden states produced by vLLM."""

    def setup(self) -> None:  # noqa: B027
        """Lazy initialization (safe to call from dataloader worker)."""

    @abstractmethod
    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        """Return a previously cached sample, or ``None``."""

    @abstractmethod
    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        """Retrieve a freshly generated sample by its vLLM-returned handle."""

    def cache(self, handle: str, file_idx: int) -> None:  # noqa: B027
        """Persist a generated sample to the cache location."""

    def delete(self, handle: str) -> None:  # noqa: B027
        """Clean up a generated sample (e.g. delete a temp file)."""


class HiddenStatesBackend(ABC):
    """Plugin interface for hidden-states transfer backends.

    Each backend registers itself via ``@HiddenStatesBackend.register(name)``
    and implements these four static hooks so that scripts (``train.py``,
    ``launch_vllm.py``) can discover and configure backends without hardcoding.
    """

    registry: ClassVar[dict[str, type[HiddenStatesBackend]]] = {}

    @classmethod
    def register(
        cls,
        name: str,
    ) -> Callable[[type[HiddenStatesBackend]], type[HiddenStatesBackend]]:
        def decorator(
            subclass: type[HiddenStatesBackend],
        ) -> type[HiddenStatesBackend]:
            if name in cls.registry:
                raise ValueError(f"Backend '{name}' is already registered.")
            cls.registry[name] = subclass
            return subclass

        return decorator

    @staticmethod
    @abstractmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``train.py``."""
        ...

    @staticmethod
    @abstractmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        """Add backend-specific CLI arguments to ``launch_vllm.py``."""
        ...

    @staticmethod
    @abstractmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> HiddenStatesTransfer:
        """Construct a :class:`HiddenStatesTransfer` from parsed train args."""
        ...

    @staticmethod
    @abstractmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        """Construct the ``kv_transfer_config`` dict for ``vllm serve``."""
        ...


# ---------------------------------------------------------------------------
# File-based backend (shared filesystem)
# ---------------------------------------------------------------------------


def _load_hs_file(file_path: Path) -> dict[str, torch.Tensor] | None:
    lock_path = str(file_path) + ".lock"
    if Path(lock_path).exists():
        wait_for_lock(lock_path)

    if file_path.exists():
        return load_file(file_path)

    return None


class FileTransfer(HiddenStatesTransfer):
    """File-system based hidden-states transfer (shared filesystem)."""

    def __init__(self, hidden_states_path: Path):
        self.hidden_states_path = hidden_states_path

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:
        path = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        return _load_hs_file(path)

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        return _load_hs_file(Path(handle))

    def cache(self, handle: str, file_idx: int) -> None:
        self.hidden_states_path.mkdir(parents=True, exist_ok=True)
        target = self.hidden_states_path / f"hs_{file_idx}.safetensors"
        shutil.move(handle, target)

    def delete(self, handle: str) -> None:
        Path(handle).unlink()


@HiddenStatesBackend.register("file")
class FileBackend(HiddenStatesBackend):
    """Shared-filesystem backend using safetensors files."""

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default=None,
            help=(
                "The path where cached hidden states files are stored. (Default: "
                "args.data_path / 'hidden_states')"
            ),
        )

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--hidden-states-path",
            type=str,
            default="/tmp/hidden_states",  # noqa: S108
            help="The directory to save hidden states to. Default '/tmp/hidden_states'",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,
    ) -> FileTransfer:
        hs_path = (
            Path(args.hidden_states_path)
            if args.hidden_states_path
            else Path(data_path) / "hidden_states"
        )
        return FileTransfer(hs_path)

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        return {
            "kv_connector": "ExampleHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_extra_config": {
                "shared_storage_path": args.hidden_states_path,
            },
        }


# ---------------------------------------------------------------------------
# Mooncake-based backend (distributed store)
# ---------------------------------------------------------------------------


class MooncakeTransfer(HiddenStatesTransfer):
    """Mooncake distributed store based hidden-states transfer."""

    def __init__(self, store: MooncakeHiddenStatesStore, proxy: bool = False):
        self.store = store
        self.proxy = proxy

    def setup(self) -> None:
        if self.proxy:
            return
        if not self.store.is_setup:
            # Ascend Direct transport needs an active NPU context
            # (aclrtSetDevice) in the calling process. DataLoader workers
            # inherit the parent rank's device but do not activate it, so
            # engine.initialize can fail with a null runtime context.
            if self.store.config.protocol == "ascend":
                try:
                    import torch_npu  # noqa: PLC0415

                    torch_npu.npu.set_device(torch_npu.npu.current_device())
                except Exception:  # noqa: BLE001
                    pass
            self.store.setup()

    def get_cached(self, file_idx: int) -> dict[str, torch.Tensor] | None:  # noqa: ARG002
        return None

    def get_generated(self, handle: str) -> dict[str, torch.Tensor] | None:
        if self.proxy:
            return None
        return self.store.get_sample(handle)

    def delete(self, handle: str) -> None:
        if self.proxy:
            return
        self.store.delete_sample(handle)


@HiddenStatesBackend.register("mooncake")
class MooncakeBackend(HiddenStatesBackend):
    """Mooncake distributed store backend (no shared filesystem required)."""

    @staticmethod
    def _add_mooncake_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--mooncake-master",
            type=str,
            default="127.0.0.1:50051",
            help="Mooncake master server address. Used with backend=mooncake.",
        )
        parser.add_argument(
            "--mooncake-metadata-server",
            type=str,
            default="P2PHANDSHAKE",
            help=(
                "Mooncake metadata server (or P2PHANDSHAKE). "
                "Used with backend=mooncake."
            ),
        )
        parser.add_argument(
            "--mooncake-protocol",
            choices=["tcp", "rdma", "ascend"],
            default="tcp",
            help="Mooncake transport protocol. Used with backend=mooncake.",
        )
        parser.add_argument(
            "--mooncake-device-name",
            type=str,
            default="",
            help="Ascend Direct/RDMA device name; empty enables auto-selection.",
        )
        parser.add_argument(
            "--mooncake-global-segment-gib",
            type=float,
            default=4.0,
            help=(
                "Memory registered by each Mooncake client for globally visible "
                "objects, in GiB. Increase for many concurrent long sequences."
            ),
        )
        parser.add_argument(
            "--mooncake-local-buffer-gib",
            type=float,
            default=2.0,
            help="Mooncake client's local staging buffer, in GiB.",
        )
        parser.add_argument(
            "--mooncake-adxl-pool-slots",
            type=int,
            default=2,
            help="Number of pre-registered NPU buffers used by Ascend Direct.",
        )
        parser.add_argument(
            "--mooncake-adxl-slot-mib",
            type=int,
            default=512,
            help="Size of each pre-registered Ascend Direct buffer in MiB.",
        )

    @staticmethod
    def add_train_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)

    @staticmethod
    def add_launch_args(parser: argparse.ArgumentParser) -> None:
        MooncakeBackend._add_mooncake_args(parser)
        parser.add_argument(
            "--mooncake-writer-threads",
            type=int,
            default=4,
            help="Number of asynchronous Mooncake writer threads in the vLLM client.",
        )

    @staticmethod
    def from_train_args(
        args: argparse.Namespace,
        data_path: str,  # noqa: ARG004
    ) -> MooncakeTransfer:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        store = MooncakeHiddenStatesStore(
            MooncakeStoreConfig(
                local_hostname=local_hostname,
                metadata_server=args.mooncake_metadata_server,
                master_server_address=args.mooncake_master,
                global_segment_size=round(args.mooncake_global_segment_gib * 1024**3),
                local_buffer_size=round(args.mooncake_local_buffer_gib * 1024**3),
                protocol=args.mooncake_protocol,
                device_name=args.mooncake_device_name,
                adxl_pool_slots=getattr(args, "mooncake_adxl_pool_slots", 2),
                adxl_slot_bytes=getattr(args, "mooncake_adxl_slot_mib", 512) * 1024**2,
            )
        )
        return MooncakeTransfer(store, proxy=adxl_proxy_enabled())

    @staticmethod
    def build_kv_transfer_config(args: argparse.Namespace) -> dict[str, Any]:
        local_hostname = os.environ.get(
            "MOONCAKE_LOCAL_HOSTNAME"
        ) or socket.gethostbyname(socket.gethostname())

        mooncake_cfg = MooncakeStoreConfig(
            local_hostname=local_hostname,
            metadata_server=args.mooncake_metadata_server,
            master_server_address=args.mooncake_master,
            global_segment_size=round(args.mooncake_global_segment_gib * 1024**3),
            local_buffer_size=round(args.mooncake_local_buffer_gib * 1024**3),
            protocol=args.mooncake_protocol,
            device_name=args.mooncake_device_name,
            num_writer_threads=args.mooncake_writer_threads,
            adxl_pool_slots=getattr(args, "mooncake_adxl_pool_slots", 2),
            adxl_slot_bytes=getattr(args, "mooncake_adxl_slot_mib", 512) * 1024**2,
        )

        return {
            "kv_connector": "MooncakeHiddenStatesConnector",
            "kv_role": "kv_producer",
            "kv_connector_module_path": (
                "hs_connectors.mooncake_hidden_states_connector"
            ),
            "kv_connector_extra_config": {
                "mooncake": dataclasses.asdict(mooncake_cfg),
            },
        }

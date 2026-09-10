"""Mooncake-backed store for hidden states, keyed by request id.

The file backend (``ExampleHiddenStatesConnector``) needs the vLLM target and
the trainer to share a filesystem; this stores the same
``{"hidden_states", "token_ids"}`` payload in a Mooncake store instead, so they
can run on different nodes.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch

logger = logging.getLogger(__name__)

_MANIFEST_VERSION = 1


class MooncakeIntegrityError(RuntimeError):
    """A Mooncake object was present but failed an integrity check."""


class NonFiniteTensorError(MooncakeIntegrityError):
    """A producer attempted to publish a tensor containing NaN or infinity."""


def _cpu_contiguous(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().to("cpu").contiguous()


def _tensor_checksum(tensor: torch.Tensor) -> str:
    """Return a fast checksum without copying the contiguous CPU tensor."""
    byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
    checksum = zlib.crc32(memoryview(byte_view)) & 0xFFFFFFFF
    return f"{checksum:08x}"


def assert_finite(name: str, tensor: torch.Tensor) -> None:
    """Raise ``NonFiniteTensorError`` if ``tensor`` holds NaN or infinity.

    ``amin``/``amax`` propagate NaN and surface either infinity, so the whole
    tensor is screened in two reductions with no full-size boolean temporary.
    Call this while the data is still on the accelerator: on the host it is an
    extra pass over the whole sample on the critical path of a producer write.
    """
    if not tensor.is_floating_point() or tensor.numel() == 0:
        return

    bounds = torch.stack((tensor.amin(), tensor.amax()))
    if bool(torch.isfinite(bounds).all()):
        return

    # Only pay for the exact counts once we already know the sample is bad.
    nan_count = int(torch.isnan(tensor).sum().item())
    inf_count = int(torch.isinf(tensor).sum().item())
    raise NonFiniteTensorError(
        f"Non-finite producer tensor {name!r}: shape={tuple(tensor.shape)}, "
        f"dtype={tensor.dtype}, nan_count={nan_count}, inf_count={inf_count}"
    )


def _check_store_result(operation: str, key: str, result: Any) -> None:
    """Mooncake's Python API returns negative status codes for some failures."""
    if isinstance(result, int) and result != 0:
        raise RuntimeError(
            f"Mooncake {operation} failed for key={key} with status={result}"
        )


def _dtype_from_str(s: str) -> torch.dtype:
    s = s.replace("torch.", "")
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    try:
        return mapping[s]
    except KeyError as exc:
        raise MooncakeIntegrityError(f"Unsupported tensor dtype in manifest: {s!r}") from exc


def _shape_nbytes(shape: tuple[int, ...], dtype: torch.dtype) -> int:
    numel = 1
    for d in shape:
        if d < 0:
            raise ValueError(f"tensor shape dimensions must be non-negative, got {shape}")
        numel *= d
    return numel * torch.empty((), dtype=dtype).element_size()


@dataclass
class MooncakeStoreConfig:
    """Connection settings, passed straight to ``MooncakeDistributedStore.setup``."""

    local_hostname: str = "localhost"
    metadata_server: str = "P2PHANDSHAKE"
    master_server_address: str = "127.0.0.1:50051"
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 2 * 1024 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    num_writer_threads: int = 4
    adxl_pool_slots: int = 2
    adxl_slot_bytes: int = 512 * 1024 * 1024

    @classmethod
    def from_dict(cls, d: dict | None) -> MooncakeStoreConfig:
        d = d or {}
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = set(d) - known
        if unknown:
            logger.warning("Unknown MooncakeStoreConfig keys ignored: %s", unknown)
        return cls(**{k: v for k, v in d.items() if k in known})


class _AdxlPool:
    """Fixed-size registered NPU buffers shared by Mooncake ADXL transfers.

    Registration is deliberately done once in :meth:`setup`.  A slot remains
    registered for the lifetime of the store; put/get only acquire a slot and
    submit a transfer using its already-registered pointer.
    """

    def __init__(self, store: Any, num_slots: int, slot_bytes: int):
        if num_slots <= 0 or slot_bytes <= 0:
            raise ValueError("ADXL pool slots and slot bytes must be positive")
        self._store = store
        self._num_slots = num_slots
        self._slot_bytes = slot_bytes
        self._slots: list[torch.Tensor] = []
        self._free: list[int] = []
        self._condition = threading.Condition()
        self._device: torch.device | None = None

    @property
    def slot_bytes(self) -> int:
        return self._slot_bytes

    @property
    def is_setup(self) -> bool:
        return bool(self._slots)

    def setup(self, device: torch.device) -> None:
        if self.is_setup:
            return
        slots: list[torch.Tensor] = []
        registered: list[int] = []
        try:
            for _ in range(self._num_slots):
                slot = torch.empty(self._slot_bytes, dtype=torch.uint8, device=device)
                result = self._store.register_buffer(slot.data_ptr(), self._slot_bytes)
                _check_store_result("register_buffer", "<adxl-pool>", result)
                slots.append(slot)
                registered.append(slot.data_ptr())
        except Exception:
            for ptr in registered:
                try:
                    self._store.unregister_buffer(ptr)
                except Exception:
                    logger.exception("Failed to unregister ADXL pool buffer")
            raise
        self._slots = slots
        self._free = list(range(len(slots)))
        self._device = device

    def close(self) -> None:
        """Unregister pool buffers. Best effort; native handles may be process-owned."""
        for slot in self._slots:
            try:
                _check_store_result(
                    "unregister_buffer",
                    "<adxl-pool>",
                    self._store.unregister_buffer(slot.data_ptr()),
                )
            except Exception:
                logger.exception("Failed to unregister ADXL pool buffer")
        self._slots.clear()
        self._free.clear()
        self._device = None

    @contextmanager
    def acquire(self, nbytes: int):
        if nbytes < 0 or nbytes > self._slot_bytes:
            raise ValueError(
                f"ADXL tensor requires {nbytes} bytes, pool slot is {self._slot_bytes} bytes"
            )
        with self._condition:
            while not self._free:
                self._condition.wait()
            index = self._free.pop()
        try:
            yield self._slots[index]
        finally:
            with self._condition:
                self._free.append(index)
                self._condition.notify()

    def put(self, key: str, tensor: torch.Tensor) -> None:
        nbytes = tensor.numel() * tensor.element_size()
        with self.acquire(nbytes) as slot:
            source = tensor.view(torch.uint8).reshape(-1)
            slot[:nbytes].copy_(source, non_blocking=False)
            result = self._store.batch_put_from([key], [slot.data_ptr()], [nbytes])
            if len(result) != 1:
                raise RuntimeError(
                    f"batch_put_from returned {len(result)} results for {key}"
                )
            _check_store_result("batch_put_from", key, result[0])

    def get(
        self,
        key: str,
        nbytes: int,
        dtype: torch.dtype,
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        with self.acquire(nbytes) as slot:
            result = self._store.batch_get_into([key], [slot.data_ptr()], [nbytes])
            if len(result) != 1 or int(result[0]) != nbytes:
                raise MooncakeIntegrityError(
                    f"Mooncake batch_get_into failed for {key}: expected {nbytes}, got {result}"
                )
            raw = slot[:nbytes].clone().to("cpu")
        return raw.view(dtype).reshape(shape)


class MooncakeHiddenStatesStore:
    """Stores/loads tensor dicts in a Mooncake store.

    Each sample is written via ``put_tensor`` under ``{key}:{name}`` plus a
    versioned ``{key}:meta`` JSON manifest. The manifest includes shape, dtype,
    and CRC32 for every tensor and is written last, so its presence marks the
    sample complete and ``get_sample`` can poll for it.
    """

    def __init__(self, config: MooncakeStoreConfig):
        self.config = config
        self._store = None
        self._engine = None
        self._owner_pid: int | None = None
        self._register_lock = threading.Lock()
        self._adxl_pool: _AdxlPool | None = None

    @property
    def is_setup(self):
        return self._store is not None and self._owner_pid == os.getpid()

    def setup(self, device: torch.device | None = None) -> MooncakeHiddenStatesStore:
        pid = os.getpid()
        if self._store is not None and self._owner_pid == pid:
            return self
        if self._store is not None and self._owner_pid != pid:
            # Native Mooncake handles must never be reused after fork.
            self._store = None
            self._engine = None
            self._adxl_pool = None
        try:
            from mooncake.engine import (  # type: ignore[import-not-found] # noqa: PLC0415
                TransferEngine,
            )
            from mooncake.store import (  # type: ignore[import-not-found] # noqa: PLC0415
                MooncakeDistributedStore,
            )
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ImportError(
                "Mooncake is required for the Mooncake hidden-states backend. "
                "Install a transfer-engine package compatible with the target "
                "accelerator, such as `mooncake-transfer-engine-npu` on Ascend."
            ) from e

        engine = TransferEngine()
        result = engine.initialize(
            self.config.local_hostname,
            self.config.metadata_server,
            self.config.protocol,
            self.config.device_name,
        )
        _check_store_result("engine setup", self.config.local_hostname, result)

        local_segment = f"{self.config.local_hostname}:{engine.get_rpc_port()}"
        store = MooncakeDistributedStore()
        result = store.setup(
            local_hostname=local_segment,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
            engine=engine.get_engine(),
        )
        _check_store_result("setup", local_segment, result)
        # MooncakeDistributedStore retains only the native engine handle.
        self._engine = engine
        self._store = store
        self._owner_pid = pid
        if self.config.protocol == "ascend":
            device = device or self._current_npu_device()
            if device.type != "npu":
                raise ValueError(f"Ascend ADXL pool requires an NPU device, got {device}")
            self._adxl_pool = _AdxlPool(
                store, self.config.adxl_pool_slots, self.config.adxl_slot_bytes
            )
            self._adxl_pool.setup(device)
        return self

    @staticmethod
    def _current_npu_device() -> torch.device:
        try:
            import torch_npu  # type: ignore[import-not-found] # noqa: PLC0415

            return torch.device("npu", torch_npu.npu.current_device())
        except (ImportError, AttributeError) as exc:
            raise RuntimeError("Ascend Mooncake requires torch_npu and an active NPU") from exc

    @contextmanager
    def _npu_context(self):
        if self.config.protocol != "ascend":
            yield
            return
        import torch_npu  # type: ignore[import-not-found] # noqa: PLC0415

        previous = torch_npu.npu.current_device()
        target = (
            self._adxl_pool._device
            if self._adxl_pool is not None
            else torch.device("npu", previous)
        )
        torch_npu.npu.set_device(target)
        try:
            yield
        finally:
            torch_npu.npu.set_device(previous)

    def put_sample(self, key: str, tensors: dict[str, torch.Tensor]) -> None:
        if self._store is None:
            raise RuntimeError("call setup() first")

        # Keep accelerator tensors resident for Ascend Direct.  Host tensors
        # continue to use the portable put_tensor path (for example token ids).
        prepared = {
            name: (
                tensor.detach().contiguous()
                if self.config.protocol == "ascend" and tensor.device.type == "npu"
                else _cpu_contiguous(tensor)
            )
            for name, tensor in tensors.items()
        }
        manifest_tensors: dict[str, dict[str, Any]] = {}
        for name, tensor in prepared.items():
            checksum_tensor = _cpu_contiguous(tensor)
            manifest_tensors[name] = {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "nbytes": _shape_nbytes(tuple(tensor.shape), tensor.dtype),
                "checksum": _tensor_checksum(checksum_tensor),
            }

        written_keys: list[str] = []
        meta_key = f"{key}:meta"
        try:
            for name, tensor in prepared.items():
                tensor_key = f"{key}:{name}"
                if self.config.protocol == "ascend" and tensor.device.type == "npu":
                    if not tensor.is_contiguous():
                        raise ValueError(
                            f"Ascend Direct tensor {name!r} must be contiguous"
                        )
                    if self._adxl_pool is None:
                        raise RuntimeError("ADXL pool was not initialized")
                    with self._npu_context(), self._register_lock:
                        self._adxl_pool.put(tensor_key, tensor)
                    result = 0
                    operation = "batch_put_from"
                else:
                    result = self._store.put_tensor(tensor_key, tensor)
                    operation = "put_tensor"
                written_keys.append(tensor_key)
                _check_store_result(operation, tensor_key, result)

            manifest = {
                "version": _MANIFEST_VERSION,
                "status": "ok",
                "tensors": manifest_tensors,
            }
            result = self._store.put(meta_key, json.dumps(manifest).encode("utf-8"))
            _check_store_result("put", meta_key, result)
        except Exception:
            # Do not leave partially published tensor objects behind. In particular,
            # never publish the completion manifest after a negative put status.
            cleanup_keys = [*written_keys, meta_key]
            if cleanup_keys:
                try:
                    self._store.batch_remove(cleanup_keys, force=True)
                except Exception:  # pragma: no cover - best-effort cleanup
                    logger.exception("Failed to clean partial Mooncake sample %s", key)
            raise

    def put_error(self, key: str, error: str) -> None:
        """Publish a small terminal marker so consumers fail fast and can retry."""
        if self._store is None:
            raise RuntimeError("call setup() first")
        manifest = {
            "version": _MANIFEST_VERSION,
            "status": "error",
            "error": error[:4096],
            "tensors": {},
        }
        meta_key = f"{key}:meta"
        result = self._store.put(meta_key, json.dumps(manifest).encode("utf-8"))
        _check_store_result("put", meta_key, result)

    def delete_sample(self, key: str) -> None:
        """Remove all keys for a sample from the store."""
        if self._store is None:
            raise RuntimeError("call setup() first")
        raw = self._store.get(f"{key}:meta")
        if not raw:
            return
        try:
            manifest = json.loads(raw)
            names = list(manifest.get("tensors", {}))
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError, TypeError):
            logger.warning(
                "Corrupt manifest for key=%s; falling back to default tensor names", key
            )
            names = ["hidden_states", "token_ids"]
        keys_to_remove = [f"{key}:{name}" for name in names] + [f"{key}:meta"]
        results = self._store.batch_remove(keys_to_remove, force=True)
        for key_to_remove, status in zip(keys_to_remove, results, strict=True):
            _check_store_result("batch_remove", key_to_remove, status)

    def get_sample(
        self, key: str, timeout: float = 120.0, poll_interval: float = 0.05
    ) -> dict[str, torch.Tensor]:
        if self._store is None:
            raise RuntimeError("call setup() first")
        raw_manifest = self._wait_for(f"{key}:meta", timeout, poll_interval)
        tensor_specs = self._parse_manifest(key, raw_manifest)

        result = {}
        for name, spec in tensor_specs.items():
            expected_shape = tuple(spec.get("shape", ()))
            expected_dtype = _dtype_from_str(str(spec.get("dtype", "")))
            expected_nbytes = int(
                spec.get("nbytes", _shape_nbytes(expected_shape, expected_dtype))
            )
            if self.config.protocol == "ascend" and self._adxl_pool is not None:
                with self._npu_context(), self._register_lock:
                    tensor = self._adxl_pool.get(
                        f"{key}:{name}", expected_nbytes, expected_dtype, expected_shape
                    )
            else:
                tensor = self._store.get_tensor(f"{key}:{name}")
            if tensor is None:
                raise MooncakeIntegrityError(
                    f"Mooncake tensor unavailable for key={key}:{name}"
                )
            self._validate_tensor(key, name, tensor, spec)
            result[name] = tensor
        return result

    @staticmethod
    def _parse_manifest(key: str, raw_manifest: bytes) -> dict[str, dict[str, Any]]:
        try:
            manifest = json.loads(raw_manifest)
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise MooncakeIntegrityError(
                f"Corrupt Mooncake manifest for key={key}: {e}"
            ) from e

        if not isinstance(manifest, dict):
            raise MooncakeIntegrityError(
                f"Invalid Mooncake manifest type for key={key}: "
                f"{type(manifest).__name__}"
            )
        if manifest.get("status") == "error":
            raise MooncakeIntegrityError(
                f"Mooncake producer rejected key={key}: "
                f"{manifest.get('error', 'unknown producer error')}"
            )
        if manifest.get("version") != _MANIFEST_VERSION:
            raise MooncakeIntegrityError(
                f"Unsupported Mooncake manifest version for key={key}: "
                f"{manifest.get('version')!r}"
            )
        tensor_specs = manifest.get("tensors")
        if not isinstance(tensor_specs, dict) or not tensor_specs:
            raise MooncakeIntegrityError(
                f"Mooncake manifest has no tensors for key={key}"
            )
        return tensor_specs

    @staticmethod
    def _validate_tensor(
        key: str,
        name: str,
        tensor: torch.Tensor,
        spec: dict[str, Any],
    ) -> None:
        if not isinstance(spec, dict):
            raise MooncakeIntegrityError(
                f"Invalid tensor manifest for key={key}:{name}: "
                f"expected object, got {type(spec).__name__}"
            )
        expected_shape = tuple(spec.get("shape", ()))
        expected_dtype = spec.get("dtype")
        if tuple(tensor.shape) != expected_shape:
            raise MooncakeIntegrityError(
                f"Mooncake shape mismatch for key={key}:{name}: "
                f"expected={expected_shape}, actual={tuple(tensor.shape)}"
            )
        if str(tensor.dtype) != expected_dtype:
            raise MooncakeIntegrityError(
                f"Mooncake dtype mismatch for key={key}:{name}: "
                f"expected={expected_dtype}, actual={tensor.dtype}"
            )
        expected_checksum = spec.get("checksum")
        actual_checksum = _tensor_checksum(_cpu_contiguous(tensor))
        if actual_checksum != expected_checksum:
            raise MooncakeIntegrityError(
                f"Mooncake checksum mismatch for key={key}:{name}: "
                f"expected={expected_checksum}, actual={actual_checksum}"
            )

    def _wait_for(self, key: str, timeout: float, poll_interval: float) -> bytes:
        if self._store is None:
            raise RuntimeError("call setup() first")
        deadline = time.monotonic() + timeout
        while True:
            raw = self._store.get(key)
            if raw:
                return raw
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for Mooncake key: {key}")
            time.sleep(poll_interval)

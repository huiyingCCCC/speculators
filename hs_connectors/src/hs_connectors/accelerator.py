"""Accelerator stream helpers shared by hidden-state connectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from contextlib import AbstractContextManager


@dataclass(frozen=True)
class AcceleratorContext:
    """Stream operations bound to the device that owns the hidden-state cache."""

    module: Any
    copy_stream: Any

    @classmethod
    def for_device(cls, device: torch.device) -> AcceleratorContext:
        module = torch.get_device_module(device)
        missing = [
            name for name in ("Stream", "Event", "stream") if not hasattr(module, name)
        ]
        if missing:
            raise RuntimeError(
                f"Accelerator {device.type!r} does not support asynchronous "
                f"hidden-state copies; missing APIs: {', '.join(missing)}"
            )
        return cls(module=module, copy_stream=module.Stream(device=device))

    def record_event(self) -> Any:
        """Record completion of work already queued on the current stream."""
        event = self.module.Event()
        event.record()
        return event

    def activate_device(self) -> None:
        """Make the copy stream's device current in the calling thread."""
        if not hasattr(self.module, "set_device"):
            raise RuntimeError(
                f"Accelerator {self.copy_stream.device.type!r} cannot activate "
                "the hidden-state copy device; missing API: set_device"
            )
        self.module.set_device(self.copy_stream.device)

    def use_copy_stream(self) -> AbstractContextManager[Any]:
        return self.module.stream(self.copy_stream)

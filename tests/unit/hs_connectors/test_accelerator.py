from contextlib import nullcontext

import pytest
import torch
from hs_connectors.accelerator import AcceleratorContext


class _FakeEvent:
    def __init__(self):
        self.recorded = False

    def record(self):
        self.recorded = True


class _FakeStream:
    def __init__(self, device):
        self.device = device


class _FakeDeviceModule:
    Event = _FakeEvent
    Stream = _FakeStream

    def __init__(self):
        self.activated_stream = None

    def stream(self, stream):
        self.activated_stream = stream
        return nullcontext()


@pytest.mark.parametrize("device_type", ["cuda", "npu"])
def test_accelerator_context_uses_cache_device_module(monkeypatch, device_type):
    module = _FakeDeviceModule()
    device = torch.device(f"{device_type}:3")
    monkeypatch.setattr(torch, "get_device_module", lambda actual: module)

    context = AcceleratorContext.for_device(device)
    event = context.record_event()
    with context.use_copy_stream():
        pass

    assert context.copy_stream.device == device
    assert event.recorded
    assert module.activated_stream is context.copy_stream


def test_accelerator_context_rejects_incomplete_device_module(monkeypatch):
    monkeypatch.setattr(torch, "get_device_module", lambda _device: object())

    with pytest.raises(RuntimeError, match="missing APIs: Stream, Event, stream"):
        AcceleratorContext.for_device(torch.device("npu:0"))

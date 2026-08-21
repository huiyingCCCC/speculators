"""Manual cross-node smoke test for NPU hidden-state transfer via Mooncake."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass

import torch
from hs_connectors.mooncake_store import (
    MooncakeHiddenStatesStore,
    MooncakeStoreConfig,
)


@dataclass(frozen=True)
class PayloadSpec:
    sequence_length: int = 8
    num_layers: int = 3
    hidden_size: int = 256

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.sequence_length, self.num_layers, self.hidden_size


def build_payload(spec: PayloadSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Build data that can be reproduced independently by the consumer."""
    numel = spec.sequence_length * spec.num_layers * spec.hidden_size
    hidden_states = (
        torch.arange(numel, dtype=torch.float32).remainder(2048).div(128).sub(8)
    ).to(torch.bfloat16)
    return hidden_states.reshape(spec.shape), torch.arange(spec.sequence_length)


def build_store(args: argparse.Namespace) -> MooncakeHiddenStatesStore:
    return MooncakeHiddenStatesStore(
        MooncakeStoreConfig(
            local_hostname=args.local_ip,
            master_server_address=args.master,
            metadata_server="P2PHANDSHAKE",
            protocol="tcp",
            device_name="",
            global_segment_size=args.global_segment_gib * 1024**3,
            local_buffer_size=args.local_buffer_mib * 1024**2,
        )
    ).setup()


def activate_npu(device_name: str) -> torch.device:
    import torch_npu  # noqa: F401, PLC0415

    device = torch.device(device_name)
    torch.npu.set_device(device)
    # Force ACL context creation before Mooncake installs AscendDirectTransport.
    torch.empty(1, device=device)
    torch.npu.synchronize()
    return device


def run_producer(args: argparse.Namespace) -> None:
    spec = PayloadSpec()
    expected_hidden_states, token_ids = build_payload(spec)
    device = activate_npu(args.device)

    npu_hidden_states = expected_hidden_states.to(device)
    copy_stream = torch.npu.Stream(device=device)
    with torch.npu.stream(copy_stream):
        cpu_hidden_states = torch.empty_like(
            npu_hidden_states,
            device="cpu",
            pin_memory=True,
        )
        cpu_hidden_states.copy_(npu_hidden_states, non_blocking=True)
    copy_stream.synchronize()

    if not torch.equal(cpu_hidden_states, expected_hidden_states):
        raise RuntimeError("NPU-to-host hidden-state copy changed tensor contents")

    print("NPU to pinned CPU copy passed", flush=True)
    store = build_store(args)
    store.put_sample(
        args.key,
        {"hidden_states": cpu_hidden_states, "token_ids": token_ids},
    )
    print(f"Published key: {args.key}", flush=True)
    print("Keep this process running until the consumer passes.", flush=True)

    try:
        input("Press Enter to remove the sample and exit... ")
    except (EOFError, KeyboardInterrupt):
        print(file=sys.stderr)
    finally:
        store.delete_sample(args.key)


def run_consumer(args: argparse.Namespace) -> None:
    activate_npu(args.device)
    expected_hidden_states, expected_token_ids = build_payload(PayloadSpec())
    store = build_store(args)
    sample = store.get_sample(args.key, timeout=args.timeout)

    hidden_states = sample["hidden_states"]
    token_ids = sample["token_ids"]
    if not torch.equal(hidden_states, expected_hidden_states):
        raise RuntimeError("received hidden states differ from the producer payload")
    if not torch.equal(token_ids, expected_token_ids):
        raise RuntimeError("received token IDs differ from the producer payload")
    if not bool(torch.isfinite(hidden_states).all()):
        raise RuntimeError("received hidden states contain NaN or infinity")

    print(f"Received key: {args.key}")
    print(
        f"hidden_states: shape={tuple(hidden_states.shape)}, "
        f"dtype={hidden_states.dtype}"
    )
    print(f"token_ids: shape={tuple(token_ids.shape)}, dtype={token_ids.dtype}")
    print("Cross-node NPU hidden-state transfer PASSED")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("producer", "consumer"))
    parser.add_argument(
        "--local-ip",
        required=True,
        help="Routable IP of the machine running this process.",
    )
    parser.add_argument(
        "--master",
        default="127.0.0.1:50051",
        help="Mooncake master address.",
    )
    parser.add_argument("--key", default="npu-hidden-state-cross-node-smoke")
    parser.add_argument("--global-segment-gib", type=int, default=1)
    parser.add_argument("--local-buffer-mib", type=int, default=128)
    parser.add_argument(
        "--device",
        default="npu:0",
        help="NPU device to activate before Mooncake setup.",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "producer":
        run_producer(args)
    else:
        run_consumer(args)


if __name__ == "__main__":
    main()

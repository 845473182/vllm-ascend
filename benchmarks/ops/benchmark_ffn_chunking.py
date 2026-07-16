# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark routed MoE FFN token chunking.

Compares no chunk, the production shape-aware heuristic, fixed two chunks,
and fixed four chunks. ``FFN latency`` measures only expert MLP computation on
an already dispatched expert-sorted buffer. ``forward latency`` additionally
includes a deterministic token expansion/permutation and output restoration;
it intentionally does not benchmark distributed MoE communication.

Examples::

    python benchmarks/ops/benchmark_ffn_chunking.py --device npu:0
    python benchmarks/ops/benchmark_ffn_chunking.py --device cpu --smoke
"""

from __future__ import annotations

import argparse
import csv
import gc
import math
import statistics
import time
from dataclasses import dataclass, replace
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import torch_npu
except ImportError:
    torch_npu = None

from vllm_ascend.ops.fused_moe.moe_ffn_chunking import (
    balanced_chunk_ranges,
    estimate_moe_ffn_num_chunks,
    run_moe_ffn_in_token_chunks,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)


TOKEN_COUNTS = (256, 512, 1024, 2048, 4096, 8192, 16384)
STRATEGIES = ("no_chunk", "shape_aware", "fixed_2", "fixed_4")
MIB = 1024**2


@dataclass(frozen=True)
class RoutedBatch:
    original: torch.Tensor
    routed: torch.Tensor
    group_list: torch.Tensor
    permutation: torch.Tensor
    inverse_permutation: torch.Tensor


class MoEFFNBenchmark:
    def __init__(
        self,
        *,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.device = device
        self.w1 = torch.randn(
            num_experts,
            hidden_size,
            2 * intermediate_size,
            device=device,
            dtype=dtype,
        ) / math.sqrt(hidden_size)
        self.w2 = torch.randn(
            num_experts,
            intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        ) / math.sqrt(intermediate_size)

    def make_batch(self, num_tokens: int, dtype: torch.dtype) -> RoutedBatch:
        original = torch.randn(num_tokens, self.hidden_size, device=self.device, dtype=dtype)
        expanded = original.repeat_interleave(self.top_k, dim=0)
        row_ids = torch.arange(expanded.shape[0], device=self.device)
        expert_ids = row_ids.remainder(self.num_experts)
        permutation = torch.argsort(expert_ids)
        inverse_permutation = torch.argsort(permutation)
        routed = expanded.index_select(0, permutation)
        counts = torch.bincount(expert_ids, minlength=self.num_experts).to(torch.int64)
        return RoutedBatch(original, routed, counts, permutation, inverse_permutation)

    def make_input(self, batch: RoutedBatch) -> MoEMlpComputeInput:
        return MoEMlpComputeInput(
            hidden_states=batch.routed,
            group_list=batch.group_list,
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(w1=self.w1, w2=self.w2),
            quant=MoEQuantParams(),
            fusion=False,
        )

    def apply_mlp(self, value: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        if self.device.type == "npu":
            gate_up = torch_npu.npu_grouped_matmul(
                x=[value.hidden_states],
                weight=[self.w1],
                split_item=2,
                group_list_type=value.group_list_type,
                group_type=0,
                group_list=value.group_list,
            )[0]
            gate_up = torch_npu.npu_swiglu(gate_up)
            output = torch_npu.npu_grouped_matmul(
                x=[gate_up],
                weight=[self.w2],
                split_item=2,
                group_list_type=value.group_list_type,
                group_type=0,
                group_list=value.group_list,
            )[0]
            return output, None

        counts = value.group_list.tolist()
        output = torch.empty_like(value.hidden_states)
        offset = 0
        for expert_id, count in enumerate(counts):
            expert_input = value.hidden_states[offset : offset + count]
            gate, up = (expert_input @ self.w1[expert_id]).chunk(2, dim=-1)
            output[offset : offset + count].copy_((F.silu(gate) * up) @ self.w2[expert_id])
            offset += count
        return output, None

    def run_ffn(self, value: MoEMlpComputeInput, num_chunks: int) -> torch.Tensor:
        if num_chunks == 1:
            return self.apply_mlp(value)[0]
        return run_moe_ffn_in_token_chunks(value, num_chunks=num_chunks, apply_mlp=self.apply_mlp)[0]

    def run_forward(self, batch: RoutedBatch, num_chunks: int) -> torch.Tensor:
        expanded = batch.original.repeat_interleave(self.top_k, dim=0)
        routed = expanded.index_select(0, batch.permutation)
        routed_output = self.run_ffn(replace(self.make_input(batch), hidden_states=routed), num_chunks)
        restored = routed_output.index_select(0, batch.inverse_permutation)
        return restored.reshape(batch.original.shape[0], self.top_k, self.hidden_size).mean(dim=1)


def synchronize(device: torch.device) -> None:
    if device.type == "npu":
        torch.npu.synchronize()


def latency_ms(operation, device: torch.device, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        output = operation()
        synchronize(device)
        del output
    samples = []
    for _ in range(iterations):
        synchronize(device)
        start = time.perf_counter()
        output = operation()
        synchronize(device)
        samples.append((time.perf_counter() - start) * 1000)
        del output
    return statistics.median(samples)


def peak_allocated_mib(operation, device: torch.device) -> float | None:
    if device.type != "npu":
        return None
    gc.collect()
    synchronize(device)
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()
    baseline = torch.npu.memory_allocated()
    output = operation()
    synchronize(device)
    peak = torch.npu.max_memory_allocated()
    del output
    return max(0, peak - baseline) / MIB


def strategy_chunks(args, strategy: str, routed_tokens: int) -> int:
    if strategy == "no_chunk":
        return 1
    if strategy == "shape_aware":
        return estimate_moe_ffn_num_chunks(
            num_routed_tokens=routed_tokens,
            hidden_size=args.hidden_size,
            intermediate_size=args.intermediate_size,
            experts_per_token=args.top_k,
            live_factor=args.live_factor,
            target_hidden_factor=args.target_hidden_factor,
            min_chunk_size=args.min_chunk_size,
        )
    return min(int(strategy.removeprefix("fixed_")), routed_tokens)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default=None)
    parser.add_argument("--tokens", default=",".join(map(str, TOKEN_COUNTS)))
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--min-chunk-size", type=int, default=1024)
    parser.add_argument("--live-factor", type=float, default=3.0)
    parser.add_argument("--target-hidden-factor", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.device == "auto":
        has_npu = torch_npu is not None and hasattr(torch, "npu") and torch.npu.is_available()
        args.device = "npu:0" if has_npu else "cpu"
    device = torch.device(args.device)
    if device.type == "npu":
        torch.npu.set_device(device)
    if args.smoke:
        args.tokens = "256,512,1024"
        args.hidden_size = min(args.hidden_size, 64)
        args.intermediate_size = min(args.intermediate_size, 128)
        args.num_experts = min(args.num_experts, 4)
        args.warmup = 1
        args.iterations = 1

    dtype_name = args.dtype or ("float16" if device.type == "npu" else "float32")
    dtype = getattr(torch, dtype_name)
    token_counts = tuple(int(value) for value in args.tokens.split(","))
    model = MoEFFNBenchmark(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_experts=args.num_experts,
        top_k=args.top_k,
        device=device,
        dtype=dtype,
    )

    rows: list[dict[str, object]] = []
    for num_tokens in token_counts:
        batch = model.make_batch(num_tokens, dtype)
        mlp_input = model.make_input(batch)
        reference = model.run_ffn(mlp_input, 1)
        synchronize(device)
        for strategy in STRATEGIES:
            num_chunks = strategy_chunks(args, strategy, batch.routed.shape[0])
            sizes = [end - start for start, end in balanced_chunk_ranges(batch.routed.shape[0], num_chunks)]
            candidate = model.run_ffn(mlp_input, num_chunks)
            synchronize(device)
            max_error = float((candidate.float() - reference.float()).abs().max().item())
            ffn_operation = lambda: model.run_ffn(mlp_input, num_chunks)
            forward_operation = lambda: model.run_forward(batch, num_chunks)
            row = {
                "M": num_tokens,
                "routed_tokens": batch.routed.shape[0],
                "strategy": strategy,
                "num_chunks": num_chunks,
                "chunk_sizes": ";".join(map(str, sizes)),
                "ffn_latency_ms": latency_ms(ffn_operation, device, args.warmup, args.iterations),
                "forward_latency_ms": latency_ms(forward_operation, device, args.warmup, args.iterations),
                "ffn_peak_delta_mib": peak_allocated_mib(ffn_operation, device),
                "forward_peak_delta_mib": peak_allocated_mib(forward_operation, device),
                "max_abs_error": max_error,
            }
            rows.append(row)
            print(row)
            del candidate
        del reference, mlp_input, batch

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()

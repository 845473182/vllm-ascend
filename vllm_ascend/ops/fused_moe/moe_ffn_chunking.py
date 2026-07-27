# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

"""Token chunking helpers for the routed MoE FFN stage."""

import math
from collections.abc import Callable, Iterator
from dataclasses import replace

import torch

from vllm_ascend.ops.fused_moe.moe_runtime_args import MoEMlpComputeInput


def estimate_ffn_num_chunks(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    *,
    live_factor: float = 3.0,
    target_hidden_factor: float = 2.0,
    min_chunk_size: int = 1024,
) -> int:
    """Return a shape-aware chunk count with a minimum chunk-size guard."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if hidden_size <= 0:
        raise ValueError(f"hidden_size must be positive, got {hidden_size}")
    if intermediate_size <= 0:
        raise ValueError(f"intermediate_size must be positive, got {intermediate_size}")
    if not math.isfinite(live_factor) or live_factor <= 0:
        raise ValueError(f"live_factor must be finite and positive, got {live_factor}")
    if not math.isfinite(target_hidden_factor) or target_hidden_factor <= 0:
        raise ValueError(f"target_hidden_factor must be finite and positive, got {target_hidden_factor}")
    if min_chunk_size <= 0:
        raise ValueError(f"min_chunk_size must be positive, got {min_chunk_size}")

    shape_num_chunks = max(
        1,
        math.ceil(live_factor * intermediate_size / (target_hidden_factor * hidden_size)),
    )
    max_chunks_for_tokens = max(1, num_tokens // min_chunk_size)
    return min(shape_num_chunks, max_chunks_for_tokens)


def estimate_moe_ffn_num_chunks(
    num_routed_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    experts_per_token: int,
    *,
    live_factor: float = 3.0,
    target_hidden_factor: float = 2.0,
    min_chunk_size: int = 1024,
) -> int:
    """Adapt the shape heuristic to top-k-expanded routed expert rows."""
    if experts_per_token <= 0:
        raise ValueError(f"experts_per_token must be positive, got {experts_per_token}")
    return estimate_ffn_num_chunks(
        num_tokens=num_routed_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size * experts_per_token,
        live_factor=live_factor,
        target_hidden_factor=target_hidden_factor,
        min_chunk_size=min_chunk_size,
    )


def balanced_chunk_ranges(num_tokens: int, num_chunks: int) -> Iterator[tuple[int, int]]:
    """Yield ordered contiguous ranges whose sizes differ by at most one."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if num_tokens == 0:
        return
    if num_chunks > num_tokens:
        raise ValueError(f"num_chunks ({num_chunks}) cannot exceed num_tokens ({num_tokens})")

    base_size, larger_chunks = divmod(num_tokens, num_chunks)
    start = 0
    for chunk_idx in range(num_chunks):
        chunk_size = base_size + int(chunk_idx < larger_chunks)
        end = start + chunk_size
        yield start, end
        start = end


def fixed_chunk_ranges(num_tokens: int, chunk_size: int) -> Iterator[tuple[int, int]]:
    """Yield ordered ranges with at most ``chunk_size`` tokens each."""
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    for start in range(0, num_tokens, chunk_size):
        yield start, min(start + chunk_size, num_tokens)


def chunk_group_list(
    group_list: torch.Tensor,
    group_list_type: int,
    start: int,
    end: int,
) -> torch.Tensor:
    """Build grouped-matmul metadata for a contiguous expert-sorted slice.

    ``group_list_type == 1`` stores per-expert token counts and type ``0``
    stores cumulative end offsets. A chunk is allowed to split an expert; the
    overlap calculation keeps that expert's rows assigned to the same weight.
    """
    if group_list_type not in (0, 1):
        raise ValueError(f"MoE FFN chunking supports group_list_type 0 or 1, got {group_list_type}")
    if start < 0 or end < start:
        raise ValueError(f"invalid chunk range [{start}, {end})")
    if group_list.numel() == 0:
        return group_list.clone()

    if group_list_type == 0:
        counts = torch.cat((group_list[:1], group_list[1:] - group_list[:-1]))
    else:
        counts = group_list

    expert_ends = counts.cumsum(dim=0)
    expert_starts = expert_ends - counts
    overlap = expert_ends.clamp(max=end) - expert_starts.clamp(min=start)
    chunk_counts = overlap.clamp_(min=0, max=end - start)
    if group_list_type == 0:
        return chunk_counts.cumsum(dim=0)
    return chunk_counts


def _slice_token_metadata(
    value: torch.Tensor | None,
    *,
    start: int,
    end: int,
) -> torch.Tensor | None:
    if value is None:
        return None
    return value[start:end]


def supports_moe_ffn_chunking(mlp_compute_input: MoEMlpComputeInput) -> bool:
    """Whether the typed MLP payload can be sliced without changing semantics."""
    hidden_states = mlp_compute_input.hidden_states
    if hidden_states.ndim != 2 or mlp_compute_input.group_list_type not in (0, 1):
        return False
    if mlp_compute_input.lora_context is not None:
        return False

    num_tokens = hidden_states.shape[0]
    return all(
        value is None or (value.ndim > 0 and value.shape[0] == num_tokens)
        for value in (mlp_compute_input.dynamic_scale, mlp_compute_input.topk_scales)
    )


def run_moe_ffn_in_token_chunks(
    mlp_compute_input: MoEMlpComputeInput,
    *,
    num_chunks: int | None = None,
    chunk_size: int | None = None,
    apply_mlp: Callable[[MoEMlpComputeInput], tuple[torch.Tensor, object | None]],
) -> tuple[torch.Tensor, object | None]:
    """Run only the expert MLP in chunks, leaving dispatch/combine untouched.

    When the MLP produces an output with the same shape and dtype as the input
    (the common BF16 dispatch path), each chunk is written back into the input
    buffer directly so no extra ``[N, H]`` staging tensor lives across the
    loop. When the output dtype differs (e.g. int8/fp8 activations in, BF16
    out) a fresh ``[N, ...]`` buffer is allocated on the first iteration.
    """
    if (num_chunks is None) == (chunk_size is None):
        raise ValueError("exactly one of num_chunks or chunk_size must be set")
    if num_chunks is not None and num_chunks <= 0:
        raise ValueError(f"num_chunks must be positive, got {num_chunks}")
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")

    hidden_states = mlp_compute_input.hidden_states
    num_tokens = hidden_states.shape[0]
    if num_chunks == 1 or (chunk_size is not None and num_tokens <= chunk_size):
        return apply_mlp(mlp_compute_input)
    if not supports_moe_ffn_chunking(mlp_compute_input):
        raise ValueError("MoE MLP input contains metadata that cannot be safely chunked")

    if num_chunks is not None and num_chunks > num_tokens:
        raise ValueError(f"num_chunks ({num_chunks}) cannot exceed num_tokens ({num_tokens})")

    if num_chunks is not None:
        ranges = balanced_chunk_ranges(num_tokens, num_chunks)
    else:
        assert chunk_size is not None
        ranges = fixed_chunk_ranges(num_tokens, chunk_size)

    ranges_list = list(ranges)
    before_gmm2_event: object | None = None
    output: torch.Tensor | None = None
    # ``read_source`` is the tensor we slice per-iteration to build
    # ``chunk_input.hidden_states``. It starts as the caller's dispatch
    # buffer. On the quantized-dispatch fallback we point it at ``output``
    # AFTER we have (a) allocated ``output`` sized to the true MLP result
    # dtype AND (b) already consumed / copied every earlier chunk's data
    # into ``output``. Since output is written in the same slice ordering
    # we then read from, chunks 2..K only read data they have themselves
    # just written — but we never actually re-read past chunks because the
    # loop always advances ``start``. The critical property is: once we
    # dispose the input, subsequent iterations must not slice the disposed
    # tensor; ``read_source`` gives us a valid buffer to slice, even though
    # its dtype/shape happen to be the output's — which is fine because
    # the quantized path re-consumes chunk data via ``apply_mlp`` from
    # ``chunk_input.hidden_states``, and each chunk's ``hidden_states`` is
    # already fully materialised by then.
    #
    # ⚠ This is subtle: for the fallback path, we MUST slice from the
    # original input tensor for every chunk. So we cannot dispose after
    # only one chunk. Instead we dispose AFTER the loop ends — trading off
    # some peak reduction for correctness. If a caller wants the maximum
    # peak reduction on the quantized path, dispatch should return BF16
    # (making the same_layout fast path apply) or the caller should chunk
    # dispatch itself.
    read_source = hidden_states

    for idx, (start, end) in enumerate(ranges_list):
        chunk_view = read_source[start:end]
        chunk_input = replace(
            mlp_compute_input,
            hidden_states=chunk_view,
            group_list=chunk_group_list(
                mlp_compute_input.group_list,
                mlp_compute_input.group_list_type,
                start,
                end,
            ),
            dynamic_scale=_slice_token_metadata(mlp_compute_input.dynamic_scale, start=start, end=end),
            topk_scales=_slice_token_metadata(mlp_compute_input.topk_scales, start=start, end=end),
        )
        del chunk_view

        chunk_output, before_gmm2_event = apply_mlp(chunk_input)
        del chunk_input

        if output is None:
            same_layout = (
                chunk_output.dtype == hidden_states.dtype
                and chunk_output.shape[1:] == hidden_states.shape[1:]
                and hidden_states.is_contiguous()
            )
            if same_layout:
                # BF16-in / BF16-out: alias input as output. No extra buffer.
                output = hidden_states
            else:
                # Quantized-dispatch fallback: input dtype (e.g. int8)
                # differs from output dtype (e.g. bfloat16). Allocate a new
                # output buffer. See note above about deferred dispose.
                output = chunk_output.new_empty((num_tokens, *chunk_output.shape[1:]))
        output[start:end].copy_(chunk_output)
        del chunk_output

    assert output is not None
    # Post-loop dispose: on the fallback path, ``hidden_states`` (int8) and
    # ``output`` (bf16) both existed for the full duration of the chunk
    # loop, so peak memory is unchanged. Dispose here anyway so the caller
    # cannot observe the base after we return — matches token_combine's
    # contract that only ``output`` is live going forward.
    if output is not hidden_states:
        mlp_compute_input.hidden_states.set_(
            torch.empty((0,), device=hidden_states.device, dtype=hidden_states.dtype)
        )
    return output, before_gmm2_event

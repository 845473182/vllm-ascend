# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_ffn_chunking import (
    balanced_chunk_ranges,
    chunk_group_list,
    estimate_ffn_num_chunks,
    estimate_moe_ffn_num_chunks,
    run_moe_ffn_in_token_chunks,
    supports_moe_ffn_chunking,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)


def _estimate(num_tokens: int, **overrides: int | float) -> int:
    values: dict[str, int | float] = {
        "num_tokens": num_tokens,
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "live_factor": 3.0,
        "target_hidden_factor": 2.0,
        "min_chunk_size": 1024,
    }
    values.update(overrides)
    return estimate_ffn_num_chunks(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("num_tokens", "expected_chunks"),
    [
        (256, 1),
        (1024, 1),
        (2048, 2),
        (3072, 3),
        (4096, 4),
        (4097, 4),
        (8192, 5),
        (16384, 5),
    ],
)
def test_estimate_ffn_num_chunks(num_tokens: int, expected_chunks: int) -> None:
    assert _estimate(num_tokens) == expected_chunks


def test_min_chunk_size_can_limit_shape_target_to_one() -> None:
    assert _estimate(2047, hidden_size=1024, intermediate_size=8192, min_chunk_size=2048) == 1


def test_moe_shape_estimate_accounts_for_topk_expansion() -> None:
    assert estimate_moe_ffn_num_chunks(
        num_routed_tokens=8192,
        hidden_size=7168,
        intermediate_size=2048,
        experts_per_token=8,
        min_chunk_size=1024,
    ) == 4


@pytest.mark.parametrize(
    ("num_tokens", "num_chunks", "expected_sizes"),
    [
        (4096, 2, [2048, 2048]),
        (4096, 3, [1366, 1365, 1365]),
        (4100, 3, [1367, 1367, 1366]),
        (16384, 5, [3277, 3277, 3277, 3277, 3276]),
    ],
)
def test_balanced_chunk_ranges(num_tokens: int, num_chunks: int, expected_sizes: list[int]) -> None:
    ranges = list(balanced_chunk_ranges(num_tokens, num_chunks))
    sizes = [end - start for start, end in ranges]

    assert sizes == expected_sizes
    assert ranges[0][0] == 0
    assert ranges[-1][1] == num_tokens
    assert all(left[1] == right[0] for left, right in zip(ranges, ranges[1:]))
    assert max(sizes) - min(sizes) <= 1


@pytest.mark.parametrize(
    ("group_list", "group_list_type", "start", "end", "expected"),
    [
        ([3, 2, 4], 1, 0, 4, [3, 1, 0]),
        ([3, 2, 4], 1, 4, 9, [0, 1, 4]),
        ([3, 5, 9], 0, 0, 4, [3, 4, 4]),
        ([3, 5, 9], 0, 4, 9, [0, 1, 5]),
    ],
)
def test_chunk_group_list(
    group_list: list[int],
    group_list_type: int,
    start: int,
    end: int,
    expected: list[int],
) -> None:
    actual = chunk_group_list(torch.tensor(group_list), group_list_type, start, end)
    torch.testing.assert_close(actual, torch.tensor(expected))


def _make_mlp_input(
    hidden_states: torch.Tensor,
    *,
    group_list: torch.Tensor,
    group_list_type: int = 1,
    dynamic_scale: torch.Tensor | None = None,
    topk_scales: torch.Tensor | None = None,
    lora_context=None,
) -> MoEMlpComputeInput:
    return MoEMlpComputeInput(
        hidden_states=hidden_states,
        group_list=group_list,
        group_list_type=group_list_type,
        dynamic_scale=dynamic_scale,
        topk_scales=topk_scales,
        weights=MoEWeights(w1=torch.empty(0), w2=torch.empty(0)),
        quant=MoEQuantParams(),
        fusion=False,
        lora_context=lora_context,
    )


def _expert_tagged_mlp(mlp_input: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
    counts = mlp_input.group_list
    if mlp_input.group_list_type == 0:
        counts = torch.cat((counts[:1], counts[1:] - counts[:-1]))
    expert_ids = torch.repeat_interleave(torch.arange(counts.numel()), counts)
    return mlp_input.hidden_states + expert_ids[:, None] * 100, None


@pytest.mark.parametrize("group_list_type", [0, 1])
def test_chunked_moe_ffn_matches_unchunked_and_preserves_expert_assignment(group_list_type: int) -> None:
    hidden_states = torch.arange(90, dtype=torch.float32).reshape(9, 10)
    counts = torch.tensor([3, 2, 4])
    group_list = counts.cumsum(0) if group_list_type == 0 else counts
    mlp_input = _make_mlp_input(hidden_states, group_list=group_list, group_list_type=group_list_type)

    expected, _ = _expert_tagged_mlp(mlp_input)
    actual, _ = run_moe_ffn_in_token_chunks(mlp_input, num_chunks=2, apply_mlp=_expert_tagged_mlp)

    assert actual.shape == hidden_states.shape
    torch.testing.assert_close(actual, expected)


def test_chunked_moe_ffn_slices_per_token_metadata() -> None:
    hidden_states = torch.randn(9, 4)
    dynamic_scale = torch.arange(9, dtype=torch.float32).reshape(9, 1)
    topk_scales = torch.arange(9, dtype=torch.float32)
    mlp_input = _make_mlp_input(
        hidden_states,
        group_list=torch.tensor([3, 2, 4]),
        dynamic_scale=dynamic_scale,
        topk_scales=topk_scales,
    )
    seen: list[tuple[torch.Tensor, torch.Tensor]] = []

    def apply_mlp(chunk_input: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        assert chunk_input.dynamic_scale is not None
        assert chunk_input.topk_scales is not None
        seen.append((chunk_input.dynamic_scale.clone(), chunk_input.topk_scales.clone()))
        return chunk_input.hidden_states, None

    run_moe_ffn_in_token_chunks(mlp_input, num_chunks=2, apply_mlp=apply_mlp)

    torch.testing.assert_close(torch.cat([item[0] for item in seen]), dynamic_scale)
    torch.testing.assert_close(torch.cat([item[1] for item in seen]), topk_scales)


def test_chunked_moe_ffn_preallocates_output_with_actual_result_dtype() -> None:
    hidden_states = torch.randint(-4, 4, (17, 8), dtype=torch.int8)
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([5, 6, 6]))

    def bf16_output(value: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        return value.hidden_states.to(torch.bfloat16), None

    output, _ = run_moe_ffn_in_token_chunks(mlp_input, num_chunks=4, apply_mlp=bf16_output)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, hidden_states.to(torch.bfloat16))


def test_single_chunk_uses_original_mlp_payload() -> None:
    hidden_states = torch.randn(4, 8)
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([2, 2]))
    seen: list[MoEMlpComputeInput] = []

    def apply_mlp(value: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        seen.append(value)
        return value.hidden_states, None

    output, _ = run_moe_ffn_in_token_chunks(mlp_input, num_chunks=1, apply_mlp=apply_mlp)

    assert seen == [mlp_input]
    assert output is hidden_states


def test_unsupported_metadata_is_bypassed() -> None:
    hidden_states = torch.randn(4, 8)
    bad_scale = torch.randn(3, 1)
    assert not supports_moe_ffn_chunking(
        _make_mlp_input(hidden_states, group_list=torch.tensor([2, 2]), dynamic_scale=bad_scale)
    )
    assert not supports_moe_ffn_chunking(
        _make_mlp_input(hidden_states, group_list=torch.tensor([2, 2]), lora_context=object())
    )
    assert not supports_moe_ffn_chunking(
        _make_mlp_input(hidden_states, group_list=torch.tensor([[0, 2]]), group_list_type=2)
    )

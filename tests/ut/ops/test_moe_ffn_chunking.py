# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.ops.fused_moe.moe_ffn_chunking import (
    balanced_chunk_ranges,
    chunk_group_list,
    fixed_chunk_ranges,
    run_moe_ffn_in_token_chunks,
    supports_moe_ffn_chunking,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEMlpComputeInput,
    MoEQuantParams,
    MoEWeights,
)


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
    ("num_tokens", "chunk_size", "expected_sizes"),
    [
        (32768, 32768, [32768]),
        (65536, 32768, [32768, 32768]),
        (70000, 32768, [32768, 32768, 4464]),
    ],
)
def test_fixed_chunk_ranges(num_tokens: int, chunk_size: int, expected_sizes: list[int]) -> None:
    ranges = list(fixed_chunk_ranges(num_tokens, chunk_size))
    assert [end - start for start, end in ranges] == expected_sizes
    assert ranges[0][0] == 0
    assert ranges[-1][1] == num_tokens


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


def test_fixed_size_chunking_preserves_order_and_keeps_tail() -> None:
    hidden_states = torch.arange(36, dtype=torch.float32).reshape(9, 4)
    original = hidden_states.clone()
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([3, 2, 4]))
    seen_sizes: list[int] = []

    def apply_mlp(chunk_input: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        seen_sizes.append(chunk_input.hidden_states.shape[0])
        return chunk_input.hidden_states + 1, None

    output, _ = run_moe_ffn_in_token_chunks(mlp_input, chunk_size=4, apply_mlp=apply_mlp)

    assert seen_sizes == [4, 4, 1]
    # When the MLP output dtype matches the input, the chunk runner writes
    # in-place into ``hidden_states`` to avoid a second [N, H] allocation.
    assert output.data_ptr() == hidden_states.data_ptr()
    torch.testing.assert_close(output, original + 1)


@pytest.mark.parametrize("num_tokens", [3, 4])
def test_fixed_size_chunking_bypasses_when_input_fits_one_chunk(num_tokens: int) -> None:
    hidden_states = torch.randn(num_tokens, 8)
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([num_tokens]))
    seen: list[MoEMlpComputeInput] = []

    def apply_mlp(value: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        seen.append(value)
        return value.hidden_states, None

    output, _ = run_moe_ffn_in_token_chunks(mlp_input, chunk_size=4, apply_mlp=apply_mlp)

    assert seen == [mlp_input]
    assert output is hidden_states


def test_fixed_size_chunking_rejects_non_positive_chunk_size() -> None:
    hidden_states = torch.randn(4, 8)
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([4]))

    with pytest.raises(ValueError, match="chunk_size must be positive"):
        run_moe_ffn_in_token_chunks(mlp_input, chunk_size=0, apply_mlp=_expert_tagged_mlp)


def test_chunk_runner_requires_exactly_one_partition_strategy() -> None:
    hidden_states = torch.randn(4, 8)
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([2, 2]))

    with pytest.raises(ValueError, match="exactly one"):
        run_moe_ffn_in_token_chunks(mlp_input, apply_mlp=_expert_tagged_mlp)
    with pytest.raises(ValueError, match="exactly one"):
        run_moe_ffn_in_token_chunks(
            mlp_input,
            num_chunks=2,
            chunk_size=2,
            apply_mlp=_expert_tagged_mlp,
        )


def test_chunked_moe_ffn_preallocates_output_with_actual_result_dtype() -> None:
    hidden_states = torch.randint(-4, 4, (17, 8), dtype=torch.int8)
    expected = hidden_states.to(torch.bfloat16)
    original_data_ptr = hidden_states.data_ptr()
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([5, 6, 6]))

    def bf16_output(value: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        return value.hidden_states.to(torch.bfloat16), None

    output, _ = run_moe_ffn_in_token_chunks(mlp_input, num_chunks=4, apply_mlp=bf16_output)

    assert output.dtype == torch.bfloat16
    # Dtype mismatch forces a fresh output buffer instead of the in-place path.
    assert output.data_ptr() != original_data_ptr
    torch.testing.assert_close(output, expected)
    # The fallback path disposes the base ``hidden_states`` mid-loop so its
    # ``N * H`` bytes are not held alongside the new output buffer.
    assert mlp_input.hidden_states.numel() == 0


def test_chunked_moe_ffn_writes_output_in_place_when_dtype_matches() -> None:
    hidden_states = torch.arange(36, dtype=torch.bfloat16).reshape(9, 4)
    original = hidden_states.clone()
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([3, 2, 4]))
    base_ptr = hidden_states.data_ptr()

    def bump(value: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        # Force a fresh contiguous chunk output so ``copy_`` never aliases the
        # source with the destination (mirrors real GMM2 output semantics).
        return (value.hidden_states.clone() + 1).contiguous(), None

    output, _ = run_moe_ffn_in_token_chunks(mlp_input, chunk_size=4, apply_mlp=bump)

    assert output.data_ptr() == base_ptr
    torch.testing.assert_close(output, original + 1)


def test_chunked_moe_ffn_falls_back_when_input_is_not_contiguous() -> None:
    # A non-contiguous input cannot be safely reused as the output buffer.
    base = torch.arange(72, dtype=torch.bfloat16).reshape(9, 8)
    hidden_states = base[:, :4]
    assert not hidden_states.is_contiguous()
    expected = hidden_states + 1
    original_data_ptr = hidden_states.data_ptr()
    mlp_input = _make_mlp_input(hidden_states, group_list=torch.tensor([3, 2, 4]))

    def bump(value: MoEMlpComputeInput) -> tuple[torch.Tensor, None]:
        return (value.hidden_states.clone() + 1).contiguous(), None

    output, _ = run_moe_ffn_in_token_chunks(mlp_input, chunk_size=4, apply_mlp=bump)

    assert output.data_ptr() != original_data_ptr
    torch.testing.assert_close(output, expected)


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

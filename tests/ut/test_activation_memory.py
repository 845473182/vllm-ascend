# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.activation_memory import (
    activation_peak_profile_session,
    record_activation_peak,
)


def test_nested_activation_peak_scopes_preserve_overall_peak():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 130, 170, 180, 180]),
        memory_allocated=MagicMock(side_effect=[100, 120, 125, 105]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning"),
        activation_peak_profile_session(enabled=True, rank=3) as profiler,
    ):
        assert profiler is not None
        with (
            record_activation_peak("target_model", "target", num_tokens=32),
            record_activation_peak(
                "target_attention",
                "target.layer.0.attention",
                layer_idx=0,
                num_tokens=32,
            ),
        ):
            pass

    assert profiler.overall_peak_bytes == 180
    assert len(profiler.records) == 2
    attention, target = profiler.records
    assert attention.category == "target_attention"
    assert attention.baseline_bytes == 120
    assert attention.peak_bytes == 170
    assert attention.peak_growth_bytes == 50
    assert attention.retained_bytes == 5
    assert target.category == "target_model"
    assert target.baseline_bytes == 100
    assert target.peak_bytes == 180
    assert target.peak_growth_bytes == 80
    assert target.retained_bytes == 5
    assert fake_npu.reset_peak_memory_stats.call_count == 3


def test_disabled_activation_peak_session_is_noop():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(),
        memory_allocated=MagicMock(),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        activation_peak_profile_session(enabled=False, rank=0) as profiler,
    ):
        assert profiler is None
        with record_activation_peak("target_moe", "target.layer.0.moe"):
            pass

    fake_npu.synchronize.assert_not_called()
    fake_npu.reset_peak_memory_stats.assert_not_called()
    fake_npu.max_memory_allocated.assert_not_called()
    fake_npu.memory_allocated.assert_not_called()

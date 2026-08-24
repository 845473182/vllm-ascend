# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from vllm_ascend.activation_memory import (
    activation_peak_profile_session,
    log_ffn_chunk_decision,
    log_mtp_moe_chunk_detail,
    record_activation_buffer,
    record_activation_peak,
    record_mtp_moe_substage,
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
        record_activation_buffer("mtp_hidden_buffer", "mtp.hidden", 70)
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
    assert len(profiler.buffers) == 1
    assert profiler.buffers[0].category == "mtp_hidden_buffer"
    assert profiler.buffers[0].size_bytes == 70
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
        record_activation_buffer("mtp_hidden_buffer", "mtp.hidden", 1024)
        with record_activation_peak("target_moe", "target.layer.0.moe"):
            pass

    fake_npu.synchronize.assert_not_called()
    fake_npu.reset_peak_memory_stats.assert_not_called()
    fake_npu.max_memory_allocated.assert_not_called()
    fake_npu.memory_allocated.assert_not_called()


def test_ffn_chunk_decision_uses_current_moe_scope():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 120, 120]),
        memory_allocated=MagicMock(side_effect=[100, 100]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        activation_peak_profile_session(enabled=True, rank=2),
        record_activation_peak(
            "mtp_moe",
            "mtp.layer.61.moe",
            layer_idx=61,
            num_tokens=1500,
        ),
    ):
        log_ffn_chunk_decision(
            comm_method="FusedMC2CommImpl",
            path="E2E_DISPATCH_FFN_COMBINE",
            enabled=True,
            raw_tokens=1500,
            dispatched_tokens=None,
            chunk_size=512,
            num_chunks=3,
            applied=True,
            reason="chunked",
        )

    chunk_log = next(call for call in mock_warning.call_args_list if "[FFN_CHUNK]" in call.args[0])
    assert chunk_log.args[1:] == (
        2,
        "MTP_MOE",
        "mtp.layer.61.moe",
        61,
        "FusedMC2CommImpl",
        "E2E_DISPATCH_FFN_COMBINE",
        True,
        1500,
        "-",
        512,
        3,
        True,
        "chunked",
    )


def test_mtp_moe_substage_and_chunk_detail_use_parent_scope():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 120, 130, 140, 140]),
        memory_allocated=MagicMock(side_effect=[100, 105, 105, 100]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        activation_peak_profile_session(enabled=True, rank=0) as profiler,
        record_activation_peak("mtp_moe", "mtp.layer.0.moe", layer_idx=0, num_tokens=1500),
    ):
        with record_mtp_moe_substage("dispatch", "moe.chunk.1.dispatch", num_tokens=500):
            log_mtp_moe_chunk_detail(
                chunk_idx=0,
                num_chunks=3,
                raw_start=0,
                raw_end=500,
                dispatched_tokens=3040,
            )

    assert profiler is not None
    assert [record.category for record in profiler.records] == ["mtp_moe_dispatch", "mtp_moe"]
    detail_log = next(call for call in mock_warning.call_args_list if "[FFN_CHUNK_DETAIL]" in call.args[0])
    assert detail_log.args[1:] == (0, 0, 1, 3, 0, 500, 500, 3040)


def test_mtp_moe_substage_is_noop_in_target_scope():
    fake_npu = SimpleNamespace(
        synchronize=MagicMock(),
        reset_peak_memory_stats=MagicMock(),
        max_memory_allocated=MagicMock(side_effect=[100, 110, 120, 120]),
        memory_allocated=MagicMock(side_effect=[100, 100]),
    )

    with (
        patch("vllm_ascend.activation_memory.torch.npu", fake_npu, create=True),
        patch("vllm_ascend.activation_memory.logger.warning") as mock_warning,
        activation_peak_profile_session(enabled=True, rank=0) as profiler,
        record_activation_peak("target_moe", "target.layer.0.moe", layer_idx=0, num_tokens=1500),
    ):
        with record_mtp_moe_substage("dispatch", "moe.dispatch", num_tokens=1500):
            pass
        log_mtp_moe_chunk_detail(
            chunk_idx=0,
            num_chunks=1,
            raw_start=0,
            raw_end=1500,
            dispatched_tokens=9000,
        )

    assert profiler is not None
    assert [record.category for record in profiler.records] == ["target_moe"]
    assert not any("[FFN_CHUNK_DETAIL]" in call.args[0] for call in mock_warning.call_args_list)

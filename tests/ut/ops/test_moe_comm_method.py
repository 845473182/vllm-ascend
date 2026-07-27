from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import torch
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from tests.ut.base import TestBase
from vllm_ascend.ops.fused_moe.moe_comm_method import (
    AllGatherCommImpl,
    AlltoAllCommImpl,
    MC2CommImpl,
    MoECommMethod,
)
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEAllGatherCombineMetadata,
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    MoEQuantParams,
    MoERoutingParams,
    MoEWeights,
    slice_fused_experts_input_along_tokens,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import MoETokenDispatchOutput
from vllm_ascend.quantization.methods.base import QuantType


class TestMoECommMethod(TestBase):
    def setUp(self):
        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant = False
        self.mock_ascend_config.enable_fused_mc2 = False
        self.mock_ascend_config.ffn_chunk_memory_snapshot_dir = None
        self.mock_ascend_config.ffn_chunk_memory_snapshot_rank = 0
        self.mock_ascend_config.ffn_chunk_memory_snapshot_max_entries = 100000
        self._patch_get_ascend_config = patch(
            "vllm_ascend.ops.fused_moe.moe_comm_method.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config_module = patch(
            "vllm_ascend.ascend_config.get_ascend_config",
            return_value=self.mock_ascend_config,
        )
        self._patch_get_ascend_config.start()
        self._patch_get_ascend_config_module.start()
        # Mock FusedMoEConfig
        self.moe_config = MagicMock(spec=FusedMoEConfig)
        self.moe_config.num_experts = 8
        self.moe_config.num_local_experts = 2
        self.moe_config.experts_per_token = 2
        self.moe_config.tp_group = MagicMock()
        self.moe_config.tp_group.device_group = MagicMock()
        self.moe_config.dp_size = 1
        self.moe_config.tp_size = 1
        self.moe_config.pcp_size = 1
        self.moe_config.ep_size = 1
        self.moe_config.dp_group = MagicMock()
        self.moe_config.global_redundant_expert_num = 0

    def tearDown(self):
        self._patch_get_ascend_config.stop()
        self._patch_get_ascend_config_module.stop()

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_apply_mlp_chunks_only_routed_expert_compute(self, mock_token_dispatcher, mock_prepare_finalize):
        self.mock_ascend_config.enable_ffn_chunking = True
        self.mock_ascend_config.ffn_chunk_size = 2
        self.moe_config.intermediate_size_per_partition = 8
        self.moe_config.intermediate_size = 8
        comm_impl = AllGatherCommImpl(self.moe_config)
        # Use a fresh input tensor per invocation because the chunk runner
        # now writes results back into ``hidden_states`` in place when the
        # output dtype matches, so the second call would otherwise observe
        # the first call's mutations.
        first_hidden_states = torch.arange(36, dtype=torch.float32).reshape(9, 4)
        original = first_hidden_states.clone()
        second_hidden_states = original.clone()

        def _make_input(hidden: torch.Tensor) -> MoEMlpComputeInput:
            return MoEMlpComputeInput(
                hidden_states=hidden,
                group_list=torch.tensor([3, 2, 4]),
                group_list_type=1,
                dynamic_scale=None,
                topk_scales=None,
                weights=MoEWeights(w1=torch.empty(0), w2=torch.empty(0)),
                quant=MoEQuantParams(),
                fusion=False,
            )

        seen_group_lists = []

        def fake_apply_mlp(value):
            seen_group_lists.append(value.group_list.clone())
            return value.hidden_states + 1, None

        comm_impl._apply_mlp = fake_apply_mlp
        with patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.info_once") as mock_info_once:
            output, _ = comm_impl._apply_mlp_with_optional_chunking(_make_input(first_hidden_states))
            comm_impl._apply_mlp_with_optional_chunking(_make_input(second_hidden_states))

        torch.testing.assert_close(output, original + 1)
        assert output.data_ptr() == first_hidden_states.data_ptr()
        mock_info_once.assert_called_once()
        assert "MoE FFN token chunking is active" in mock_info_once.call_args.args[0]
        assert [value.tolist() for value in seen_group_lists] == [
            [2, 0, 0],
            [1, 1, 0],
            [0, 1, 1],
            [0, 0, 2],
            [0, 0, 1],
        ] * 2

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_apply_mlp_memory_debug_reports_moe_peak(self, mock_token_dispatcher, mock_prepare_finalize):
        self.mock_ascend_config.ffn_chunk_memory_debug = True
        comm_impl = AllGatherCommImpl(self.moe_config)
        comm_impl.enable_ffn_chunking = True
        comm_impl._last_ffn_num_chunks = 2
        hidden_states = torch.arange(36, dtype=torch.float32).reshape(9, 4)
        mlp_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([3, 2, 4]),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(w1=torch.empty(0), w2=torch.empty(0)),
            quant=MoEQuantParams(),
            fusion=False,
        )
        output = hidden_states + 1
        event = object()
        mib = 1 << 20

        with (
            patch("torch.npu.synchronize") as mock_synchronize,
            patch("torch.npu.reset_peak_memory_stats") as mock_reset_peak,
            patch("torch.npu.memory_allocated", return_value=100 * mib),
            patch("torch.npu.max_memory_allocated", return_value=150 * mib),
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.warning") as mock_warning,
        ):
            actual = comm_impl._apply_mlp_with_memory_debug(mlp_input, lambda _: (output, event))

        assert actual[0] is output
        assert actual[1] is event
        assert mock_synchronize.call_count == 2
        mock_reset_peak.assert_called_once_with()
        assert comm_impl._ffn_chunk_memory_debug_logged
        log_args = mock_warning.call_args.args
        rendered_log = log_args[0] % log_args[1:]
        assert "[MOE_MEMORY]" in rendered_log
        assert "ffn_chunk=ON" in rendered_log
        assert "routed_tokens=9" in rendered_log
        assert "num_chunks=2" in rendered_log
        assert "moe_peak_allocated=150.000 MiB" in rendered_log
        assert "moe_peak_increase=50.000 MiB" in rendered_log

    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_apply_mlp_memory_debug_dumps_snapshot(self, mock_token_dispatcher, mock_prepare_finalize):
        self.mock_ascend_config.ffn_chunk_memory_debug = True
        self.mock_ascend_config.ffn_chunk_memory_snapshot_rank = 0
        self.mock_ascend_config.ffn_chunk_memory_snapshot_max_entries = 123
        hidden_states = torch.arange(36, dtype=torch.float32).reshape(9, 4)
        mlp_input = MoEMlpComputeInput(
            hidden_states=hidden_states,
            group_list=torch.tensor([3, 2, 4]),
            group_list_type=1,
            dynamic_scale=None,
            topk_scales=None,
            weights=MoEWeights(w1=torch.empty(0), w2=torch.empty(0)),
            quant=MoEQuantParams(),
            fusion=False,
        )

        with TemporaryDirectory() as snapshot_dir:
            self.mock_ascend_config.ffn_chunk_memory_snapshot_dir = snapshot_dir
            comm_impl = AllGatherCommImpl(self.moe_config)
            output = hidden_states.clone() + 1

            def fake_disposing_apply_mlp(value):
                value.hidden_states.set_(torch.empty(0, dtype=value.hidden_states.dtype))
                return output, None

            with (
                patch("torch.npu.synchronize"),
                patch("torch.npu.reset_peak_memory_stats"),
                patch("torch.npu.memory_allocated", return_value=100),
                patch("torch.npu.max_memory_allocated", return_value=150),
                patch("torch.npu.memory._record_memory_history") as mock_record_history,
                patch("torch.npu.memory._dump_snapshot") as mock_dump_snapshot,
                patch("vllm_ascend.ops.fused_moe.moe_comm_method.logger.warning") as mock_warning,
            ):
                actual = comm_impl._apply_mlp_with_memory_debug(mlp_input, fake_disposing_apply_mlp)

        assert actual[0] is output
        assert mlp_input.hidden_states.shape == (0,)
        assert mock_record_history.call_count == 2
        mock_record_history.assert_any_call(
            enabled="all",
            context="all",
            stacks="python",
            max_entries=123,
        )
        mock_record_history.assert_any_call(enabled=None)
        snapshot_path = mock_dump_snapshot.call_args.args[0]
        assert "moe_ffn_chunk_off_rank0_tokens9_chunks1_" in snapshot_path
        assert snapshot_path.endswith(".pickle")
        rendered_logs = [call.args[0] % call.args[1:] for call in mock_warning.call_args_list]
        assert any("routed_tokens=9" in message for message in rendered_logs)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    def test_all_gather_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithMC2")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithMC2")
    def test_mc2_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "mc2"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=torch.tensor([1, 0, 1, 0]),
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = MC2CommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        prepare_output = comm_impl.prepare(hidden_states, router_logits)
        h_out = prepare_output.hidden_states
        padded_hidden_states_shape = prepare_output.padded_hidden_states_shape

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

        # Test finalize method
        comm_impl.finalize(h_out, reduce_results=True, padded_hidden_states_shape=padded_hidden_states_shape)
        mock_pf_instance.finalize.assert_called_once_with(h_out, True, None)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAll2All")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_comm_impl(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "alltoall"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        mock_token_dispatcher.return_value = mock_td_instance

        # Create instance
        comm_impl = AlltoAllCommImpl(self.moe_config)

        # Test prepare method
        hidden_states = torch.randn(3, 8)
        router_logits = torch.randn(3, 2)
        _ = comm_impl.prepare(hidden_states, router_logits)

        # Verify prepare was called with correct arguments
        mock_pf_instance.prepare.assert_called_once_with(hidden_states, router_logits, False, False, QuantType.NONE)

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAllGather")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.unified_apply_mlp")
    @patch("torch.npu.current_stream", MagicMock())
    def test_fused_experts_method(
        self, mock_unified_apply_mlp, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        # Mock forward context
        mock_context = MagicMock()
        mock_context.moe_comm_method = "all_gather"
        mock_get_forward_context.return_value = mock_context

        # Mock prepare finalize
        mock_pf_instance = MagicMock()
        mock_pf_instance.prepare.return_value = MoEPrepareOutput(
            hidden_states=torch.randn(4, 8),
            router_logits=torch.randn(4, 2),
            mc2_mask=None,
            padded_hidden_states_shape=None,
        )
        mock_pf_instance.finalize.return_value = torch.randn(4, 8)
        mock_prepare_finalize.return_value = mock_pf_instance

        # Mock token dispatcher
        mock_td_instance = MagicMock()
        dispatch_topk_weights = torch.tensor([[0.5, 0.5], [0.3, 0.7], [0.8, 0.2], [0.6, 0.4]])
        mock_td_instance.token_dispatch.return_value = MoETokenDispatchOutput(
            hidden_states=torch.randn(6, 8),
            group_list=torch.tensor([2, 2, 2]),
            group_list_type=1,
            combine_metadata=MoEAllGatherCombineMetadata(
                topk_weights=dispatch_topk_weights,
                expanded_row_idx=torch.arange(8, dtype=torch.int32),
                restore_shape=torch.Size([4, 8]),
            ),
        )
        mock_td_instance.token_combine.return_value = torch.randn(4, 8)
        mock_token_dispatcher.return_value = mock_td_instance

        # Mock unified_apply_mlp returns (tensor, event) tuple
        mock_unified_apply_mlp.return_value = (torch.randn(6, 8), MagicMock())

        # Create instance
        comm_impl = AllGatherCommImpl(self.moe_config)

        # Test fused_experts method
        hidden_states = torch.randn(4, 8).contiguous()
        w1 = torch.randn(16, 8).contiguous()
        w2 = torch.randn(16, 8).contiguous()
        topk_weights = dispatch_topk_weights
        topk_ids = torch.tensor([[0, 1], [1, 2], [2, 0], [1, 1]])

        # Make sure tensors are contiguous and have correct strides
        hidden_states = hidden_states.contiguous()
        w1 = w1.contiguous()
        w2 = w2.contiguous()

        result = comm_impl.fused_experts(
            fused_experts_input=MoEFusedExpertsInput(
                hidden_states=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                weights=MoEWeights(
                    w1=[w1],
                    w2=[w2],
                ),
                routing=MoERoutingParams(
                    expert_map=None,
                    global_redundant_expert_num=0,
                    mc2_mask=None,
                    apply_router_weight_on_input=False,
                ),
                activation="silu",
                need_trans=False,
                dynamic_eplb=False,
                quant=MoEQuantParams(),
            )
        )

        # Verify result shape
        self.assertEqual(result.routed_out.shape, (4, 8))

        # Verify token_dispatch was called
        mock_td_instance.token_dispatch.assert_called_once()

        # Verify unified_apply_mlp was called
        mock_unified_apply_mlp.assert_called_once()
        mlp_compute_input = mock_unified_apply_mlp.call_args.kwargs["mlp_compute_input"]
        self.assertFalse(mlp_compute_input.fusion)
        self.assertFalse(mlp_compute_input.quant.is_mxfp)

        # Verify token_combine was called
        mock_td_instance.token_combine.assert_called_once_with(
            hidden_states=mock_unified_apply_mlp.return_value[0],
            combine_metadata=mock_td_instance.token_dispatch.return_value.combine_metadata,
        )

    def test_slice_fused_experts_input_along_tokens(self):
        hidden_states = torch.arange(72, dtype=torch.float32).reshape(9, 8)
        topk_ids = torch.arange(18, dtype=torch.int32).reshape(9, 2)
        topk_weights = torch.arange(18, dtype=torch.float32).reshape(9, 2)
        mc2_mask = torch.arange(9, dtype=torch.int32) > 3
        pertoken_scale = torch.arange(9, dtype=torch.float32)
        fused_input = MoEFusedExpertsInput(
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            weights=MoEWeights(w1=torch.empty(0), w2=torch.empty(0)),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=0,
                mc2_mask=mc2_mask,
                apply_router_weight_on_input=False,
                pertoken_scale=pertoken_scale,
            ),
            quant=MoEQuantParams(),
        )

        sliced = slice_fused_experts_input_along_tokens(fused_input, 3, 7)

        torch.testing.assert_close(sliced.hidden_states, hidden_states[3:7])
        torch.testing.assert_close(sliced.topk_ids, topk_ids[3:7])
        torch.testing.assert_close(sliced.topk_weights, topk_weights[3:7])
        torch.testing.assert_close(sliced.routing.mc2_mask, mc2_mask[3:7])
        torch.testing.assert_close(sliced.routing.pertoken_scale, pertoken_scale[3:7])
        # Static topology fields are shared, not copied.
        assert sliced.weights is fused_input.weights
        assert sliced.quant is fused_input.quant
        assert sliced.routing.global_redundant_expert_num == 0
        # The original payload is untouched.
        assert fused_input.hidden_states.shape[0] == 9

    def _make_e2e_fused_input(self, hidden_states: torch.Tensor) -> MoEFusedExpertsInput:
        num_tokens = hidden_states.shape[0]
        return MoEFusedExpertsInput(
            hidden_states=hidden_states,
            topk_weights=torch.ones(num_tokens, 2),
            topk_ids=torch.zeros(num_tokens, 2, dtype=torch.int32),
            weights=MoEWeights(w1=torch.empty(0), w2=torch.empty(0)),
            routing=MoERoutingParams(
                expert_map=None,
                global_redundant_expert_num=0,
                mc2_mask=torch.ones(num_tokens, dtype=torch.bool),
                apply_router_weight_on_input=False,
                pertoken_scale=torch.ones(num_tokens),
            ),
            quant=MoEQuantParams(),
        )

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAll2All")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_fused_experts_e2e_chunking(self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context):
        mock_context = MagicMock()
        mock_context.moe_comm_method = "alltoall"
        mock_get_forward_context.return_value = mock_context
        self.mock_ascend_config.enable_ffn_chunking = True
        self.mock_ascend_config.ffn_chunk_size = 4

        dispatch_calls = []

        def fake_dispatch(token_dispatch_input):
            hs = token_dispatch_input.hidden_states
            # Routing side-inputs must be sliced alongside the token dim.
            assert token_dispatch_input.routing.mc2_mask.shape[0] == hs.shape[0]
            assert token_dispatch_input.routing.pertoken_scale.shape[0] == hs.shape[0]
            dispatch_calls.append(hs.shape[0])
            return MoETokenDispatchOutput(
                hidden_states=hs.clone(),
                group_list=torch.tensor([hs.shape[0]]),
                group_list_type=1,
                combine_metadata=MagicMock(expanded_row_idx=None),
                dynamic_scale=None,
                topk_scales=None,
            )

        mock_td_instance = MagicMock()
        mock_td_instance.token_dispatch.side_effect = fake_dispatch
        mock_td_instance.token_combine.side_effect = lambda hidden_states, combine_metadata: hidden_states
        mock_token_dispatcher.return_value = mock_td_instance

        comm_impl = AlltoAllCommImpl(self.moe_config)
        apply_calls = []

        def fake_apply_mlp(value):
            apply_calls.append(value.hidden_states.shape[0])
            return (value.hidden_states.to(torch.bfloat16) * 2), f"gmm2_evt_{len(apply_calls)}"

        comm_impl._apply_mlp = fake_apply_mlp

        stream_events = ["evt_dispatch", "evt_combine_0", "evt_combine_1", "evt_combine_2"]
        mock_stream = MagicMock()
        mock_stream.return_value.record_event.side_effect = stream_events

        hidden_states = torch.arange(72, dtype=torch.bfloat16).reshape(9, 8)
        with patch("torch.npu.current_stream", mock_stream):
            result = comm_impl.fused_experts(fused_experts_input=self._make_e2e_fused_input(hidden_states))

        # 9 tokens with chunk_size=4 -> 3 chunks of [4, 4, 1].
        assert dispatch_calls == [4, 4, 1]
        assert apply_calls == [4, 4, 1]
        assert mock_td_instance.token_combine.call_count == 3
        # Partial chunk outputs are assembled into one full routed_out.
        torch.testing.assert_close(result.routed_out, hidden_states * 2)
        # Per-expert token counts are summed across chunks for EPLB heat.
        assert result.expert_tokens.tolist() == [9]
        assert result.group_list_type == 1
        # Earliest dispatch event, latest gmm2/combine events.
        assert result.before_dispatch_evt == "evt_dispatch"
        assert result.before_gmm2_evt == "gmm2_evt_3"
        assert result.before_combine_evt == "evt_combine_2"
        assert comm_impl._last_ffn_num_chunks == 3

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAll2All")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_e2e_chunking_falls_back_when_batch_fits(
        self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        mock_get_forward_context.return_value = MagicMock()
        self.mock_ascend_config.enable_ffn_chunking = True
        self.mock_ascend_config.ffn_chunk_size = 8
        mock_token_dispatcher.return_value = MagicMock()

        comm_impl = AlltoAllCommImpl(self.moe_config)
        hidden_states = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

        with (
            patch.object(MoECommMethod, "fused_experts", autospec=True, return_value="base_result") as mock_base,
            patch.object(AlltoAllCommImpl, "_fused_experts_chunked_e2e") as mock_e2e,
        ):
            result = comm_impl.fused_experts(fused_experts_input=self._make_e2e_fused_input(hidden_states))

        # 4 tokens <= chunk_size 8 -> base single-pass path, no chunking.
        assert result == "base_result"
        mock_base.assert_called_once()
        mock_e2e.assert_not_called()

    @patch("vllm_ascend.ascend_forward_context.get_forward_context")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.PrepareAndFinalizeWithAll2All")
    @patch("vllm_ascend.ops.fused_moe.moe_comm_method.TokenDispatcherWithAll2AllV")
    def test_alltoall_e2e_chunking_falls_back_when_dp_token_counts_diverge(
        self, mock_token_dispatcher, mock_prepare_finalize, mock_get_forward_context
    ):
        mock_get_forward_context.return_value = MagicMock()
        self.mock_ascend_config.enable_ffn_chunking = True
        self.mock_ascend_config.ffn_chunk_size = 2
        self.mock_ascend_config.dp_allreduce_on_npu = False
        mock_token_dispatcher.return_value = MagicMock()
        self.moe_config.dp_size = 2

        def fake_all_reduce(payload, op=None, group=None):
            # Another DP rank holds 8 tokens while this rank holds 4:
            # max=8, min=4 -> not uniform -> every rank must fall back.
            payload[0] = 8
            payload[1] = -4

        comm_impl = AlltoAllCommImpl(self.moe_config)
        hidden_states = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

        with (
            patch("vllm_ascend.ops.fused_moe.moe_comm_method.get_dp_group") as mock_get_dp_group,
            patch("torch.distributed.all_reduce", side_effect=fake_all_reduce),
            patch.object(MoECommMethod, "fused_experts", autospec=True, return_value="base_result") as mock_base,
            patch.object(AlltoAllCommImpl, "_fused_experts_chunked_e2e") as mock_e2e,
        ):
            mock_get_dp_group.return_value = MagicMock(world_size=2, cpu_group=MagicMock())
            result = comm_impl.fused_experts(fused_experts_input=self._make_e2e_fused_input(hidden_states))

        assert result == "base_result"
        mock_base.assert_called_once()
        mock_e2e.assert_not_called()

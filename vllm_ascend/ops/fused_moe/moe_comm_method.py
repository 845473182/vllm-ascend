# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from vllm.distributed import get_dp_group
from vllm.model_executor.layers.fused_moe import FusedMoEConfig

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.ascend_forward_context import _EXTRA_CTX, MoECommType
from vllm_ascend.ops.fused_moe.moe_ffn_chunking import (
    balanced_chunk_ranges,
    run_moe_ffn_in_token_chunks,
    supports_moe_ffn_chunking,
)
from vllm_ascend.ops.fused_moe.moe_mlp import unified_apply_mlp
from vllm_ascend.ops.fused_moe.moe_runtime_args import (
    MoEFusedExpertsInput,
    MoEMlpComputeInput,
    MoEPrepareOutput,
    build_mlp_compute_input,
    build_token_dispatch_input,
    slice_fused_experts_input_along_tokens,
)
from vllm_ascend.ops.fused_moe.prepare_finalize import (
    PrepareAndFinalize,
    PrepareAndFinalizeWithAll2All,
    PrepareAndFinalizeWithAllGather,
    PrepareAndFinalizeWithMC2,
)
from vllm_ascend.ops.fused_moe.token_dispatcher import (
    MoETokenDispatcher,
    TokenDispatcherWithAll2AllV,
    TokenDispatcherWithAllGather,
    TokenDispatcherWithMC2,
)
from vllm_ascend.quantization.quant_type import QuantType

_MoECommMethods: dict[MoECommType | None, MoECommMethod] = {}


def get_moe_comm_method(moe_comm_type: MoECommType | None) -> MoECommMethod | None:
    return _MoECommMethods.get(moe_comm_type)


def setup_moe_comm_method(moe_config):
    if moe_config.ep_size > 1:
        _MoECommMethods[MoECommType.ALLTOALL] = AlltoAllCommImpl(moe_config)
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)
        _MoECommMethods[MoECommType.MC2] = MC2CommImpl(moe_config)
        _MoECommMethods[MoECommType.FUSED_MC2] = FusedMC2CommImpl(moe_config)
    else:
        _MoECommMethods[MoECommType.ALLGATHER] = AllGatherCommImpl(moe_config)


def set_gmmswigluquant_method():
    from vllm_ascend.ascend_config import get_ascend_config

    ascend_config = get_ascend_config()
    return ascend_config.ascend_fusion_config.fusion_ops_gmmswigluquant


def _e2e_uniform_chunk_count(num_tokens: int, chunk_size: int, moe_config: FusedMoEConfig) -> int:
    """Chunk count every rank in the EP group can use (0 disables chunking).

    A chunked forward issues one dispatch/combine collective per chunk, so
    all ranks must agree on the NUMBER of chunks. Chunk SIZES may differ:
    the AllToAllV dispatcher already tolerates uneven per-rank token splits
    (``input_splits``/``output_splits``), and TP ranks always share the
    batch. We take one MIN/MAX all-reduce over the DP group and compute

        K = min(max_local_ceil(num_tokens / chunk_size), global_min_tokens)

    so every rank can produce K non-empty balanced chunks. K == 1 (or an
    idle rank with 0 tokens) disables chunking everywhere, keeping the
    collective call count identical on the fallback path.
    """
    k_local = max(1, (num_tokens + chunk_size - 1) // chunk_size)
    if getattr(moe_config, "dp_size", 1) <= 1:
        return k_local if k_local > 1 else 0
    dp_group = get_dp_group()
    if dp_group.world_size <= 1:
        return k_local if k_local > 1 else 0
    if get_ascend_config().dp_allreduce_on_npu:
        payload = torch.tensor([k_local, -num_tokens], dtype=torch.int64, device="npu")
        group = dp_group.device_group
    else:
        payload = torch.tensor([k_local, -num_tokens], dtype=torch.int64)
        group = dp_group.cpu_group
    torch.distributed.all_reduce(payload, op=torch.distributed.ReduceOp.MAX, group=group)
    k_uniform = min(int(payload[0].item()), int(-payload[1].item()))
    return k_uniform if k_uniform > 1 else 0


@dataclass
class FusedExpertsResult:
    routed_out: torch.Tensor
    # This field is for shared experts and should be set by the MoE
    # communication method that supports shared experts in parallel with routed
    # experts.
    before_dispatch_evt: torch.npu.Event | None = None
    before_gmm2_evt: torch.npu.Event | None = None
    before_combine_evt: torch.npu.Event | None = None
    # For dynamic_eplb
    group_list_type: int = 1
    expert_tokens: torch.Tensor | None = None
    swiglu_limit: float = 0.0


class MoECommMethod(ABC):
    """Base class for MoE communication methods."""

    def __init__(self, moe_config: FusedMoEConfig):
        self.moe_config = moe_config

        self.token_dispatcher = self._get_token_dispatcher()
        self.prepare_finalize = self._get_prepare_finalize()
        self.use_fusion_ops = set_gmmswigluquant_method()

        ascend_config = get_ascend_config()
        self.enable_ffn_chunking = getattr(ascend_config, "enable_ffn_chunking", False) is True
        self.ffn_chunk_size = ascend_config.ffn_chunk_size

    def prepare(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        enable_shared_expert_dp: bool = False,
        replace_allreduce: bool = False,
        quant_type: QuantType = QuantType.NONE,
    ) -> MoEPrepareOutput:
        return self.prepare_finalize.prepare(
            hidden_states,
            router_logits,
            enable_shared_expert_dp,
            replace_allreduce,
            quant_type,
        )

    def finalize(
        self,
        hidden_states: torch.Tensor,
        reduce_results: bool,
        padded_hidden_states_shape: torch.Size | None = None,
    ) -> torch.Tensor:
        hidden_states = self.prepare_finalize.finalize(hidden_states, reduce_results, padded_hidden_states_shape)
        return hidden_states

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        # Check constraints
        assert fused_experts_input.hidden_states.dtype in [
            torch.float32,
            torch.float16,
            torch.bfloat16,
            torch.int8,
            torch.float8_e4m3fn,
            torch.uint8,
        ], f"Unsupported hidden_states dtype: {fused_experts_input.hidden_states.dtype}"

        moe_comm_method = _EXTRA_CTX.moe_comm_method
        assert moe_comm_method is not None, "Missing communication context"

        before_dispatch_evt = torch.npu.current_stream().record_event()
        routed_topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            routed_topk_ids = fused_experts_input.routing.log2phy[routed_topk_ids]

        token_dispatch_input = build_token_dispatch_input(
            fused_experts_input=fused_experts_input,
            topk_ids=routed_topk_ids,
        )
        token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

        mlp_compute_input = build_mlp_compute_input(
            fused_experts_input=fused_experts_input,
            token_dispatch_output=token_dispatch_output,
            use_fusion_ops=self.use_fusion_ops,
        )

        apply_mlp = self._apply_mlp_with_optional_chunking if self.enable_ffn_chunking else self._apply_mlp
        mlp_output, before_gmm2_evt = apply_mlp(mlp_compute_input)

        before_combine_evt = torch.npu.current_stream().record_event()
        routed_out = self.token_dispatcher.token_combine(
            hidden_states=mlp_output,
            combine_metadata=token_dispatch_output.combine_metadata,
        )

        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=token_dispatch_output.group_list_type,
            expert_tokens=token_dispatch_output.group_list,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    def _apply_mlp(self, mlp_compute_input: MoEMlpComputeInput) -> tuple[torch.Tensor, object | None]:
        return unified_apply_mlp(mlp_compute_input=mlp_compute_input)

    def _estimate_ffn_num_chunks(self, mlp_compute_input: MoEMlpComputeInput) -> int:
        if not supports_moe_ffn_chunking(mlp_compute_input):
            return 1
        num_tokens = mlp_compute_input.hidden_states.shape[0]
        return max(1, (num_tokens + self.ffn_chunk_size - 1) // self.ffn_chunk_size)

    def _apply_mlp_with_optional_chunking(
        self,
        mlp_compute_input: MoEMlpComputeInput,
    ) -> tuple[torch.Tensor, object | None]:
        """Chunk only the local routed expert MLP, never dispatch/combine."""
        num_chunks = self._estimate_ffn_num_chunks(mlp_compute_input)
        if num_chunks <= 1:
            return self._apply_mlp(mlp_compute_input)

        return run_moe_ffn_in_token_chunks(
            mlp_compute_input,
            chunk_size=self.ffn_chunk_size,
            apply_mlp=self._apply_mlp,
        )

    @abstractmethod
    def _get_token_dispatcher(self) -> MoETokenDispatcher:
        raise NotImplementedError("_get_token_dispatcher function not implemented.")

    @abstractmethod
    def _get_prepare_finalize(self) -> PrepareAndFinalize:
        raise NotImplementedError("_get_prepare_finalize function not implemented.")


class AllGatherCommImpl(MoECommMethod):
    """This implementation is the same as NativeAllGatherCommImpl,
    but uses NPU-specific ops for better performance.

    This implementation should be compatible with all scenarios, and
    thus it is the default implementation for MoE communication methods.
    It uses `torch_npu.npu_moe_init_routing_v2` for pre-processing
    and `torch_npu.npu_moe_token_unpermute` for post-processing
    to handle the token-to-expert mapping and communication efficiently.

    NOTE(Yizhou): TBH, it is really weird that we were supposed to use
    `torch_npu.npu_moe_init_routing_v2` and `torch_npu.npu_moe_finalize_routing`
    or `torch_npu.npu_moe_token_permute` and `torch_npu.npu_moe_token_unpermute`
    for pre-processing and post-processing, respectively.
    But `npu_moe_finalize_routing` will lead to accuracy issues so we have to
    use `torch_npu.npu_moe_token_unpermute` instead.
    This is a workaround and should be removed after the issue is fixed.
    """

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAllGather(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAllGather(self.moe_config)


class MC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)


class AlltoAllCommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_grouped_matmul` is available.

    This implementation uses all-to-all communication to exchange tokens
    between data parallel ranks before and after the MLP computation. It should
    have better performance than AllGatherCommImpl when DP size > 1.
    """

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithAll2AllV(
            top_k=self.moe_config.experts_per_token,
            num_experts=self.moe_config.num_experts,
            num_local_experts=self.moe_config.num_local_experts,
        )

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithAll2All(self.moe_config)

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        num_tokens = fused_experts_input.hidden_states.shape[0]
        num_chunks = self._e2e_chunk_count(num_tokens)
        if num_chunks == 0:
            return super().fused_experts(fused_experts_input)
        return self._fused_experts_chunked_e2e(fused_experts_input, num_chunks)

    def _e2e_chunk_count(self, num_tokens: int) -> int:
        """Uniform per-rank chunk count for the e2e path (0 disables it).

        Only the All2AllV dispatcher is validated for repeated per-chunk
        calls; the MC2 combine kernel's sub-batch contract is unverified, so
        this override exists only on ``AlltoAllCommImpl``. Chunk boundaries
        may differ across ranks (AllToAllV handles uneven splits), but the
        chunk COUNT must be uniform across the EP group because every chunk
        issues collectives — see ``_e2e_uniform_chunk_count``.
        """
        if not self.enable_ffn_chunking:
            return 0
        return _e2e_uniform_chunk_count(num_tokens, self.ffn_chunk_size, self.moe_config)

    def _fused_experts_chunked_e2e(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        num_chunks: int,
    ) -> FusedExpertsResult:
        """Run dispatch -> MLP -> combine per raw-token chunk.

        Each chunk is an independent dispatch/combine round trip: the
        dispatched activation (``expand_x``), the GMM intermediates and the
        combine workspace all live only for the duration of one chunk, so
        their peak scales with the per-rank chunk size instead of the full
        batch. Chunk sizes are balanced locally and may differ from other
        ranks'; only the chunk COUNT is uniform. The final outputs are
        accumulated into a single preallocated ``routed_out`` buffer, which
        ``finalize`` consumes afterwards exactly as in the unchunked path.
        """
        num_tokens = fused_experts_input.hidden_states.shape[0]
        ranges = list(balanced_chunk_ranges(num_tokens, num_chunks))

        # Earliest dispatch event lets the shared-expert stream start as soon
        # as the first chunk's dispatch begins; the latest GMM2/combine events
        # keep it from racing ahead of the last chunk (see the single-arc
        # semantics in the base implementation).
        before_dispatch_evt = torch.npu.current_stream().record_event()

        routed_out: torch.Tensor | None = None
        expert_tokens: torch.Tensor | None = None
        group_list_type = 1
        before_gmm2_evt = None
        before_combine_evt = None

        for start, end in ranges:
            chunk_input = slice_fused_experts_input_along_tokens(fused_experts_input, start, end)
            routed_topk_ids = chunk_input.topk_ids
            if chunk_input.routing.log2phy is not None:
                routed_topk_ids = chunk_input.routing.log2phy[routed_topk_ids]

            token_dispatch_input = build_token_dispatch_input(
                fused_experts_input=chunk_input,
                topk_ids=routed_topk_ids,
            )
            token_dispatch_output = self.token_dispatcher.token_dispatch(token_dispatch_input=token_dispatch_input)

            chunk_group_list = token_dispatch_output.group_list
            group_list_type = token_dispatch_output.group_list_type

            mlp_compute_input = build_mlp_compute_input(
                fused_experts_input=chunk_input,
                token_dispatch_output=token_dispatch_output,
                use_fusion_ops=self.use_fusion_ops,
            )
            mlp_output, before_gmm2_evt = self._apply_mlp(mlp_compute_input)
            before_combine_evt = torch.npu.current_stream().record_event()
            chunk_routed_out = self.token_dispatcher.token_combine(
                hidden_states=mlp_output,
                combine_metadata=token_dispatch_output.combine_metadata,
            )
            del mlp_output, token_dispatch_output

            if routed_out is None:
                routed_out = chunk_routed_out.new_empty((num_tokens, *chunk_routed_out.shape[1:]))
            routed_out[start:end].copy_(chunk_routed_out)
            del chunk_routed_out

            if chunk_group_list is not None:
                # Per-expert token counts are partial per chunk; EPLB heat
                # collection needs the total across the whole batch.
                if expert_tokens is None:
                    expert_tokens = chunk_group_list.clone()
                else:
                    expert_tokens += chunk_group_list

        assert routed_out is not None
        return FusedExpertsResult(
            routed_out=routed_out,
            before_dispatch_evt=before_dispatch_evt,
            before_gmm2_evt=before_gmm2_evt,
            before_combine_evt=before_combine_evt,
            group_list_type=group_list_type,
            expert_tokens=expert_tokens,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )


class FusedMC2CommImpl(MoECommMethod):
    """This implementation is for the scenarios listed below:
    1. `enable_expert_parallel=True`.
    2. `npu_moe_distribute_dispatch` and `npu_moe_distribute_combine` are available.
    3. `enable_expert_parallel=False` is not supported.

    This implementation uses the MC2 communication method, which is optimized for
    Communication and Computation parallelism on Ascend devices.
    """

    def __init__(self, moe_config):
        super().__init__(moe_config)
        if get_ascend_config().enable_fused_mc2 == 1:
            self.expert_token_nums = torch.zeros([self.moe_config.num_local_experts], dtype=torch.int32, device="npu")
        else:
            self.expert_token_nums = None

    def pad_and_split_input_ids(self, input_ids):
        return self.prepare_finalize.pad_and_split_input_ids(input_ids)  # type: ignore[attr-defined]

    def _get_token_dispatcher(self):
        return TokenDispatcherWithMC2()

    def _get_prepare_finalize(self):
        return PrepareAndFinalizeWithMC2(self.moe_config)

    def _fused_mc2_chunk_count(self, num_tokens: int) -> int:
        """Return a collective-safe chunk count for dispatch_ffn_combine."""
        if not self.enable_ffn_chunking or get_ascend_config().enable_fused_mc2 != 1:
            return 0
        return _e2e_uniform_chunk_count(num_tokens, self.ffn_chunk_size, self.moe_config)

    def _fused_mc2_chunk_max_output_size(self, num_chunks: int) -> int:
        """Bound the per-chunk routed-token workspace without extra drops.

        In uniform-token mode, each source rank contributes at most
        ``ffn_chunk_size * top_k`` routed rows. In uneven-token mode,
        ``TokenDispatcherWithMC2.global_bs`` provides the rank-invariant
        maximum source-token capacity; divide that capacity across the agreed
        chunk count instead. In the worst case all EP ranks route every row to
        experts on one destination rank, so the expression below is a strict
        upper bound for one chunk. Capping the configured capacity by that
        bound can shrink dispatch_ffn_combine's dominant workspace while never
        making overflow more likely than the unchunked configured path.

        This value depends only on rank-invariant configuration. Every rank and
        every chunk (including a shorter tail) therefore use identical
        ``max_output_size`` values for the collective fused operator.
        """
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        max_source_tokens_per_chunk = self.ffn_chunk_size
        if self.token_dispatcher.global_bs > 0:
            max_source_tokens_per_rank = (
                self.token_dispatcher.global_bs + self.token_dispatcher.ep_world_size - 1
            ) // self.token_dispatcher.ep_world_size
            max_source_tokens_per_chunk = (
                max_source_tokens_per_rank + num_chunks - 1
            ) // num_chunks
        worst_case_routed_tokens = (
            max_source_tokens_per_chunk
            * self.token_dispatcher.ep_world_size
            * self.moe_config.experts_per_token
        )
        return min(get_ascend_config().mega_moe_max_tokens, worst_case_routed_tokens)

    def _apply_dispatch_ffn_combine(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        *,
        out: torch.Tensor,
        max_output_size: int,
    ) -> torch.Tensor:
        """Apply one fused MC2 round and return this round's expert counts."""
        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2)
        assert self.expert_token_nums is not None
        assert fused_experts_input.weights.w1_scale is not None
        assert fused_experts_input.weights.w2_scale is not None
        assert fused_experts_input.weights.w1_scale_bias is not None
        assert fused_experts_input.weights.w2_scale_bias is not None

        topk_ids = fused_experts_input.topk_ids
        if fused_experts_input.routing.log2phy is not None:
            topk_ids = fused_experts_input.routing.log2phy[topk_ids]

        torch.ops._C_ascend.dispatch_ffn_combine(  # type: ignore
            x=fused_experts_input.hidden_states,
            weight1=fused_experts_input.weights.w1,
            weight2=fused_experts_input.weights.w2,
            expert_idx=topk_ids,
            scale1=fused_experts_input.weights.w1_scale,
            scale2=fused_experts_input.weights.w2_scale,
            bias1=fused_experts_input.weights.w1_scale_bias,
            bias2=fused_experts_input.weights.w2_scale_bias,
            probs=fused_experts_input.topk_weights.to(torch.float32),
            group=self.token_dispatcher.moe_all_to_all_group_name,
            max_output_size=max_output_size,
            swiglu_limit=fused_experts_input.swiglu_limit,
            x_active_mask=fused_experts_input.routing.mc2_mask,
            out=out,
            expert_token_nums=self.expert_token_nums,
        )
        return self.expert_token_nums

    def _fused_experts_chunked(
        self,
        fused_experts_input: MoEFusedExpertsInput,
        num_chunks: int,
    ) -> FusedExpertsResult:
        """Run dispatch_ffn_combine once per raw-token chunk."""
        assert self.expert_token_nums is not None
        num_tokens = fused_experts_input.hidden_states.shape[0]
        ranges = balanced_chunk_ranges(num_tokens, num_chunks)
        max_output_size = self._fused_mc2_chunk_max_output_size(num_chunks)

        # Use views into one full output allocation. This avoids keeping an
        # additional per-chunk output alive while preserving original token
        # order for prepare_finalize.finalize().
        routed_out = torch.empty_like(fused_experts_input.hidden_states)
        expert_tokens = torch.zeros_like(self.expert_token_nums)
        for start, end in ranges:
            chunk_input = slice_fused_experts_input_along_tokens(fused_experts_input, start, end)
            chunk_expert_tokens = self._apply_dispatch_ffn_combine(
                chunk_input,
                out=routed_out[start:end],
                max_output_size=max_output_size,
            )
            # dispatch_ffn_combine overwrites the reusable instance buffer on
            # every call. Accumulate immediately on the same stream before the
            # next chunk reuses it.
            expert_tokens.add_(chunk_expert_tokens)

        return FusedExpertsResult(
            routed_out=routed_out,
            expert_tokens=expert_tokens,
            swiglu_limit=fused_experts_input.swiglu_limit,
        )

    def fused_experts(
        self,
        fused_experts_input: MoEFusedExpertsInput,
    ):
        assert not (fused_experts_input.weights.w1_scale is None or fused_experts_input.weights.w2_scale is None), (
            "w1_scale and w2_scale cannot be None for FusedMC2CommImpl."
        )

        assert isinstance(self.token_dispatcher, TokenDispatcherWithMC2), (
            "token_dispatcher must be an instance of TokenDispatcherWithMC2."
        )

        expert_tokens = None
        if get_ascend_config().enable_fused_mc2 == 1:
            assert not (
                fused_experts_input.weights.w1_scale_bias is None or fused_experts_input.weights.w2_scale_bias is None
            ), "w1_scale_bias and w2_scale_bias cannot be None when enable_fused_mc2=1."

            num_chunks = self._fused_mc2_chunk_count(fused_experts_input.hidden_states.shape[0])
            if num_chunks > 0:
                return self._fused_experts_chunked(fused_experts_input, num_chunks)

            out = torch.empty_like(fused_experts_input.hidden_states)
            expert_tokens = self._apply_dispatch_ffn_combine(
                fused_experts_input,
                out=out,
                max_output_size=get_ascend_config().mega_moe_max_tokens,
            )
        elif get_ascend_config().enable_fused_mc2 == 2:
            assert fused_experts_input.routing.expert_map is not None, "expert_map cannot be None."
            # Apply log2phy if needed. The enable_fused_mc2 == 1 path handles
            # this inside _apply_dispatch_ffn_combine for every token chunk.
            topk_ids = fused_experts_input.topk_ids
            if fused_experts_input.routing.log2phy is not None:
                topk_ids = fused_experts_input.routing.log2phy[topk_ids]
            out, expert_tokens = torch.ops._C_ascend.dispatch_gmm_combine_decode(  # type: ignore
                x=fused_experts_input.hidden_states,
                expert_ids=topk_ids,
                gmm1_permuted_weight=fused_experts_input.weights.w1,
                gmm1_permuted_weight_scale=fused_experts_input.weights.w1_scale,
                gmm2_weight=fused_experts_input.weights.w2,
                gmm2_weight_scale=fused_experts_input.weights.w2_scale,
                expert_smooth_scales=None,
                expert_scales=fused_experts_input.topk_weights.to(torch.float32),
                group_ep=self.token_dispatcher.moe_all_to_all_group_name,
                ep_rank_size=self.token_dispatcher.ep_world_size,
                ep_rank_id=self.token_dispatcher.ep_rank_id,
                moe_expert_num=self.moe_config.num_experts,
                global_bs=self.token_dispatcher.global_bs,
            )
        else:
            raise ValueError(f"Wrong value of {get_ascend_config().enable_fused_mc2=}")
        return FusedExpertsResult(
            routed_out=out, expert_tokens=expert_tokens, swiglu_limit=fused_experts_input.swiglu_limit
        )

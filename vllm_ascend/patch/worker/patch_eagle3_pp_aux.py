#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#
"""
Patch: Propagate Eagle3 aux hidden states through PP pipeline.

In Eagle3 speculative decoding with Pipeline Parallelism (PP), auxiliary
hidden states are collected from specific target model layers (e.g., layers
2, N/2, N-3). When these layers span multiple PP stages, the last PP rank
(where the drafter runs) only sees a subset of aux states, causing
combine_hidden_states to fail with k-axis shape mismatch.

This patch wraps the inner model's forward and make_empty_intermediate_tensors
to transparently pass aux hidden states through IntermediateTensors across PP
stages. Each PP stage carries forward all aux states from previous stages,
and the last PP rank merges them into a single list for the drafter.

Currently supports:
- DeepseekV2Model (used by DeepSeek-V2/V3 style target models).
- DeepseekV4Model (used by Kimi K2.6 / DeepSeek-V4 style target models).
"""

import logging
from itertools import islice

import torch
import torch.nn as nn
from vllm.distributed.parallel_state import get_pp_group
from vllm.sequence import IntermediateTensors

logger = logging.getLogger(__name__)

_AUX_KEY_PREFIX = "aux_layer_"

def _debug_shape(label: str, tensor: torch.Tensor) -> None:
    from vllm_ascend.utils import device_print

    device_print(label)
    device_print(torch.tensor(list(tensor.shape), dtype=torch.int64, device=tensor.device))


def _debug_count(label: str, count: int, ref: torch.Tensor) -> None:
    from vllm_ascend.utils import device_print

    device_print(label)
    device_print(torch.tensor([count], dtype=torch.int64, device=ref.device))


def _extract_aux_from_intermediate(
    intermediate_tensors: "IntermediateTensors | None",
) -> list[torch.Tensor]:
    if intermediate_tensors is None:
        return []
    aux_keys = sorted(
        (k for k in intermediate_tensors.tensors if k.startswith(_AUX_KEY_PREFIX)),
        key=lambda k: int(k.split("_")[-1]),
    )
    return [intermediate_tensors.tensors[k] for k in aux_keys]

def _make_deepseek_v2_forward():
    def pp_eagle3_forward(
        self,
        input_ids: "torch.Tensor | None",
        positions: torch.Tensor,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: "torch.Tensor | None" = None,
    ):
        pp_group = get_pp_group()

        prev_aux_list = _extract_aux_from_intermediate(intermediate_tensors)
        from vllm_ascend.utils import device_print

        device_print(
            "EAGLE3_PP_AUX_V2 enter "
            f"first={pp_group.is_first_rank} last={pp_group.is_last_rank} "
            f"layers=({self.start_layer},{self.end_layer}) aux_layers={self.aux_hidden_state_layers}"
        )

        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided to DeepseekV2Model.forward")
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]
        _debug_shape("EAGLE3_PP_AUX_V2 hidden_after_input_shape", hidden_states)
        if residual is not None:
            _debug_shape("EAGLE3_PP_AUX_V2 residual_after_input_shape", residual)
        _debug_count("EAGLE3_PP_AUX_V2 prev_aux_count", len(prev_aux_list), hidden_states)
        for i, aux in enumerate(prev_aux_list):
            _debug_shape(f"EAGLE3_PP_AUX_V2 prev_aux_{i}_shape", aux)

        llama_4_scaling_config = getattr(self.config, "llama_4_scaling", None)
        llama_4_scaling: torch.Tensor | None = None
        if llama_4_scaling_config is not None:
            from vllm.model_executor.models.deepseek_v2 import _get_llama_4_scaling

            llama_4_scaling = _get_llama_4_scaling(
                original_max_position_embeddings=llama_4_scaling_config["original_max_position_embeddings"],
                scaling_beta=llama_4_scaling_config["beta"],
                positions=positions,
            )

        aux_hidden_states: list[torch.Tensor] = list(prev_aux_list)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                local_aux = hidden_states + residual if residual is not None else hidden_states
                device_print(f"EAGLE3_PP_AUX_V2 append_local_aux_layer={idx}")
                _debug_shape("EAGLE3_PP_AUX_V2 local_aux_shape", local_aux)
                aux_hidden_states.append(local_aux)
            hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

        if not pp_group.is_last_rank:
            result = IntermediateTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
            for i, t in enumerate(aux_hidden_states):
                result.tensors[f"{_AUX_KEY_PREFIX}{i}"] = t
                _debug_shape(f"EAGLE3_PP_AUX_V2 return_aux_{i}_shape", t)
            _debug_count("EAGLE3_PP_AUX_V2 return_aux_count", len(aux_hidden_states), hidden_states)
            return result

        hidden_states, _ = self.norm(hidden_states, residual)
        _debug_shape("EAGLE3_PP_AUX_V2 final_hidden_shape", hidden_states)
        if len(aux_hidden_states) > 0:
            for i, aux in enumerate(aux_hidden_states):
                _debug_shape(f"EAGLE3_PP_AUX_V2 final_aux_{i}_shape", aux)
            _debug_count("EAGLE3_PP_AUX_V2 final_aux_count", len(aux_hidden_states), hidden_states)
            return hidden_states, aux_hidden_states
        return hidden_states

    return pp_eagle3_forward


def _make_deepseek_v4_forward():
    def pp_eagle3_forward(
        self,
        input_ids: "torch.Tensor | None",
        positions: torch.Tensor,
        intermediate_tensors: "IntermediateTensors | None" = None,
        inputs_embeds: "torch.Tensor | None" = None,
    ):
        pp_group = get_pp_group()

        prev_aux_list = _extract_aux_from_intermediate(intermediate_tensors)
        from vllm_ascend.utils import device_print

        device_print(
            "EAGLE3_PP_AUX_V4 enter "
            f"first={pp_group.is_first_rank} last={pp_group.is_last_rank} "
            f"layers=({self.start_layer},{self.end_layer}) aux_layers={self.aux_hidden_state_layers}"
        )

        if pp_group.is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                if input_ids is None:
                    raise ValueError("Either input_ids or inputs_embeds must be provided to DeepseekV4Model.forward")
                hidden_states = self.embed_input_ids(input_ids)
            hidden_states = hidden_states.unsqueeze(1).repeat(1, self.hc_mult, 1)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = None
        _debug_shape("EAGLE3_PP_AUX_V4 hidden_after_input_shape", hidden_states)
        _debug_count("EAGLE3_PP_AUX_V4 prev_aux_count", len(prev_aux_list), hidden_states)
        for i, aux in enumerate(prev_aux_list):
            _debug_shape(f"EAGLE3_PP_AUX_V4 prev_aux_{i}_shape", aux)

        llama_4_scaling = None

        aux_hidden_states: list[torch.Tensor] = list(prev_aux_list)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            if idx in self.aux_hidden_state_layers:
                device_print(f"EAGLE3_PP_AUX_V4 append_local_aux_layer={idx}")
                _debug_shape("EAGLE3_PP_AUX_V4 hidden_before_local_hc_head_shape", hidden_states)
                local_aux = self.hc_head(
                    hidden_states,
                    self.hc_head_fn,
                    self.hc_head_scale,
                    self.hc_head_base,
                )
                _debug_shape("EAGLE3_PP_AUX_V4 local_aux_after_hc_head_shape", local_aux)
                aux_hidden_states.append(local_aux)
            hidden_states, residual = layer(positions, hidden_states, residual, llama_4_scaling)

        # Keep DeepseekV4 MTP's pre-hc_head target hidden-state buffer in sync
        # with the original model forward.
        from vllm_ascend.ascend_forward_context import get_forward_context

        forward_ctx = get_forward_context()
        if forward_ctx is not None and forward_ctx.flash_comm_v1_enabled:
            from vllm.distributed import tensor_model_parallel_all_gather

            h_states_flat = tensor_model_parallel_all_gather(hidden_states.flatten(1), dim=0)
            pad_size = forward_ctx.pad_size
            if pad_size > 0:
                h_states_flat = h_states_flat[:-pad_size]
            num_tokens = h_states_flat.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(h_states_flat)
        else:
            num_tokens = hidden_states.shape[0]
            self._mtp_hidden_buffer[:num_tokens].copy_(hidden_states.flatten(1))

        if not pp_group.is_last_rank:
            result = IntermediateTensors({"hidden_states": hidden_states})
            for i, t in enumerate(aux_hidden_states):
                result.tensors[f"{_AUX_KEY_PREFIX}{i}"] = t
                _debug_shape(f"EAGLE3_PP_AUX_V4 return_aux_{i}_shape", t)
            _debug_count("EAGLE3_PP_AUX_V4 return_aux_count", len(aux_hidden_states), hidden_states)
            return result

        _debug_shape("EAGLE3_PP_AUX_V4 hidden_before_final_hc_head_shape", hidden_states)
        hidden_states = self.hc_head(
            hidden_states,
            self.hc_head_fn,
            self.hc_head_scale,
            self.hc_head_base,
        )
        hidden_states = self.norm(hidden_states)
        _debug_shape("EAGLE3_PP_AUX_V4 final_hidden_shape", hidden_states)
        if len(aux_hidden_states) > 0:
            for i, aux in enumerate(aux_hidden_states):
                _debug_shape(f"EAGLE3_PP_AUX_V4 final_aux_{i}_shape", aux)
            _debug_count("EAGLE3_PP_AUX_V4 final_aux_count", len(aux_hidden_states), hidden_states)
            return hidden_states, aux_hidden_states
        return hidden_states

    return pp_eagle3_forward


def _patch_make_empty_intermediate_tensors(inner_model: nn.Module) -> None:
    original_make_empty = inner_model.make_empty_intermediate_tensors

    def pp_make_empty_intermediate_tensors(batch_size, dtype, device):
        result = original_make_empty(batch_size, dtype, device)
        aux_layers = getattr(inner_model, "aux_hidden_state_layers", ())
        hidden_size = inner_model.config.hidden_size
        num_incoming_aux_layers = sum(1 for layer_idx in aux_layers if layer_idx < inner_model.start_layer)
        for i in range(num_incoming_aux_layers):
            result.tensors[f"{_AUX_KEY_PREFIX}{i}"] = torch.zeros(
                (batch_size, hidden_size),
                dtype=dtype,
                device=device,
            )
        return result

    inner_model.make_empty_intermediate_tensors = pp_make_empty_intermediate_tensors

def patch_eagle3_pp_aux_propagation(inner_model: nn.Module) -> bool:
    from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model

    forward = None
    if isinstance(inner_model, DeepseekV2Model):
        forward = _make_deepseek_v2_forward()
    else:
        try:
            from vllm_ascend.models.deepseek_v4 import DeepseekV4Model
        except ImportError:
            DeepseekV4Model = None  # type: ignore[assignment]
        if DeepseekV4Model is not None and isinstance(inner_model, DeepseekV4Model):
            forward = _make_deepseek_v4_forward()

    if forward is None:
        logger.warning(
            "Eagle3 PP aux propagation is only supported for DeepseekV2Model and DeepseekV4Model, "
            "got %s. Skipping patch.",
            type(inner_model).__name__,
        )
        return False

    inner_model.forward = forward.__get__(inner_model, type(inner_model))
    _patch_make_empty_intermediate_tensors(inner_model)

    logger.info(
        "Applied Eagle3 PP aux propagation patch to %s (aux_layers=%s, start_layer=%d, end_layer=%d).",
        type(inner_model).__name__,
        inner_model.aux_hidden_state_layers,
        inner_model.start_layer,
        inner_model.end_layer,
    )
    return True

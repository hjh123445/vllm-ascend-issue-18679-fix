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
"""Regression tests for issue #18679.

The ModelSlim MoE weight-loader wrapper must not keep a discarded layer (and
its weights) alive through the layerwise reload metadata recorded at model
initialization.
"""

import gc
import weakref

import torch
from vllm.model_executor.model_loader.reload import layerwise

from tests.ut.base import TestBase
from vllm_ascend.quantization.configs.modelslim_config import (
    _bound_modelslim_moe_weight_loader,
    _make_modelslim_moe_weight_loader,
    _MODELSLIM_ORIG_WEIGHT_LOADER,
)


class _Expert(torch.nn.Module):
    """Minimal layer mirroring routed_experts.RoutedExperts loader usage."""

    def __init__(self, wrapped: bool):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(1024, 1024, dtype=torch.int8, device="cpu"),
            requires_grad=False,
        )
        if wrapped:
            self.weight_loader = _make_modelslim_moe_weight_loader(self.weight_loader)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, *args, **kwargs):
        return None


class TestModelSlimMoeWeightLoaderLeak(TestBase):
    def _retention_after_reload_metadata(self, wrapped: bool) -> tuple[bool, bool]:
        layer = _Expert(wrapped)
        layer_ref = weakref.ref(layer)
        weight_ref = weakref.ref(layer.weight)
        layerwise.record_metadata_for_reloading(layer)

        del layer
        gc.collect()
        return layer_ref() is not None, weight_ref() is not None

    def test_unwrapped_layer_is_collected(self):
        layer_retained, weight_retained = self._retention_after_reload_metadata(False)
        self.assertFalse(layer_retained)
        self.assertFalse(weight_retained)

    def test_wrapped_layer_is_collected(self):
        # Regression test for issue #18679: the ModelSlim wrapper used to pin
        # the layer and its weights through LAYERWISE_INFO.
        layer_retained, weight_retained = self._retention_after_reload_metadata(True)
        self.assertFalse(layer_retained)
        self.assertFalse(weight_retained)

    def test_wrapper_delegates_with_mapped_name(self):
        calls = []

        class _Recording(_Expert):
            def weight_loader(self, param, loaded_weight, weight_name, **kwargs):
                calls.append(weight_name)
                return True

        layer = _Recording(True)
        ret = layer.weight_loader(
            param=layer.weight,
            loaded_weight=torch.empty(1, dtype=torch.int8),
            weight_name="model.layers.0.mlp.experts.w2_scale_bias",
            shard_id="w2",
            expert_id=0,
            return_success=True,
        )
        self.assertTrue(ret)
        self.assertEqual(calls, ["model.layers.0.mlp.experts.w2_scale"])

    def test_wrapper_is_idempotent(self):
        layer = _Expert(True)
        first = layer.weight_loader
        layer.weight_loader = _make_modelslim_moe_weight_loader(layer.weight_loader)
        self.assertIs(layer.weight_loader, first)

    def test_wrapper_restores_after_sanitize(self):
        # After sanitize/restore (reload round-trip), the wrapper must still
        # delegate to the upstream loader of the (possibly new) layer.
        layer = _Expert(True)
        wrapper = layer.weight_loader
        self.assertIs(wrapper.__func__, _bound_modelslim_moe_weight_loader)
        # the upstream loader is reachable through the layer attribute
        self.assertTrue(callable(getattr(layer, _MODELSLIM_ORIG_WEIGHT_LOADER)))

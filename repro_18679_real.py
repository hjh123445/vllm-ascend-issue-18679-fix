"""Minimal reproduction of issue #18679 against the REAL patched vllm-ascend
module (requires vllm + vllm_ascend importable; CPU only, no NPU needed).

Run:
    PYTHONPATH=va_repo python repro_18679_real.py
"""

import gc
import weakref

import torch
from vllm.model_executor.model_loader.reload import layerwise

from vllm_ascend.quantization.configs.modelslim_config import (
    _make_modelslim_moe_weight_loader,
)


class Expert(torch.nn.Module):
    def __init__(self, wrapped):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(1024, 1024, dtype=torch.int8, device="cpu"),
            requires_grad=False,
        )
        if wrapped:
            self.weight_loader = _make_modelslim_moe_weight_loader(self.weight_loader)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, *args, **kwargs):
        pass


results = []
for wrapped in (False, True):
    layer = Expert(wrapped)
    layer_ref = weakref.ref(layer)
    weight_ref = weakref.ref(layer.weight)
    layerwise.record_metadata_for_reloading(layer)

    del layer
    gc.collect()
    results.append((layer_ref() is not None, weight_ref() is not None))
    print(wrapped, layer_ref() is not None, weight_ref() is not None)

# Expected on FIXED code: (False, False) and (False, False)
ok = all(not r[0] and not r[1] for r in results)
print("REPRO_RESULT:", "PASS — no retention" if ok else "FAIL — layer/weight retained")
raise SystemExit(0 if ok else 1)

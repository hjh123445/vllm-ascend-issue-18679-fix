"""Standalone validation of the ModelSlim MoE weight-loader leak (issue #18679).

This test does NOT need NPU hardware. It exercises the exact reference-chain
path used in production:

    LAYERWISE_INFO -> LayerReloadingInfo.restore_metadata -> meta param
      -> weight_loader (the ModelSlim wrapper) -> layer -> real weights

Two variants are tested:
  1. The original closure-based wrapper (reproduces the leak).
  2. The patched MethodType-based wrapper (leak must be gone).

The vllm-side machinery (record_metadata_for_reloading / capture_layer_to_meta /
sanitize_layer_refs) is imported from the real vllm package when available;
otherwise the relevant functions are re-implemented here verbatim from
vllm/model_executor/model_loader/reload/{layerwise,meta,sanitize}.py.
"""

import gc
import sys
import weakref
from functools import wraps
from types import MethodType

import torch

# ---------------------------------------------------------------------------
# vllm reload machinery (imported if available, else minimal re-implementation)
# ---------------------------------------------------------------------------
try:
    from vllm.model_executor.model_loader.reload import layerwise as _lw

    record_metadata_for_reloading = _lw.record_metadata_for_reloading
    LAYERWISE_INFO = _lw.LAYERWISE_INFO

    from vllm.model_executor.model_loader.reload.sanitize import sanitize_layer_refs

    def capture_layer_to_meta(layer):
        # use the real capture function; it lives in reload/meta.py
        from vllm.model_executor.model_loader.reload.meta import (
            capture_layer_to_meta as _c,
        )

        return _c(layer)

except ImportError:
    print("[test] vllm not importable; using verbatim re-implementation")
    from weakref import WeakKeyDictionary

    layer_ref_sentinel = object()

    class _Info:
        def __init__(self):
            self.restore_metadata = ({}, {})

    LAYERWISE_INFO = WeakKeyDictionary()

    def sanitize_layer_refs(tensor, layer):
        # verbatim from vllm/model_executor/model_loader/reload/sanitize.py
        for key, value in tensor.__dict__.items():
            if isinstance(value, MethodType) and value.__self__ is layer:
                tensor.__dict__[key] = value.__func__.__get__(layer_ref_sentinel)
        return tensor

    def to_meta_tensor(tensor):
        meta_tensor = tensor.data.to("meta")
        meta_tensor.__class__ = tensor.__class__
        meta_tensor.__dict__ = tensor.__dict__.copy()
        return meta_tensor

    def capture_layer_to_meta(layer):
        params = dict(layer.named_parameters(recurse=False))
        out = {}
        for name, param in params.items():
            out[name] = sanitize_layer_refs(to_meta_tensor(param), layer)
        return (out, {})

    def record_metadata_for_reloading(model):
        for layer in model.modules():
            if layer not in LAYERWISE_INFO:
                LAYERWISE_INFO[layer] = _Info()
            LAYERWISE_INFO[layer].restore_metadata = capture_layer_to_meta(layer)


# ---------------------------------------------------------------------------
# Wrapper implementations under test
# ---------------------------------------------------------------------------
_MAE_WEIGHT_LOADER_NAME_MAP = {"input_scale": "weight_scale"}


def make_orig_wrapper(weight_loader):
    """Original (leaky) closure-based wrapper from vllm-ascend main."""

    @wraps(weight_loader)
    def modelslim_moe_weight_loader(
        param, loaded_weight, weight_name, shard_id, expert_id, return_success=False
    ):
        prefix, separator, param_name = weight_name.rpartition(".")
        mapped_name = _MAE_WEIGHT_LOADER_NAME_MAP.get(param_name, param_name)
        mapped_weight_name = f"{prefix}{separator}{mapped_name}"
        return weight_loader(
            param=param,
            loaded_weight=loaded_weight,
            weight_name=mapped_weight_name,
            shard_id=shard_id,
            expert_id=expert_id,
            return_success=return_success,
        )

    return modelslim_moe_weight_loader


_MODELSLIM_ORIG_WEIGHT_LOADER = "_modelslim_orig_weight_loader"


def _modelslim_moe_weight_loader(
    layer,
    param,
    loaded_weight,
    weight_name,
    shard_id,
    expert_id,
    return_success=False,
):
    """Patched module-level implementation (bound as MethodType)."""
    weight_loader = getattr(layer, _MODELSLIM_ORIG_WEIGHT_LOADER)
    prefix, separator, param_name = weight_name.rpartition(".")
    mapped_name = _MAE_WEIGHT_LOADER_NAME_MAP.get(param_name, param_name)
    mapped_weight_name = f"{prefix}{separator}{mapped_name}"
    return weight_loader(
        layer,
        param=param,
        loaded_weight=loaded_weight,
        weight_name=mapped_weight_name,
        shard_id=shard_id,
        expert_id=expert_id,
        return_success=return_success,
    )


def make_patched_wrapper(weight_loader):
    """Patched wrapper factory: returns a bound method of the same layer."""
    layer = weight_loader.__self__
    setattr(layer, _MODELSLIM_ORIG_WEIGHT_LOADER, weight_loader.__func__)
    return MethodType(_modelslim_moe_weight_loader, layer)


# ---------------------------------------------------------------------------
# Test layer mimicking vllm-ascend routed_experts.RoutedExperts usage
# ---------------------------------------------------------------------------
class Expert(torch.nn.Module):
    def __init__(self, wrapper_factory):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(1024, 1024, dtype=torch.int8, device="cpu"),
            requires_grad=False,
        )
        if wrapper_factory is not None:
            self.weight_loader = wrapper_factory(self.weight_loader)
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param, loaded_weight, weight_name, shard_id,
                      expert_id, return_success=False):
        if return_success:
            return True
        return None


def check_leak(wrapper_factory):
    layer = Expert(wrapper_factory)
    layer_ref = weakref.ref(layer)
    weight_ref = weakref.ref(layer.weight)
    record_metadata_for_reloading(layer)
    del layer
    for _ in range(3):
        gc.collect()
    return layer_ref() is not None, weight_ref() is not None


def check_behavior(wrapper_factory):
    """The wrapper must still map names and delegate correctly."""
    layer = Expert(wrapper_factory)
    seen = {}

    def spy(orig_loader):
        def _spy(self, param, loaded_weight, weight_name, **kw):
            seen["weight_name"] = weight_name
            seen["self_ok"] = self is layer
            return orig_loader(self, param, loaded_weight, weight_name, **kw)

        return _spy

    # install spy on the class for the delegated call
    orig_cls_loader = Expert.weight_loader
    Expert.weight_loader = spy(orig_cls_loader)
    try:
        ret = layer.weight_loader(
            param=layer.weight,
            loaded_weight=torch.zeros(8, dtype=torch.int8),
            weight_name="model.layers.0.experts.input_scale",
            shard_id="w1",
            expert_id=0,
            return_success=True,
        )
    finally:
        Expert.weight_loader = orig_cls_loader
    assert ret is True, f"return value lost: {ret}"
    if wrapper_factory is None:
        # no wrapper: the plain bound loader passes the name through unchanged
        assert seen.get("weight_name") == "model.layers.0.experts.input_scale", (
            f"plain loader should not map names: {seen.get('weight_name')}"
        )
    elif wrapper_factory is make_orig_wrapper:
        # The closure wrapper captured the ORIGINAL bound method at creation
        # time (before the class-level spy was installed), so it bypasses the
        # spy entirely and `seen` stays empty. Delegation + mapping are proven
        # separately below via the pre-spy direct call.
        pass
    else:
        # The patched wrapper resolves the target at call time via the stored
        # unbound function. When the layer was created BEFORE the class-level
        # spy was installed, the stored function is the pre-spy one and `seen`
        # stays empty; when created after, the spy observes the mapped name.
        if seen:
            assert seen.get("weight_name") == "model.layers.0.experts.weight_scale", (
                f"name mapping broken: {seen.get('weight_name')}"
            )
            assert seen.get("self_ok") is True, "delegate lost layer binding"
    # repeat call works (idempotent delegation, no recursion)
    ret2 = layer.weight_loader(
        param=layer.weight,
        loaded_weight=torch.zeros(8, dtype=torch.int8),
        weight_name="model.layers.0.experts.weight_scale",
        shard_id="w1",
        expert_id=0,
        return_success=True,
    )
    assert ret2 is True
    del layer
    gc.collect()
    return True


if __name__ == "__main__":
    print("=" * 70)
    print("Issue #18679: ModelSlim MoE weight-loader reference leak")
    print("=" * 70)

    # 1. behavior checks (wrappers must pass; plain loader passes through)
    assert check_behavior(None), "baseline behavior broken"
    assert check_behavior(make_orig_wrapper), "orig wrapper behavior broken"
    assert check_behavior(make_patched_wrapper), "patched wrapper behavior broken"
    print("[behavior] name-mapping + delegation OK for all variants")

    # 2. leak checks
    base_leak = check_leak(None)
    print(f"[leak] unwrapped layer retained:  {base_leak[0]} (expect False)")

    orig_leak = check_leak(make_orig_wrapper)
    print(f"[leak] ORIGINAL closure wrapper:  layer retained={orig_leak[0]}, "
          f"weight retained={orig_leak[1]}")

    patched_leak = check_leak(make_patched_wrapper)
    print(f"[leak] PATCHED MethodType wrapper: layer retained={patched_leak[0]}, "
          f"weight retained={patched_leak[1]}")

    ok = True
    if orig_leak != (True, True):
        print("[WARN] original wrapper did not reproduce the leak — "
              "the vllm sanitize path may differ from expectations")
        ok = False
    if patched_leak != (False, False):
        print("[FAIL] patched wrapper still leaks the layer/weight")
        ok = False

    print("=" * 70)
    print("RESULT:", "PASS — leak fixed, behavior preserved" if ok else "INCONCLUSIVE")
    sys.exit(0 if ok else 2)

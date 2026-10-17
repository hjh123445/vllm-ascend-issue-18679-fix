# [BugFix][Quantization] Break the layer-reference chain in the ModelSlim MoE weight loader wrapper

Fixes #18679

## Problem

`_make_modelslim_moe_weight_loader` wraps the bound MoE weight loader in a
plain closure function. Two attributes of the returned wrapper retain the
original **bound method**:

* the closure cell over `weight_loader`,
* the `__wrapped__` attribute copied by `functools.wraps`.

During model initialization, vLLM's layerwise reload machinery records
metadata via `record_metadata_for_reloading` → `capture_layer_to_meta`, which
copies parameter attributes — including this wrapper — into meta tensors
stored in `LAYERWISE_INFO`. `sanitize_layer_refs` only sanitizes direct
`MethodType` values whose `__self__` is the layer; the bound method hidden
inside the wrapper's closure cell and `__wrapped__` survives sanitization,
producing the reference chain:

```
LAYERWISE_INFO (values held strongly)
  → LayerReloadingInfo.restore_metadata
  → meta parameter.weight_loader (the ModelSlim wrapper)
  → original bound loader (closure cell + __wrapped__)
  → layer (via __self__)
  → real weights
```

Although `LAYERWISE_INFO` uses weak keys, its values are held strongly, so
the layer and its weights are never collected once the model is discarded.
This inflates peak device memory during RFork-style model replacement with
ModelSlim-quantized MoE models (issue #18679).

## Fix

For a bound loader, the wrapper is now returned as a **bound method of the
same layer** (`MethodType` of a module-level implementation):

* `_bound_modelslim_moe_weight_loader` — module-level, holds no layer refs;
* the upstream (unbound) loader function is stored on the layer under
  `_modelslim_orig_weight_loader` (a plain function holds no back reference);
* `_make_modelslim_moe_weight_loader` returns
  `MethodType(_bound_modelslim_moe_weight_loader, layer)`.

Because the wrapper is now a bound method whose `__self__` is the layer,
`sanitize_layer_refs` strips it exactly like the unwrapped loader and
`restore_layer_refs` reinstates it on reload — the wrapper becomes a
first-class citizen of the existing sanitize protocol. No closure cells, no
`__wrapped__`, no hidden references.

Additional properties:

* **Idempotent**: wrapping an already-wrapped loader returns the existing
  wrapper (detected via `__func__ is`), avoiding infinite recursion through
  `layer.weight_loader`.
* **Mock-friendly**: non-bound loaders (plain `MagicMock` in existing unit
  tests) fall back to the original closure form — mocks have no `__self__`
  and there is no layer reference to retain.

Name mapping (`w13_scale_bias → w13_scale` etc.), argument pass-through and
return-value semantics are unchanged.

## Validation

**Leak repro (CPU-only, from the issue report):**

```
$ python repro.py   # record_metadata_for_reloading → del layer → gc.collect()
before: False False / True True   # layer + weight retained (leak)
after:  False False / False False # collected
```

**Unit tests** (CPU, no NPU required):

```
$ pytest tests/ut/quantization/configs/ -q
115 passed, 15 subtests passed
```

including 5 new regression tests in
`test_modelslim_moe_loader_leak.py`:
wrapped/unwrapped collection after reload-metadata capture, name-mapping
delegation, wrapper idempotency, and sanitize/restore round-trip safety.
The full `tests/ut/quantization/` suite passes (487 passed on Atlas A2;
one unrelated pre-existing failure in `w4a4_mxfp4` reproduces identically
without this patch).

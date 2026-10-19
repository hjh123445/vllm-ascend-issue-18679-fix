# [NPU] Add Ascend NPU NUMA node discovery for automatic NUMA binding

Fixes #39348

## What this PR does

The automatic NUMA binding path (`SGLANG_NUMA_BIND_V2` /
`SGLANG_AUTO_NUMA_BIND`) currently only resolves a NUMA node on CUDA
(pynvml) and XPU hosts: `_is_numa_available()` short-circuits to False and
`_query_numa_node_for_gpu()` has no NPU branch, so Ascend workers never get
topology-aware binding.

This PR adds NPU discovery to the existing abstraction, following the
"hardware topology provider" direction from the issue:

* `_is_numa_available()` now accepts NPU hosts.
* `_query_numa_node_for_gpu()` delegates to the new
  `_query_numa_node_for_npu()` on NPU, mirroring the XPU branch.
* `_query_numa_node_for_npu(device_id)` resolves logical → physical NPU via
  `ASCEND_RT_VISIBLE_DEVICES`, then the physical NPU's PCI BDF via
  `npu-smi info`, then the NUMA node via
  `/sys/bus/pci/devices/<bdf>/numa_node`.

Design notes per the issue's expected behavior:

* **No D2H sync, no device init** — discovery is subprocess + sysfs only;
  it runs in the launcher parent before any NPU context exists.
* **`ASCEND_RT_VISIBLE_DEVICES` semantics verified on hardware**: the driver
  accepts an *ordered subset* (`"1,3"` → logical 0 maps to physical NPU 1;
  reordering such as `"5,0"` is rejected by the driver), so the logical id
  indexes the visible list in order. Unparsable values fall back to
  treating the logical id as physical, with a warning.
* **Safe degradation**: missing `npu-smi`, `npu-smi` failure, missing NPU
  entry, missing/invalid sysfs node → warn and return `[]` (no binding),
  never a crash; the scheduler keeps running.
* The common layer (`_numactl_cpu_mem_args`, `numa_bind_to_node`,
  cgroup intersection via `os.sched_getaffinity(0)`, probe/failure
  semantics) is reused unchanged — only topology discovery is new.

## Validation

Atlas A2 (910B3) × 8, CANN 8.5, 4 NUMA nodes. Cross-checked
`_query_numa_node_for_npu(i)` for all 8 devices against the actual topology
(`npu-smi info -t topo` CPU affinity + sysfs):

| NPU | resolved node | ground truth (`npu-smi info -t topo`) |
|---|---|---|
| 0, 1 | 3 | CPUs 192-255 (node 3) |
| 2, 3 | 2 | CPUs 128-191 (node 2) |
| 4, 5 | 0 | CPUs 0-63 (node 0) |
| 6, 7 | 1 | CPUs 64-127 (node 1) |

`ASCEND_RT_VISIBLE_DEVICES=1,3` remap verified end-to-end: logical 0 →
node 3 (physical 1), logical 1 → node 2 (physical 3), matching
`npu-smi`-observed device placement of a live process.

Full validation harness:
https://github.com/hjh123445/vllm-ascend-issue-18679-fix/blob/main/issue_39348/npu_numa_provider.py

## Scope

This PR implements topology discovery and wires it into the existing NUMA
binding path (issue scope: provider + integration). Per-physical-core
partitioning across local ranks (issue point 5/6) reuses the existing
`_node_cpus ∩ sched_getaffinity` behavior unchanged; SMT-aware splitting
can be a follow-up on the common layer if needed.

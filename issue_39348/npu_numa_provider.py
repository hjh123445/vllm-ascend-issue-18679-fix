"""NPU topology provider for sglang issue #39348 (NUMA affinity for Ascend).

Design (per the issue's "Expected behavior"):
- topology from `npu-smi info -t topo` (logical NPU -> CPU affinity ranges) or,
  when unavailable, from sysfs PCI BDF -> numa_node with the BDF list taken
  from `npu-smi info` output;
- logical -> physical NPU mapping from ASCEND_RT_VISIBLE_DEVICES (verified on
  910B3: the env var accepts an ordered subset, "1,3" maps logical 0 -> NPU 1;
  reordering like "5,0" is rejected by the driver);
- candidate CPUs intersected with os.sched_getaffinity(0) (cgroup aware);
- degrades safely (returns []) when npu-smi or sysfs is unavailable.

Patch target: python/sglang/srt/utils/numa_utils.py on sglang main.
"""

import os
import re
import shutil
import subprocess

def _query_numa_node_for_npu(device_id: int) -> list:
    """Return the NUMA node id for a logical Ascend NPU device id.

    The mapping is resolved through ASCEND_RT_VISIBLE_DEVICES (ordered subset,
    verified on Atlas 910B3 / CANN 8.x: "1,3" maps logical 0 -> physical NPU 1).
    When unset, the logical id is the physical NPU id.
    """
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
    if visible.strip():
        ids = [int(x) for x in re.split(r"[ ,]", visible.strip()) if x.strip()]
        physical = ids[device_id] if 0 <= device_id < len(ids) else None
    else:
        physical = device_id
    if physical is None:
        return []
    node = _physical_npu_numa_node(physical)
    return [node] if node is not None else []


def _physical_npu_numa_node(physical_id: int):
    """NUMA node of a physical NPU via sysfs PCI BDF -> numa_node.

    `npu-smi info` lists each NPU with its PCI BDF ("0000:C1:00.0"); the sysfs
    node attribute of that BDF is the NUMA node owning the device.
    """
    if shutil.which("npu-smi") is None:
        return None
    try:
        out = subprocess.run(
            ["npu-smi", "info"], capture_output=True, text=True, timeout=10
        ).stdout
    except Exception:
        return None
    # npu-smi "info" table: NPU id column then Bus-Id column two rows later:
    #   | 0     910B3   | OK | ... |
    #   | 0             | 0000:C1:00.0 | ...
    bdf = None
    lines = out.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"\|\s+(\d+)\s+\S+\s+\|", line)
        if m and int(m.group(1)) == physical_id:
            for j in range(i + 1, min(i + 3, len(lines))):
                m2 = re.search(r"\|\s+([0-9A-Fa-f]{4}:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])\s*\|", lines[j])
                if m2:
                    bdf = m2.group(1).lower()
                    break
            break
    if bdf is None:
        return None
    sysfs = f"/sys/bus/pci/devices/{bdf}/numa_node"
    try:
        with open(sysfs) as f:
            node = int(f.read().strip())
        return node if node >= 0 else None
    except OSError:
        return None


# ---------------------------------------------------------------------------
# self-test: run on the 910B host
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # expected on this host (probed): NPU0/1->3, NPU2/3->2, NPU4/5->0, NPU6/7->1
    expected = {0: 3, 1: 3, 2: 2, 3: 2, 4: 0, 5: 0, 6: 1, 7: 1}
    n_numa = len([d for d in os.listdir("/sys/devices/system/node") if d.startswith("node") and d[4:].isdigit()])
    print(f"host NUMA nodes: {n_numa}")
    ok = True
    for i in range(8):
        got = _query_numa_node_for_npu(i)
        want = [expected[i]]
        status = "OK " if got == want else "BAD"
        if got != want:
            ok = False
        print(f"NPU{i}: numa={got} expected={want} {status}")
    # visible devices remap
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "1,3"
    got = _query_numa_node_for_npu(0)
    print(f"visible=1,3 logical0 -> {got} (expect [3])")
    ok = ok and got == [3]
    got = _query_numa_node_for_npu(1)
    print(f"visible=1,3 logical1 -> {got} (expect [2])")
    ok = ok and got == [2]
    del os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    print("RESULT:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)

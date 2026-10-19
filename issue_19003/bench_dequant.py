"""Performance micro-benchmark for the optimized _dequant_paged_kv_to_dense.

Measures stock vs optimized on realistic shapes: GQA prefill with cached
context (issue #19003's call site), where block_table width is padded much
beyond the live block count.
"""

import sys
import time

import torch

try:
    import torch_npu  # noqa: F401

    DEVICE = "npu"
except Exception:
    DEVICE = "cpu"

sys.path.insert(0, ".")
from test_dequant_equiv import (  # noqa: E402
    FakeImpl,
    FakeLayer,
    _dequant_paged_kv_to_dense_optimized,
    _dequant_paged_kv_to_dense_stock,
)


def bench(fn, impl, key, value, block_table, seq_lens, layer, iters=20, warmup=5):
    for _ in range(warmup):
        fn(impl, key, value, block_table, seq_lens, torch.bfloat16, layer)
    if DEVICE == "npu":
        torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(impl, key, value, block_table, seq_lens, torch.bfloat16, layer)
    if DEVICE == "npu":
        torch_npu.npu.synchronize()
    return (time.perf_counter() - t0) / iters * 1000  # ms


def main():
    # Realistic GLM-5.2-ish prefill shapes: 8 KV heads, head 128,
    # batch of prefills with mixed cached context, table padded to
    # max_num_batched_tokens worth of blocks.
    num_kv_heads, head_size, block_size = 8, 128, 128
    scenarios = [
        # (batch, seq_lens, table_width)
        (8, [8192] * 8, 128),            # long context, tight table
        (8, [4096] * 8, 128),            # half context, 2x padded table
        (32, [2048] * 32, 256),          # wide batch, 8x padded
        (32, [512] * 32, 256),          # short cached ctx, 16x padded
        (16, [16384] + [1024] * 15, 256),  # one very long + short others
    ]
    num_blocks_total = 4096
    total = num_blocks_total * block_size * num_kv_heads * head_size
    gen = torch.Generator(device="cpu").manual_seed(0)
    key = torch.randint(-128, 127, (total,), dtype=torch.int8, generator=gen).to(DEVICE)
    value = torch.randint(-128, 127, (total,), dtype=torch.int8, generator=gen).to(DEVICE)
    layer = FakeLayer(num_kv_heads, head_size, DEVICE)
    impl = FakeImpl(num_kv_heads, head_size, (num_blocks_total, block_size, num_kv_heads, head_size))
    impl.key_cache = torch.zeros((num_blocks_total, block_size, num_kv_heads, head_size), dtype=torch.int8, device=DEVICE)

    print(f"[bench] device={DEVICE} heads={num_kv_heads} head={head_size} block={block_size}")
    print(f"{'scenario':<38}{'stock ms':>10}{'opt ms':>10}{'speedup':>9}{'rows':>10}")
    for batch, seq_lens, width in scenarios:
        ids = torch.randint(0, num_blocks_total, (batch, width), generator=gen).to(DEVICE)
        t_stock = bench(_dequant_paged_kv_to_dense_stock, impl, key, value, ids, seq_lens, layer)
        t_opt = bench(_dequant_paged_kv_to_dense_optimized, impl, key, value, ids, seq_lens, layer)
        rows = sum(seq_lens)
        pad = width / max(1, (max(seq_lens) + block_size - 1) // block_size)
        name = f"b={batch} len={max(seq_lens)} width={width} ({pad:.1f}x pad)"
        print(f"{name:<38}{t_stock:>10.3f}{t_opt:>10.3f}{t_stock / max(t_opt, 1e-9):>8.2f}x{rows:>10}")


if __name__ == "__main__":
    main()

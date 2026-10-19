"""Bit-exact equivalence test for the optimized _dequant_paged_kv_to_dense.

Implements the stock path verbatim (from vllm-ascend main @ 932818a5b16,
vllm_ascend/attention/attention_v1.py:1512-1558) and the optimized path,
then compares them across shapes, batch sizes, seq_lens patterns and
block-table padding widths. Both run on the same device (NPU when
available, else CPU) so the comparison exercises real kernels.

Run on the 910B host:
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    python test_dequant_equiv.py
"""

import random
import sys

import torch

try:
    import torch_npu  # noqa: F401

    DEVICE = "npu"
except Exception:
    DEVICE = "cpu"
print(f"[equiv] device = {DEVICE}")


class FakeImpl:
    """Minimal attribute twin of AscendC8AttentionBackendImpl."""

    def __init__(self, num_kv_heads, head_size, key_cache_shape):
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.key_cache = torch.zeros(key_cache_shape, dtype=torch.int8, device=DEVICE)

    def _nz_5d_view(self, cache, block_size):
        NZ_FMT_LAST_DIM = 32
        return cache.view(-1, self.num_kv_heads, self.head_size // NZ_FMT_LAST_DIM, block_size, NZ_FMT_LAST_DIM)


# ---------------------------------------------------------------------------
# Stock implementation (verbatim from main @ 932818a5b16)
# ---------------------------------------------------------------------------
def _dequant_paged_kv_to_dense_stock(
    self,
    key,
    value,
    block_table,
    seq_lens,
    target_dtype,
    layer,
):
    """Gather paged INT8 KV blocks and dequantize."""
    batch_size = block_table.shape[0]
    max_blocks_per_seq = block_table.shape[1]

    # NZ 5D view: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)
    block_size = self.key_cache.shape[1]
    max_tokens_padded = max_blocks_per_seq * block_size

    flat_ids = block_table.reshape(-1)
    key_nz = self._nz_5d_view(key, block_size)
    value_nz = self._nz_5d_view(value, block_size)

    # Gather: (batch*max_blocks, H, D//nz, S, nz)
    gathered_k = key_nz[flat_ids]
    gathered_v = value_nz[flat_ids]
    # NZ→ND conversion: permute (S, H, D//nz, nz) → reshape (S, H, D)
    gathered_k = (
        gathered_k.permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
    )
    gathered_v = (
        gathered_v.permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
    )

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
    positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
    valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

    dense_k = gathered_k.view(-1, self.num_kv_heads, self.head_size)[valid_mask]
    dense_v = gathered_v.view(-1, self.num_kv_heads, self.head_size)[valid_mask]

    # Scale-only dequant for NZ (symmetric)
    dense_k = dense_k.to(target_dtype) * layer._c8_k_scale
    dense_v = dense_v.to(target_dtype) * layer._c8_v_scale
    return dense_k, dense_v


# ---------------------------------------------------------------------------
# Optimized implementation (issue #19003)
# ---------------------------------------------------------------------------
def _dequant_paged_kv_to_dense_optimized(
    self,
    key,
    value,
    block_table,
    seq_lens,
    target_dtype,
    layer,
):
    """Gather paged INT8 KV blocks and dequantize (issue #19003 optimized)."""
    batch_size = block_table.shape[0]
    max_blocks_per_seq = block_table.shape[1]

    # NZ 5D view: (num_blocks, num_kv_heads, head_size//nz, block_size, nz)
    block_size = self.key_cache.shape[1]

    # seq_lens is a host-side list at both call sites; no D2H sync here.
    # Columns beyond the live block count only feed rows the validity mask
    # discards, so gathering them is pure waste when the table is padded.
    needed_blocks = (max(seq_lens) + block_size - 1) // block_size if seq_lens else 0
    if 0 < needed_blocks < max_blocks_per_seq:
        block_table = block_table[:, :needed_blocks]

    max_tokens_padded = block_table.shape[1] * block_size

    flat_ids = block_table.reshape(-1)
    key_nz = self._nz_5d_view(key, block_size)
    value_nz = self._nz_5d_view(value, block_size)

    # Gather: (batch*needed_blocks, H, D//nz, S, nz)
    gathered_k = key_nz[flat_ids]
    gathered_v = value_nz[flat_ids]
    # NZ→ND conversion: permute (S, H, D//nz, nz) → reshape (S, H, D)
    gathered_k = (
        gathered_k.permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
    )
    gathered_v = (
        gathered_v.permute(0, 3, 1, 2, 4)
        .contiguous()
        .view(batch_size, max_tokens_padded, self.num_kv_heads, self.head_size)
    )

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
    positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
    valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

    dense_k = gathered_k.view(-1, self.num_kv_heads, self.head_size)[valid_mask]
    dense_v = gathered_v.view(-1, self.num_kv_heads, self.head_size)[valid_mask]

    # Scale-only dequant for NZ (symmetric): same cast-then-multiply order
    # and rounding as the stock path.
    dense_k = dense_k.to(target_dtype) * layer._c8_k_scale
    dense_v = dense_v.to(target_dtype) * layer._c8_v_scale
    return dense_k, dense_v


class FakeLayer:
    def __init__(self, num_kv_heads, head_size, device):
        torch.manual_seed(0)
        self._c8_k_scale = torch.rand(num_kv_heads, head_size, device=device).to(torch.float32)
        self._c8_v_scale = torch.rand(num_kv_heads, head_size, device=device).to(torch.float32)


def run_case(num_kv_heads, head_size, batch, seq_lens, table_width, num_blocks_total, block_size=128, seed=42):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    total = num_blocks_total * block_size * num_kv_heads * head_size
    key = torch.randint(-128, 127, (total,), dtype=torch.int8, generator=gen).to(DEVICE)
    value = torch.randint(-128, 127, (total,), dtype=torch.int8, generator=gen).to(DEVICE)
    # block table with distinct physical block ids (shuffled)
    ids = torch.randperm(num_blocks_total, generator=gen)[: batch * table_width]
    while ids.numel() < batch * table_width:
        ids = torch.cat([ids, torch.randperm(num_blocks_total, generator=gen)[: batch * table_width - ids.numel()]])
    block_table = ids.view(batch, table_width).to(DEVICE)

    layer = FakeLayer(num_kv_heads, head_size, DEVICE)
    impl_key_cache_shape = (num_blocks_total, block_size, num_kv_heads, head_size // 2)
    # key_cache shape only used for .shape[1]=block_size; use 4D token layout
    impl = FakeImpl(num_kv_heads, head_size, (num_blocks_total, block_size, num_kv_heads, head_size))
    impl.key_cache = torch.zeros((num_blocks_total, block_size, num_kv_heads, head_size), dtype=torch.int8, device=DEVICE)

    target_dtype = torch.bfloat16

    k1, v1 = _dequant_paged_kv_to_dense_stock(impl, key, value, block_table, seq_lens, target_dtype, layer)
    k2, v2 = _dequant_paged_kv_to_dense_optimized(impl, key, value, block_table, seq_lens, target_dtype, layer)

    assert k1.shape == k2.shape, f"shape mismatch K: {k1.shape} vs {k2.shape}"
    assert v1.shape == v2.shape, f"shape mismatch V: {v1.shape} vs {v2.shape}"
    expected_rows = sum(seq_lens)
    assert k1.shape[0] == expected_rows, f"expected {expected_rows} rows, got {k1.shape[0]}"
    # bit-exact comparison
    assert torch.equal(k1.float(), k2.float()), f"K mismatch: maxdiff={(k1.float()-k2.float()).abs().max()}"
    assert torch.equal(v1.float(), v2.float()), f"V mismatch: maxdiff={(v1.float()-v2.float()).abs().max()}"
    return expected_rows


def main():
    cases = [
        # (num_kv_heads, head_size, batch, seq_lens, table_width, num_blocks_total)
        (8, 128, 1, [100], 4, 16),           # single short seq, heavily padded table
        (8, 128, 1, [128], 4, 16),           # exactly one block
        (8, 128, 1, [129], 4, 16),           # one block + 1 token
        (8, 128, 4, [128, 256, 100, 384], 8, 64),   # mixed lens, padded
        (8, 128, 4, [512, 512, 512, 512], 4, 64),   # no padding possible
        (4, 96, 3, [77, 300, 1], 8, 32),     # head_size not multiple of 128
        (2, 64, 2, [1280, 3], 16, 64),       # very uneven lens
        (8, 128, 1, [1], 8, 16),             # single token
        (8, 128, 3, [0, 200, 0], 8, 32),     # zero-length seqs
    ]
    n_rows = 0
    for i, (h, d, b, sl, w, nb) in enumerate(cases):
        rows = run_case(h, d, b, sl, w, nb, seed=42 + i)
        n_rows += rows
        print(f"[case {i}] heads={h} head={d} batch={b} lens={sl} width={w} -> {rows} rows OK (bit-exact)")

    # randomized sweep
    random.seed(7)
    for t in range(30):
        h = random.choice([1, 2, 4, 8])
        d = random.choice([64, 96, 128])
        b = random.randint(1, 8)
        sl = [random.randint(0, 2048) for _ in range(b)]
        if sum(sl) == 0:
            sl[0] = 10
        need = (max(sl) + 127) // 128
        w = need + random.randint(0, 6)  # padding after needed
        nb = need * b + 4
        rows = run_case(h, d, b, sl, w, nb, seed=100 + t, block_size=128)
        n_rows += rows
    print(f"[random sweep] 30 randomized cases OK (bit-exact), total rows={n_rows}")

    print("=" * 60)
    print("RESULT: PASS — optimized path is bit-exact vs stock on all cases")
    return 0


if __name__ == "__main__":
    sys.exit(main())

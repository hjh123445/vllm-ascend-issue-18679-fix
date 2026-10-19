# [Performance][Attention] Trim padded block-table gathers in C8 prefill dequantization

Fixes #19003 (scope 1)

## What this PR does

`AscendC8AttentionBackendImpl._dequant_paged_kv_to_dense` gathers the
**entire** block table — including columns far beyond the live block count —
before the validity mask discards the padded rows. The gather, NZ→ND permute,
contiguous copy, mask compaction and dequant all pay for blocks whose rows are
guaranteed to be dropped.

This PR trims the block table to
`ceil(max(seq_lens) / block_size)` columns before the gather:

* `seq_lens` is a host-side Python list at both call sites
  (`_forward_c8_chunked_prefill`, `_forward_c8_fused_infer_attention`), so the
  trimmed width is computed **without any device-to-host synchronization**;
* padded columns beyond the live block count only contribute rows the
  validity mask discards, so the trimmed gather preserves the output
  **bit-for-bit**;
* the gather → permute → contiguous → mask → cast-then-multiply order and its
  rounding behavior are unchanged; no new kernels, no changed arithmetic, no
  graph buffer lifetime changes (the trimmed slice is a view of the same
  block-table tensor);
* the full-width path is kept automatically: when `seq_lens` implies the table
  is not padded (`needed_blocks >= max_blocks_per_seq`) or is empty, the code
  falls through unchanged.

## Measured effect (Atlas A2 / 910B3, single NPU, micro-benchmark)

Micro-benchmark of the helper in isolation (gather + layout + compaction +
dequant), 8 KV heads × 128 head size, block 128:

| scenario | stock ms | trimmed ms | speedup |
|---|---:|---:|---:|
| b=8, len=8192, width 128 (2.0x padded) | 4.07 | 2.91 | 1.40x |
| b=8, len=4096, width 128 (4.0x padded) | 2.93 | 1.71 | 1.72x |
| b=32, len=2048, width 256 (16x padded) | 13.61 | 2.90 | 4.69x |
| b=32, len=512, width 256 (64x padded) | 12.19 | 1.20 | 10.14x |
| b=16, len=16384+15x1024, width 256 | 7.15 | 4.33 | 1.65x |

The gain grows with the padding ratio, which is exactly the cache-hit prefill
case the issue describes: with `max_num_batched_tokens`-sized tables and short
cached prefixes, the table width is dominated by the longest sequence while
most requests need a handful of blocks.

## Correctness

* Bit-exact equivalence vs the stock implementation across 9 boundary cases
  (single/multi block boundaries, exact block multiples, zero-length
  sequences, uneven batch, non-128 head sizes) and 30 randomized shape
  sweeps — 138k+ output rows compared exactly, on NPU (`torch.equal`).
* The trimmed path provably selects the same physical blocks: block ids in
  trimmed columns are never referenced by the validity mask, since
  `position < seq_len` fails for every row beyond `ceil(seq_len/block)`.

## Not in scope (per the issue)

* Fusing paged addressing + NZ→ND conversion + dequant into direct dense
  writes (issue scope 2) — kept as a follow-up to avoid touching rounding
  behavior and graph buffer lifetimes in the same change.
* Attention arithmetic, the native paged INT8 decode path, DSA/MXFP formats —
  untouched.

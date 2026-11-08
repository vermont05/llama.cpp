# PTQ1_0 ("Q1") on HIP/RDNA3 - implementation plan

Research report. No code was changed, built, or benchmarked while writing this.

- Canonical tree: `~/src/llama.cpp-prism-tune`
- Branch: `vermont-main`
- Researched against HEAD: `ccb2dc5`
- All line numbers are from that tree.

**Legend:** **[CONFIRMED]** = read directly from source. **[HYPOTHESIS]** = performance prediction, needs measurement. **[MEASURE]** = must be measured before/while implementing.

Artifacts:
- Q1 base: `~/models/swift-bonsai-2-27b/Swift-Bonsai-2-PTQ1_0.gguf`
- Q1 Katana MTP: `~/models/swift-bonsai-2-27b/Swift-Bonsai-2-27B-Katana-PTQ1_0-MTP-Q8_0.gguf`
- PQ2 baseline: `~/models/swift-bonsai-2-27b/Swift-Bonsai-2-PQ2_0.gguf`

---

## 0. The one fact that shapes everything

PTQ1_0 and PQ2_0 are the *same ternary weights* at two packings:
- **PQ2_0**: 2 bits/element, positional, 4 elems/byte -> `block_pq2_0` = `d`(2B) + `qs[32]` = **34 B/block** (2.125 bpw). [CONFIRMED: `ggml-common.h:202-207`]
- **PTQ1_0**: base-3 trits, 5 trits/byte in `qs[24]` + 4 trits/byte in `qh[2]` -> `block_ptq1_0` = `d`(2B) + `qs[24]` + `qh[2]` = **28 B/block** (1.75 bpw). [CONFIRMED: `ggml-common.h:214-220`]

Both expand to the **same q8_0 tile layout** and run the **same dp4a compute**. The *only* PTQ1-specific code in the whole backend is (a) the scalar->vector base-3 decode in the decode vec-dot, and (b) the base-3 tile loader for prefill. That is why the port is small and low-risk: **the compute is shared; only the unpack differs.** The base-3 packing is the one thing that *cannot* be reused from PQ2 - PQ2's 2-bit nibbles map to `v_perm`/dp4a directly, PTQ1's 5-trits-per-byte straddles chunk boundaries and needs the multiply-by-3 carry recurrence.

---

## 1. DECODE (matvec)

### 1.1 Exact call path (all [CONFIRMED])

```
ggml_cuda_op_mul_mat (ggml-cuda.cu)
  - type GGML_TYPE_PTQ1_0 is in the offload whitelist (ggml-cuda.cu:5651)  -> runs on GPU
  - mul_mat_vec_q dispatch (mmvq.cu)
       + get_vec_dot_q_cuda(PTQ1_0) -> vec_dot_ptq1_0_q8_1          (mmvq.cu:16)
       + VDR = VDR_PTQ1_0_Q8_1_MMVQ = 4  -> whole 128-block/call     (vecdotq.cuh:116)
       + calc_nwarps(PTQ1_0, ncols=1, RDNA3_0)  -> NOT whitelisted -> 1 warp   (mmvq.cu:473-495)
       - calc_rows_per_block(1, RDNA3_0, ..., PTQ1_0) -> generic -> 1 row       (mmvq.cu:551-573)
```

The vec-dot itself (`vecdotq.cuh:895-953`) forks on platform:
- **HIP (today):** scalar loop, `vecdotq.cuh:899-947` - one thread walks all 128 trits, one base-3 decode + one `sumi += q * qs[...]` per trit. **No dp4a.** This is the 12.5 t/s path. [CONFIRMED]
- **CUDA (today):** calls `vec_dot_ptq1_0_q8_1_multi<1>` (`vecdotq.cuh:809-891`), which is **vectorized**: per 4-byte group it widens to 16-bit lanes, runs the `x3` base-3 carry (`& 0x00FF00FF`), emits a uint32 of four signed trit-bytes via `__vsub4(__byte_perm(...), 0x01010101)`, and accumulates with `ggml_cuda_dp4a`. **This function is wrapped in `#if !defined(GGML_USE_HIP)`** (`vecdotq.cuh:807`), so it never compiles on HIP. [CONFIRMED]

The fused gate/up (ncols_dst 2-3) special case that calls `vec_dot_ptq1_0_q8_1_multi<ncols_dst>` is *also* `#if !defined(GGML_USE_HIP)`-gated (`mmvq.cu:750-774`). [CONFIRMED]

### 1.2 What can be reused from PQ2 vs. what cannot

| Piece | PQ2_0 | PTQ1_0 reuse? |
|---|---|---|
| dp4a accumulator | `ggml_cuda_dp4a` | **Yes, identical.** On RDNA3 it is `__builtin_amdgcn_sudot4` (`common.cuh:705-711`). [CONFIRMED] |
| q8_1 activation load / ds scale | shared in `mul_mat_vec_q` | **Yes.** |
| Geometry (2 warps / 4 rows) | `PRISM_RDNA3_PQ2_0_NWARPS=2`, `_ROWS=4` (`mmvq.cu:407-416`) | **Yes - copy the pattern** with new `PRISM_RDNA3_PTQ1_0_*` constants. [CONFIRMED mechanism] |
| Stacking / fusions | type-generic in `ggml_cuda_try_fuse` | **Yes** (stacking/fusions are type-generic). |
| Per-32-chunk 2-bit symbol codec | `q2_0_symbols4_hip` + `v_perm` (`vecdotq.cuh:975-983`) | **No.** PTQ1 is base-3; needs the `x3`-carry decode. This is the irreducible difference. [CONFIRMED] |

**Structural VDR difference (important):** PQ2 uses `VDR=1` (4 lanes each take one 32-element chunk); PTQ1 uses `VDR=4` (each lane takes the whole 128 block) because base-3 trits are not independently chunkable. Keep `VDR_PTQ1_0_Q8_1_MMVQ=4`. [CONFIRMED]

### 1.3 Intrinsic availability on RDNA3/HIP [CONFIRMED unless noted]
- `ggml_cuda_dp4a` -> `__builtin_amdgcn_sudot4` (RDNA3 branch, `common.cuh:710`).
- `__vsub4` -> available (already used in HIP-compiled loaders, e.g. `mmq-load-tiles.cuh:1322`).
- `get_int_b4` -> `vecdotq.cuh:27`.
- `__byte_perm` -> **compiles on HIP but is a software routine** (`vecdotq.cuh:684`: *"HIP implements `__byte_perm` as a software routine"*). The hardware `v_perm_b32` is `__builtin_amdgcn_perm`, already used by `q2_0_symbols4_hip` (`vecdotq.cuh:700-704`) and `q1_0_unpack4_hip` (`:689-694`). Both are semantically identical byte-selects, so `__builtin_amdgcn_perm(a,b,s)` is a drop-in for `__byte_perm(a,b,s)`.

### 1.4 Kernel designs

**PRIMARY (recommended):** make the existing CUDA `_multi` kernel compile and run on HIP, but swap the software `__byte_perm` for the hardware `__builtin_amdgcn_perm`.
- Remove the `#if !defined(GGML_USE_HIP)` at `vecdotq.cuh:807` so `vec_dot_ptq1_0_q8_1_multi` builds on HIP.
- Introduce a local `perm32(a,b,s)` = `__byte_perm` (CUDA) / `__builtin_amdgcn_perm` (HIP), and use it for the 6 permutes in `_multi` (lines 822,823,832,845,846,855,873). Keep `__vsub4` and `ggml_cuda_dp4a`.
- Change `vec_dot_ptq1_0_q8_1` (`vecdotq.cuh:895`) to always call `_multi<1>` (delete the scalar `#if defined(GGML_USE_HIP)` branch, or keep it behind a kill switch - see 1.5).
- Un-gate the fused-gate special case at `mmvq.cu:750`.
- Add PTQ1 to RDNA3 geometry: in `calc_nwarps` (`mmvq.cu:477-492`) add `case GGML_TYPE_PTQ1_0: return PRISM_RDNA3_PTQ1_0_NWARPS;` and in `calc_rows_per_block` (`mmvq.cu:553`) add the PTQ1 branch; define `PRISM_RDNA3_PTQ1_0_NWARPS 2` / `_ROWS 4` (mirror `:407-416`).

Pseudocode for the decode inner loop (per 4-byte group), hardware-perm variant:
```c
uint32_t packed = get_int_b4(bq->qs, g);
uint32_t v_lo = perm32(packed, 0, 0x4140);   // bytes {0,1} -> lanes
uint32_t v_hi = perm32(packed, 0, 0x4342);   // bytes {2,3} -> lanes
for (t = 0; t < 5; ++t) {                     // 5 trits per byte
    uint32_t w_lo = v_lo*3, w_hi = v_hi*3;
    v_lo = w_lo & 0x00FF00FF;  v_hi = w_hi & 0x00FF00FF;   // base-3 carry, no cross-byte
    int q = __vsub4(perm32(w_lo, w_hi, 0x7531), 0x01010101); // 4 signed trits {-1,0,1}
    int e = t*16 + 4*g;
    for (j < ncols_dst)
        sumi[j][e>>5] = ggml_cuda_dp4a(q, get_int_b4(bq8_1[j*stride+ iqs+(e>>5)].qs,(e&31)>>2), sumi[j][e>>5]);
}
// ... second 8-byte group (qs+16) and the qh tail (8 trits) exactly as in :843-879
// scale: result[j] = (float)bq->d * sum_k ds[k]*sumi[j][k]
```
Work distribution: 2 warps x 4 rows/block, `VDR=4`, one 128-block per (lane, kbx) iteration. Reduction: per-warp partials -> `tmp_shared` -> warp-0 sums (existing `mul_mat_vec_q` reduce, `mmvq.cu:794-830`). Register/LDS risk: `sumi[ncols_dst<=3][4]` + decode temporaries + `tmp/tmp_gate[3][4]`; comparable to PQ2 which already runs at 2w/4r. **[MEASURE] VGPR count in the generated ISA** - 4 warps caused spills in prior decode work (25.0 vs 27.0 t/s), so 2 warps is the expected sweet spot; do not raise it blindly.

**FALLBACK:** just delete the `#if !defined(GGML_USE_HIP)` guard and let `__byte_perm` compile as the software routine (semantically identical bytes per `vecdotq.cuh:684`). Functionally correct and bit-identical to CUDA, but slower. Use as the A/B baseline and as the kill-switch target.

### 1.5 A/B kill switches
- **Prefill** already has one: `GGML_CUDA_PTQ1_0_MMQ_MAX_BATCH` (`mmq.cu:431`), currently CUDA-only. Setting it to `0` forces the fp16-dequant+rocBLAS fallback. Un-gate it for HIP. [CONFIRMED mechanism]
- **Decode:** add `GGML_CUDA_NO_PTQ1_0_VEC=1` (consistent with the existing `GGML_CUDA_NO_*` family) to force the scalar `vec_dot_ptq1_0_q8_1` path. This is the safe rollback for the decode change.

---

## 2. PREFILL (MMQ)

### 2.1 Complete map of the CUDA-only PTQ1 MMQ support [all CONFIRMED]

PTQ1's MMQ route is **structurally identical to PQ2_0** except the tile loader:

| Role | PQ2_0 | PTQ1_0 | Shared? |
|---|---|---|---|
| ds layout | `MMQ_Q8_1_DS_LAYOUT_D4` (`mmq.cuh:66`) | `:68` (gated) | same value |
| tile x sizes | `MMQ_DP4A_TXS_Q8_0` (`:405`) | `:408` (gated) | same |
| compute | `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` (`:577`) | `:582` (gated) | **same fn** |
| writeback | `ggml_cuda_mmq_write_back_dp4a` | `:583` (gated) | **same fn** |
| VDR | `VDR_PQ2_0_Q8_1_MMQ=2` | `VDR_PTQ1_0_Q8_1_MMQ=2` (`vecdotq.cuh:118`) | same |
| **tile loader** | `ggml_cuda_mmq_load_tiles_pq2_0` | `ggml_cuda_mmq_load_tiles_ptq1_0` | **PTQ1-specific** |

**Every `#if !defined(GGML_USE_HIP)` gate to remove:**
- `mmq.cu:20` - dispatch `mul_mat_q_case<PTQ1_0>`
- `mmq.cu:374` - `should_use_mmq` type case
- `mmq.cu:425` - the `max_batch`/env-switch block (make it return `ne11 <= max_batch` on HIP too)
- `mmq.cuh:67` - ds-layout case
- `mmq.cuh:406` - tile-size case
- `mmq.cuh:579` - **dp4a util-funcs (this is the RDNA3 path - the required one)**
- `mmq.cuh:755` - MMA util-funcs (CDNA-only; **keep gated** - see hazard H3)
- `mmq.cuh:1679` - `extern DECL_MMQ_CASE(PTQ1_0)`
- `mmq-load-tiles.cuh:271` - the loader + `ggml_cuda_mmq_decode_ptq1_0_qs4`
- `template-instances/mmq-instance-ptq1_0.cu:5` - the instantiation (compare: `mmq-instance-pq2_0.cu` has no guard)

**Config - the missing piece [CONFIRMED gap]:** RDNA3 selects `ggml_cuda_mmq_get_config_rdna3` (`mmq.cuh:247,271`). `mmq-config-rdna3.cuh` has 12 `Q1_0` + 12 `Q2_0` + 10 `PQ2_0` `CASE` entries (lines 2-38) but **zero `PTQ1_0` entries** (only `mmq-config-ampere.cuh:53-63` has them). So the port **must add 10 `PTQ1_0` `CASE` rows to `mmq-config-rdna3.cuh`, copied verbatim from the `PQ2_0` rows** with the type swapped (identical `GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0`, `MMQ_ITER_K`, nthreads=128/256, occupancy=2, I/J). Without this, `ggml_cuda_mmq_get_config_rdna3(PTQ1_0,...)` returns an empty/invalid config and dispatch falls back.

### 2.2 Minimum coherent HIP port (not just "enable the dispatch")

1. **Port the loader** (`mmq-load-tiles.cuh:271-369`): drop the `#if !defined(GGML_USE_HIP)` wrapper. The loader's decode helper `ggml_cuda_mmq_decode_ptq1_0_qs4` uses only `__byte_perm`/`__vsub4`/`get_int_b4` (all available on HIP, 1.3). It writes the base-3 trits as signed int8 (4/word) into the q8_0 SRAM tile and **broadcasts the single PTQ1 scale `d` into the 4 q8_0 sub-scale slots** (`scale_entries_per_block = QK_PTQ1_0/QK8_1 = 4`, `mmq-load-tiles.cuh:348`). Then the *shared* `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` does all compute. Optionally apply the same `__byte_perm`->`__builtin_amdgcn_perm` swap as decode for speed.
2. **Un-gate** the six `mmq.cuh`/`mmq.cu`/instance locations in 2.1 (all except `mmq.cuh:755`).
3. **Add the `PTQ1_0` `CASE` rows** to `mmq-config-rdna3.cuh`.

That is the whole coherent unit. There is no separate compute to write.

### 2.3 Building blocks reusable from the PQ2 HIP MMQ loader
`ggml_cuda_mmq_load_tiles_pq2_0` (`mmq-load-tiles.cuh:186-269`) and PTQ1's loader share: the `warp_size/nwarps/I/sram_stride` prologue, the `AMD_MFMA/TURING/WMMA` vs `#else`(dp4a) tile-pointer selection, the `blocks_per_iter`/`threads_per_row`/`nrows` tiling, the `fallback`/`i_max` tail clamp, and the separate scale-load loop. **Reuse all of that; only the per-lane decode body differs** (PQ2: `q2_0_symbols4_hip` on 2-bit bytes; PTQ1: the `x3`-carry `ggml_cuda_mmq_decode_ptq1_0_qs4` on 5-trit bytes + the `qh` tail).

### 2.4 Correctness hazards
- **H1 - packed trit decode.** The `x3` carry (`v_lo = w_lo & 0x00FF00FF`) must yield exactly the CPU `dequantize_row_ptq1_0` trits (`ggml-quants.c:2255`). `__byte_perm`/`__builtin_amdgcn_perm` are the same byte-select, so the decode is platform-stable, but **verify against the exact CPU PTQ1 oracle** (dequant to f32, f32 matmul) at Bonsai dims + tails. [MEASURE]
- **H2 - scale semantics.** Single `d` broadcast to 4 sub-scales is only correct because the dp4a compute applies 4 sub-scales per 128-block (`MMQ_Q8_1_DS_LAYOUT_D4`). Confirmed by `mmq_get_q8_1_ds_layout` returning `D4` for PTQ1 (`mmq.cuh:68`). A regression here silently scales the result wrong.
- **H3 - layout/orientation (the real trap).** The PTQ1 loader selects its SRAM layout with `#if defined(TURING_MMA_AVAILABLE) ... #else (dp4a)` (`mmq-load-tiles.cuh:298-305,324-328,362-366`) - it has **no `AMD_MFMA_AVAILABLE` branch**. On RDNA3 (dp4a) the `#else` branch is taken and matches the shared dp4a compute - correct. On **CDNA (MFMA)** the loader would emit the dp4a layout while the MMA compute expects the MFMA layout -> mismatch. **Therefore keep `mmq.cuh:755` (the MMA util-funcs) gated for PTQ1**, and scope this port to RDNA/dp4a. (PQ2's loader, by contrast, has the full `AMD_MFMA||TURING||WMMA` guard at `:193`.)
- **H4 - tail handling.** `fallback`/`i_max` clamp (`mmq-load-tiles.cuh:319-321,357-359`) handles the partial last tile. Test non-multiple-of-tile `ne` (Bonsai has a 248k-vocab output head and odd hidden sizes).
- **H5 - 48 KiB shared-memory gate.** `should_use_mmq` returns false if `smpbo < 48*1024` (`mmq.cu:416-423`; `smpbo = sharedMemPerBlockOptin`, `ggml-cuda.cu:345/352`). **Already passes for PQ2_0 on this exact gfx1102 box** (PQ2 PP512 baseline in the handoff) and PTQ1 shares PQ2's config/SRAM footprint, so it is expected to pass. **[MEASURE] log the actual `sharedMemPerBlockOptin` value** to close it out.

---

## 3. MTP (Katana)

**Confirmed from source - no type-specific hole.** [CONFIRMED]
- The fix `6c5a46d` adds, in `llama_model_qwen35::graph_mtp` (`src/models/qwen35.cpp:634-644`): after `tok_embd = ggml_get_rows(ctx0, tok_embd_w, inp->tokens)` where `tok_embd_w = layer.nextn.embed_tokens ? layer.nextn.embed_tokens : model.tok_embd`, it does the `hadamard_inverses->find(tok_embd_w)` lookup and applies `llama_mul_mat_hadamard(ctx0, tok_embd, it->second.rot)` + optional `ggml_mul(ctx0, tok_embd, it->second.signs)`.
- This is **the same restoration the normal decoder path performs** (`src/llama-graph.cpp:2384-2396`: `ggml_get_rows` -> `hadamard_inverses` lookup -> `llama_mul_mat_hadamard` + `ggml_mul`), just keyed on the MTP table tensor instead of the base table.
- `hadamard_inverses` is populated from GGUF `hadamard_inverse_blocks` metadata (`src/llama-model.cpp:1941-1954`), i.e. from the **model file, not the weight type**. Every op in the path is type-agnostic: `ggml_get_rows` supports PTQ1 (`ggml-cuda.cu:5692`), and `llama_mul_mat_hadamard`/`ggml_mul` operate on the dequantized lookup output. **So PTQ1 base + Katana MTP runs the identical repaired Qwen35 route as PQ2 - nothing in the source branches on PTQ1 vs PQ2.**
- The one thing that is *file-dependent* (not type-dependent): the MTP `nextn.embed_tokens` tensor must actually be registered in `hadamard_inverse_blocks` by the Katana GGUF for the lookup to fire. That is a property of the specific artifact, which is exactly what the smoke test below checks.

**Minimal runtime smoke test to run later (do NOT run now):** on canonical `vermont-main`, load `Swift-Bonsai-2-PTQ1_0.gguf` + `Swift-Bonsai-2-27B-Katana-PTQ1_0-MTP-Q8_0.gguf`, fixed prompt, temp 0, ~128 tokens, MTP drafting on, run twice. Assert: (a) MTP initializes and the `nextn.embed_tokens` Hadamard-inverse lookup fires (visible via the `hadamard_verified` graph check, `src/llama-context.cpp:2539`); (b) draft acceptance is non-trivial (anchor: **57/70 = 81.4%**, mean 1.81, on the reference binary); (c) output is **byte-identical across the two runs** (determinism); (d) effective t/s uplift over base-only (anchor: 12.80 -> 13.91). MTP does not affect prefill.

---

## 4. Dependency map

**Decode:** `ggml-common.h` (block) -> `common.cuh` (`ptq1_0_trit`, `ggml_cuda_dp4a`) -> `vecdotq.cuh` (`vec_dot_ptq1_0_q8_1`, `_multi`, `VDR`) -> `mmvq.cu` (`get_vec_dot_q_cuda`, `calc_nwarps`, `calc_rows_per_block`, K-loop gate) -> `ggml-cuda.cu` (offload whitelist, already open).

**Prefill:** `ggml-common.h` (block) -> `mmq-load-tiles.cuh` (loader+decode, **the only new logic**) -> `mmq.cuh` (ds-layout, tile-sizes, dp4a util-funcs, extern decl) + `mmq.cu` (dispatch, `should_use_mmq`, max-batch) + `mmq-config-rdna3.cuh` (**add PTQ1 rows**) + `template-instances/mmq-instance-ptq1_0.cu` (instantiate) -> shared `ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a` compute.

---

## 5. Recommended implementation order

1. **Decode first** (small, isolated, big win, trivially A/B-able): port `_multi` to HIP with the `__builtin_amdgcn_perm` swap, wire `vec_dot_ptq1_0_q8_1` to it, add `PRISM_RDNA3_PTQ1_0_*` geometry, add `GGML_CUDA_NO_PTQ1_0_VEC` kill switch. Verify vs scalar path (bit-identical) + tg64 d0/d4096/d16384.
2. **Prefill second** (bigger surface): port loader, un-gate the six sites, add the `mmq-config-rdna3.cuh` PTQ1 rows, un-gate the instance file. Verify vs the CPU PTQ1 oracle + `test-backend-ops` MUL_MAT at Bonsai dims/tails.
3. **MTP last** (no code change expected - it is type-agnostic): just run the section 3 smoke test on the canonical branch to confirm Q1+MTP.

Decode and prefill are independent and can land as separate commits on `vermont-main`.

---

## 6. Risks & verification plan

| Risk | Mitigation / verification |
|---|---|
| Decode bit-drift (base-3 decode, FMA contraction) | Bit-identical to the scalar path (same formula); repeat the `ppl.txt -c 256` perplexity check and `GGML_CUDA_SUMS` per-node checksums. The recurring trap is compiler FMA contraction differing between fused and original - check the ISA `v_fmac`/`v_fmamk` counts and use per-chunk perplexity. |
| VGPR spill at 2w/4r | [MEASURE] ISA VGPR count; if it spills, drop to 2 rows before touching warps. Do **not** try 4 warps (measured slower in prior decode work). |
| Prefill decode mismatch (H1/H2) | Exact CPU PTQ1 oracle + `test-backend-ops` MUL_MAT at 248k-vocab head and odd hidden sizes; deterministic prompt compare. |
| CDNA layout mismatch (H3) | Keep `mmq.cuh:755` gated; scope to RDNA/dp4a. |
| 48 KiB gate (H5) | [MEASURE] log `sharedMemPerBlockOptin`; anchor is PQ2 already passing on this box. |
| Long-prompt over-claim | Do **not** infer long-context wins from PP512 alone. Report interleaved ABBA PP512, pp4096, and 12k+ on a quiet CPU (check `pgrep -af "cmake --build"` + load average first). |

**Expected result (HYPOTHESIS, not measured):** decode moves off the 12.5 t/s scalar path toward the PQ2 ~27 t/s band because the vectorized dp4a path keeps decode memory-bound (estimate: matvec ~21.3 ms at 265 GB/s, ~10 ALU ops/weight-byte that overlap the load). Prefill MMQ should beat the fp16-dequant+rocBLAS fallback, most at the 248k-vocab output head (avoids the ~2.5 GB fp16 copy). **Both must be confirmed by the ABBA benchmarks above before any claim.**

---

## 7. Things to measure before/at implementation (explicit)
1. `sharedMemPerBlockOptin` on this gfx1102 (closes H5).
2. VGPR count of the new decode kernel at 2 warps/4 rows (spill check).
3. Hardware `__builtin_amdgcn_perm` vs software `__byte_perm` delta in the decode (decides primary vs fallback as the shipped default).
4. Decode: tg64 d0/d4096/d16384, ABBA vs current, quiet CPU.
5. Prefill: PP512, pp4096, 12k+ prompt, ABBA; plus `test-backend-ops` MUL_MAT and the CPU-PTQ1-oracle comparison.
6. MTP: the section 3 smoke test (acceptance rate, determinism, t/s uplift) on canonical `vermont-main`.
7. PQ2 regression guard: PQ2 decode tg64 must be unchanged (the geometry edits are PTQ1-scoped, but confirm).

# FINAL.WORK: what is left on the Prism / Swift Bonsai 2 27B tuning

Written 2026-10-26, on top of `vermont-main` at `4b4c911`. The detailed history is in `~/git/PRISM.HANDOFF.md`.

## Where things stand

- **Served configuration:**
  - model: Katana PTQ1_0 trunk + Q4_0 MTP layer (`~/models/swift-bonsai-2-27b/Swift-Bonsai-2-27B-Katana-PTQ1_0-MTP-Q4_0.gguf`);
  - **MTP off**, ctx 90112, q4_0 KV, ub 512;
  - RX 7600M XT (gfx1102, 8 GB) on TheRock ROCm 10.1;
  - live build `bin-prod-20261026a` (= `91f3313`), launcher `~/bin/swift-bonsai-therock.sh`.
- **`4b4c911`** (the MTP conv fusion) is committed but not frozen and not live. It only matters with MTP on.
- **Numbers** (llama-bench unless noted):

| | now | realistic ceiling | % |
|---|---|---|---|
| decode, short ctx | 31.7 t/s (31.6 ms/token) | ~34 t/s | ~93 % |
| decode, 89k ctx | 24.9 t/s | ~26.5 t/s | ~90 % |
| pp512 d0 / 16k / 32k | 320.7 / 272.9 / 240.6 t/s | ~10–15 % more | ~85 % |
| MTP n1 (ctx 53k, server), not served | prose 34.9, code 35.6, after 30k prompt 28.2 | see item 4 | — |

- **Short-context decode breakdown:**
  - matvec 23.2 ms at ~260 GB/s; card peak ~288 GB/s, big matvecs reach 257–265;
  - ~5 ms of small kernels;
  - ~3 ms of idle gaps.
  - The hard limit (all bytes at peak, no overhead) is ~47 t/s; not reachable.

**Rules that still apply:**
- Take the GPU lock for every GPU run: `exec 9<>/tmp/ninfer-gpu.lock; flock -w 14400 9 ... flock -u 9`.
- Stop the service before measuring.
- Leave `~/src/llama.cpp-prism-reference` and the `bin-prod-*` dirs alone.
- Commit only if the change is faster and correct.

**How a change is checked:**
- `~/src/prism-decode2-work/fuse/check.sh <bindir>` (BASE=<live bindir>): pp512 trace, ABBA bench, ppl, KLD at -ub 7.
- `~/src/prism-decode2-work/mtp-ab.sh`: server A/B with modes 0/1/2; the md5 includes reasoning_content.
- KLD noise floor: max 5.2e-5, RMS Δp 0.001 %.

---

## 1. Long-context decode attention (+1–2 t/s at 64k+)

**Why:**
- At 89k a token costs ~8.7 ms more than at depth 0.
- The KV bytes read per token are `depth x 18432` (16 attention layers, q4_0 K+V, 1024 values each), ~1.6 GB at 88k.
- That works out to ~185 GB/s effective, vs ~260 GB/s for the matvecs.
- At 260 GB/s it would be ~6.1 ms, so ~+6 % at 89k.

**Status:** measurement prompt written but not run: `~/git/PRISM.NEXT-PROMPT-DECODE-ATTN.md` (result goes to `~/git/PRISM-DECODE-ATTN-RESULT.md`). It gets:
- tg64 at depths 0 / 16k / 32k / 64k / 86k;
- a per-depth kernel trace with attention and combine time, the grid (parallel blocks) and GB/s;
- counters at 64k (memory- vs instruction-bound);
- the split formula from the source.

**Code:**
- Kernel: `flash_attn_ext_decode_q4_0`, `ggml/src/ggml-cuda/fattn-decode-q4.cu:84`, launched at line ~484.
- The rescale loop is near line ~325.
- The split count comes from `launch_fattn` in `ggml/src/ggml-cuda/fattn-common.cuh`; parts are merged by `flash_attn_combine_results`.

**Plan:**
1. Run the prompt.
2. If the parallel-block count is too low at depth, add an override and sweep it.
3. If the counters show it is instruction-bound, skip the rescale when the running max does not change.

**Also:** with MTP, the 2-token verify uses the generic `flash_attn_ext_vec<256,2,...>`, not this kernel. That is why MTP loses after long prompts. A 2-column version of the decode kernel would fix it (only matters with MTP; see item 4).

## 2. Fuse the GDN layer chain (+~1.5 % decode, big job)

**Why:**
- A GDN (Gated DeltaNet) layer at decode is 9 kernels: norm+fwht, fused qkv/gate matvec, conv step+alpha/beta, gated_delta_net, gated norm+fwht, ssm_out, norm+fwht, gate/up, fwht+down.
- Fusing **conv step + gated_delta_net + gated norm** into one kernel saves ~10 µs × 48 layers ≈ 0.5 ms/token.
- It also cuts a share of the ~3 ms of launch gaps.

**Code:**
- `ggml/src/ggml-cuda/ssm-conv-step.cu` (`ggml_cuda_try_fuse_conv_step`, line ~182).
- `ggml/src/ggml-cuda/gated_delta_net.cu` (kernel at line ~15; already ~bandwidth-bound, 48 × ~23 µs).
- `ggml/src/ggml-cuda/gated-norm.cu` (`ggml_cuda_try_fuse_gated_norm`, line ~45).
- Matchers are called from `ggml_cuda_try_fuse` in `ggml-cuda.cu` (~line 4352).

**Watch:**
- The graph allocator reuses dead inputs' buffers for outputs (as an exact alias or at a row offset). Check overlaps before launching.
- Every fusion needs the GDN state writes to stay exact.
- MTP rollback writes n_rs_seq+1 state snapshots (see `4b4c911` for the conv side).

**Tool:** `~/src/prism-decode2-work/kprof.sh` for the per-token kernel profile.

## 3. Rewrite the prompt matmul (+10–15 % prefill, most work, least certain)

**Why:**
- The PQ2_0 / PTQ1_0 prompt matmul (`mul_mat_q<142,128,false>`) is ~75 % of pp512 time.
- It runs at ~20 T int8 ops/s, 57 % of the measured WMMA peak (35–37 T, ~62 cycles per WMMA).
- It is VALU-bound: WMMA is ~50 % of the time; the unpack is already free (v_perm byte lookup); 223 VGPRs, no spills.

**Tried and lost:** epilogue restructures (they spill) and ubatch 1024 (+1.6 % but only 17 MiB free at 90k).

**What could work:** a new tile structure that issues WMMA back to back, with the scale/accumulate epilogue moved out of the inner loop. For example, accumulate int32 over several K blocks before converting to float, if the quant format's per-block scales allow it.

**Probes and tools:** `~/src/prism-decode2-work/mmq-probe/` (mm-perf.sh, mkprobe.py, build-probe.sh, regs.sh, the `wm/wm.cpp` WMMA microbenchmark). Notes are in `~/git/PRISM.HANDOFF.md` under "Prompt matmul (PQ2_0 mmq) probes".

## 4. Outside kernel tuning: MTP, quantization, hardware

**MTP status (`4b4c911`, ctx 53248, n_max 1):**
- vs no MTP: prose +10 %, code +12 %, after a 30k prompt −3 %.
- It costs ~800–900 MiB of VRAM, i.e. ctx 90k → 53k, so it is off. n_max 2 is worse everywhere.

**Where MTP loses time (trace `~/src/prism-decode2-work/mtp-ab/kt-mtp1`):**
- A verify step is ~50 ms wall for ~1.7 tokens, but the GPU is busy only ~40 ms of it. **~30 % is idle**: host-side sequencing of draft → sample → verify in `common/speculative.cpp` and `tools/server/server-context.cpp`.
  - Overlapping these, or keeping draft sampling on the GPU, could make MTP n1 ~+25 % or more over no MTP.
- The 2-token attention uses the generic `flash_attn_ext_vec` (item 1).
- The output head is read twice per step (draft + verify), ~2.2 ms.
- n_max 2 (3-token verify) gets no HIP graphs under `GGML_CUDA_GRAPHS_DECODE_ONLY=2`.

**The VRAM question decides whether MTP can ever be served at 90k:**
- Break down the ~800 MiB: MTP layer weights, draft KV, draft compute buffers, rs-cache snapshots (~150 MB per extra GDN state copy).
- Then see what can shrink.
- `test-recurrent-state-rollback` cannot run on 8 GB (it hard-codes n_rs_seq=8, 1.35 GB rs cache × 2 contexts).

**Beyond that:** a smaller quantization of the trunk (less to read per token) or a card with more VRAM. Neither is tuning work.

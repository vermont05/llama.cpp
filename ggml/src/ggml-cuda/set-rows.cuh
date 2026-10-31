#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Two single-row SET_ROWS into Q4_0 as one launch; false if not applicable.
bool ggml_cuda_op_set_rows_q4_0_pair(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);

// KV cache quantization emulation for quality tests (PRISM_KV_EMU_K / PRISM_KV_EMU_V = q4_0 | turbo3): the rows written
// into an f16 cache_k / cache_v are first round-tripped through that quantization (turbo3: the TurboQuant 3-bit
// PolarQuant of TheTom/llama-cpp-turboquant, WHT rotation of 128-value groups, Lloyd-Max centroids, corrected norm,
// rotated back), so the attention sees what the real type would give. Slow, quality tests only.
bool ggml_cuda_kv_emu_active();

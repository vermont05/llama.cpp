#include "common.cuh"

#define CUDA_ROPE_BLOCK_SIZE 256

void ggml_cuda_op_rope(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

void ggml_cuda_op_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * set_rows);

// fwht_mm: optional FWHT-hint MUL_MAT (N = 256) applied to the roped rows; its output is written instead of the rope's.
void ggml_cuda_op_rms_norm_mul_rope_fused(ggml_backend_cuda_context & ctx, ggml_tensor * rms_norm, ggml_tensor * mul, ggml_tensor * rope, ggml_tensor * set_rows, ggml_tensor * fwht_mm = nullptr);

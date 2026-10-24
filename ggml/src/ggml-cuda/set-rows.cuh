#pragma once

#include "common.cuh"

#define CUDA_SET_ROWS_BLOCK_SIZE 256

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// Two single-row SET_ROWS into Q4_0 as one launch; false if not applicable.
bool ggml_cuda_op_set_rows_q4_0_pair(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b);

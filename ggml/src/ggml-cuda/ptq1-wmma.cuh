#pragma once

#include "common.cuh"

// PTQ1_0 x q8_1 for 2..8 columns on RDNA3 with the int8 WMMA: each weight is decoded once and the dot products of all
// columns run on the matrix unit, so a column costs little more than the decode (the vector kernel pays a VALU dot
// product per column; gfx1102 has no dp4a). Returns false if it does not apply (device, shape or GGML_CUDA_NO_PTQ1_0_WMMA).
bool ggml_cuda_mul_mat_ptq1_0_wmma(
        ggml_backend_cuda_context & ctx, const void * vx, const void * vy, float * dst,
        int ncols_x, int nrows_x, int ncols_dst,
        int stride_row_x, int stride_col_y, int stride_col_dst, cudaStream_t stream);

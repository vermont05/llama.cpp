#pragma once

// TURBO3_0 KV cache (see ggml-common.h): the cache write (SET_ROWS) and the conversion to f16/f32, which undoes the
// rotation. Both work on one 128-value group per block of 128 threads; speed matters little here (a decode step
// writes 2*8 groups per layer, and the conversion only feeds the prompt kernels that need a f16 copy).

#include "common.cuh"
#include "convert.cuh"

#define TURBO3_THREADS QK_TURBO3

static __device__ __forceinline__ int turbo3_index(const block_turbo3_0 & b, const int j) {
    return ((b.qs[j/4] >> (2*(j%4))) & 3) | (((b.signs[j/8] >> (j%8)) & 1) << 2);
}

static __device__ __forceinline__ int turbo3_nearest(const float v) {
    int idx = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i) {
        idx += v >= turbo3_midpoints[i];
    }
    return idx;
}

// sum of v over the 128 threads of the block, red: 128 floats of shared memory
static __device__ __forceinline__ float turbo3_sum(const float v, float * red) {
    const int j = threadIdx.x;
    red[j] = v;
    __syncthreads();
#pragma unroll
    for (int h = QK_TURBO3/2; h > 0; h >>= 1) {
        if (j < h) {
            red[j] += red[j + h];
        }
        __syncthreads();
    }
    const float res = red[0];
    __syncthreads();
    return res;
}

// x = D(sb) * H/sqrt(128) * D(sa) * x for the 128 values in shared memory x (all threads of the block)
static __device__ __forceinline__ void turbo3_wht(float * x, const float * sa, const float * sb) {
    const int j = threadIdx.x;
    x[j] *= sa[j];
    __syncthreads();
#pragma unroll
    for (int h = 1; h < QK_TURBO3; h <<= 1) {
        if ((j & h) == 0) {
            const float a = x[j];
            const float b = x[j + h];
            x[j]     = a + b;
            x[j + h] = a - b;
        }
        __syncthreads();
    }
    x[j] *= 0.08838834764831845f * sb[j];
    __syncthreads();
}

// quantize the 128 values src[0..127] (thread j loads src[j]) into *dst, as quantize_row_turbo3_0_ref
static __device__ void turbo3_quantize_group(const float * __restrict__ src, block_turbo3_0 * __restrict__ dst) {
    __shared__ float   x[QK_TURBO3];
    __shared__ float   red[QK_TURBO3];
    __shared__ uint8_t idxs[QK_TURBO3];

    const int j = threadIdx.x;
    x[j] = src[j];
    __syncthreads();

    const float norm     = sqrtf(turbo3_sum(x[j]*x[j], red));
    const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
    x[j] *= inv_norm;
    __syncthreads();

    turbo3_wht(x, turbo3_wht_s1, turbo3_wht_s2);

    const int idx = turbo3_nearest(x[j]);
    idxs[j] = idx;
    const float c = turbo3_centroids[idx];
    const float recon_norm = sqrtf(turbo3_sum(c*c, red)); // also orders the idxs writes before the packing

    if (j < QK_TURBO3/4) {
        dst->qs[j] = (idxs[4*j + 0] & 3) | ((idxs[4*j + 1] & 3) << 2) | ((idxs[4*j + 2] & 3) << 4) | ((idxs[4*j + 3] & 3) << 6);
    }
    if (j < QK_TURBO3/8) {
        uint8_t s = 0;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            s |= ((idxs[8*j + k] >> 2) & 1) << k;
        }
        dst->signs[j] = s;
    }
    if (j == 0) {
        dst->norm = __float2half(recon_norm > 1e-10f ? norm/recon_norm : norm);
    }
}

// dequantize *src into dst[0..127] (thread j stores dst[j]), undoing the rotation, as dequantize_row_turbo3_0
template <typename dst_t>
static __device__ void turbo3_dequantize_group(const block_turbo3_0 * __restrict__ src, dst_t * __restrict__ dst) {
    __shared__ float x[QK_TURBO3];

    const int j = threadIdx.x;
    x[j] = turbo3_centroids[turbo3_index(*src, j)] * __half2float(src->norm);
    __syncthreads();

    turbo3_wht(x, turbo3_wht_s2, turbo3_wht_s1);

    dst[j] = ggml_cuda_cast<dst_t>(x[j]);
    __syncthreads();
}

// Rotate each 128-value group of the rows of x (f32, ne0 % 128 == 0, strides in bytes) by M (inverse = false) or
// M^T (inverse = true) into the contiguous f32 y; y may be x if x is contiguous. For attention kernels that work in
// the rotated domain: q.k = (Mq).(Mk), and the output of P (M V) is rotated back with M^T.
void ggml_cuda_turbo3_rotate_rows(const char * x, float * y, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, bool inverse, cudaStream_t stream);

#include "common.cuh"
#include "fwht.cuh"
#include "quantize.cuh"

#include <cstdlib>
#include <cstring>

template <typename T>
__device__ __forceinline__ float fwht_load(const T value) {
    return value;
}

template <>
__device__ __forceinline__ float fwht_load<half>(const half value) {
    return __half2float(value);
}

template <int N, typename T, bool has_signs>
__launch_bounds__(4*ggml_cuda_get_physical_warp_size(), 1)
__global__ void fwht_cuda(const T * src, float * dst, const int64_t n_rows, const float scale,
                          const float * signs, const int n_blk) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();

    const int64_t r = (int64_t) blockIdx.x * blockDim.y + threadIdx.y;

    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    static constexpr int el_w = N / warp_size;
    float     reg[el_w];
    const int lane = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;
#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        reg[i] = fwht_load(src[i * warp_size + lane]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * warp_size + lane];
        }
    }

#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; j++) {
            const float val  = reg[j];
            const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);

            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }

#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];

                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

#pragma unroll
    for (int i = 0; i < el_w; ++i) {
        dst[i * warp_size + lane] = reg[i];
    }
}

// Large-N path. The register kernel above keeps N/warp_size floats per thread, so it stops being
// viable well before the arithmetic does: N=4096 would need 128 registers per thread and spill,
// which is why the switch below used to end at 2048 and simply decline anything larger (the whole
// op then fell back to CPU). Stage the row in shared memory instead and run all log2(N) butterfly
// stages there. One row per block; every thread handles several butterflies per stage. Slower per
// row than the register path, so it is used only where that path cannot go.
#define FWHT_SMEM_THREADS 256

template <int N, typename T, bool has_signs>
__launch_bounds__(FWHT_SMEM_THREADS, 1)
__global__ void fwht_cuda_smem(const T * src, float * dst, const int64_t n_rows, const float scale,
                               const float * signs, const int n_blk) {
    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;
    dst += r * N;

    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    ggml_cuda_pdl_sync();
    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        float v = fwht_load(src[i]) * scale;
        if (has_signs) {
            v *= signs_row[i];
        }
        s[i] = v;
    }
    __syncthreads();

    // Same butterfly and the same sign convention as the register path: the low element of a pair
    // takes x + y, the high one x - y.
#pragma unroll 1
    for (int h = 1; h < N; h *= 2) {
        for (int idx = threadIdx.x; idx < N / 2; idx += FWHT_SMEM_THREADS) {
            const int j = ((idx / h) * 2 * h) + (idx % h);
            const float x = s[j];
            const float y = s[j + h];
            s[j]     = x + y;
            s[j + h] = x - y;
        }
        __syncthreads();
    }

    for (int i = threadIdx.x; i < N; i += FWHT_SMEM_THREADS) {
        dst[i] = s[i];
    }
}


// Wide rows at small row counts (decode): one row per block instead of per warp.
// The warp kernel serialises every stage on one warp, and the shared-memory kernel above synchronises on each stage.
// Both leave most of the GPU idle at these shapes.
#define FWHT_BLOCK_THREADS 256

// FWHT stages on one N-wide row held as NE = N/NT registers per thread, then store dst (and q8_1).
// xor lane exchange within a wave32: DPP row_xmask / permlanex16 on RDNA (no LDS round trip as with
// ds_bpermute), __shfl_xor_sync elsewhere. Same pairing either way, so the arithmetic is unchanged.
template <int h>
static __device__ __forceinline__ float fwht_xor(const float x) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    if constexpr (h == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), 0x160 | h, 0xF, 0xF, false));
    }
#else
    return __shfl_xor_sync(0xFFFFFFFF, x, h, WARP_SIZE);
#endif // defined(GGML_USE_HIP) && defined(RDNA)
}

// warp_reduce_sum / warp_reduce_max over 32 lanes with the same order (offset 16 first).
static __device__ __forceinline__ float fwht_warp_sum(float x) {
    x += fwht_xor<16>(x); x += fwht_xor<8>(x); x += fwht_xor<4>(x); x += fwht_xor<2>(x); x += fwht_xor<1>(x);
    return x;
}
static __device__ __forceinline__ float fwht_warp_max(float x) {
    x = fmaxf(x, fwht_xor<16>(x)); x = fmaxf(x, fwht_xor<8>(x)); x = fmaxf(x, fwht_xor<4>(x));
    x = fmaxf(x, fwht_xor<2>(x));  x = fmaxf(x, fwht_xor<1>(x));
    return x;
}

template <int h, int NE>
static __device__ __forceinline__ void fwht_lane_stage(float (&reg)[NE], const int lane) {
#pragma unroll
    for (int j = 0; j < NE; j++) {
        const float val  = reg[j];
        const float val2 = fwht_xor<h>(val);
        reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
    }
}

template <int N, int NT>
static __device__ __forceinline__ void fwht_block_finish(float (&reg)[N / NT], float * s, const int64_t r,
                                                         float * dst, block_q8_1 * q8) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;

    const int tid  = threadIdx.x;
    const int lane = tid % warp_size;

    // stages within a warp: partner differs in the lane bits
    if constexpr (warp_size == 32) {
        fwht_lane_stage<1>(reg, lane);
        fwht_lane_stage<2>(reg, lane);
        fwht_lane_stage<4>(reg, lane);
        fwht_lane_stage<8>(reg, lane);
        fwht_lane_stage<16>(reg, lane);
    } else {
#pragma unroll
        for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
            for (int j = 0; j < NE; j++) {
                const float val  = reg[j];
                const float val2 = __shfl_xor_sync(0xFFFFFFFF, val, h, warp_size);
                reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
            }
        }
    }

    // stages across warps: partner differs in the thread-index bits above the lane
#pragma unroll
    for (int h = warp_size; h < NT; h *= 2) {
#pragma unroll
        for (int j = 0; j < NE; j++) {
            s[j * NT + tid] = reg[j];
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < NE; j++) {
            const float val  = reg[j];
            const float val2 = s[j * NT + (tid ^ h)];
            reg[j] = (tid & h) == 0 ? val + val2 : val2 - val;
        }
        __syncthreads();
    }

    // stages above the block width: partner is another register of the same thread
#pragma unroll
    for (int h = NT; h < N; h *= 2) {
        const int step = h / NT;
#pragma unroll
        for (int j = 0; j < NE; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; k++) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k]        = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }

    dst += r * N;
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        dst[i * NT + tid] = reg[i];
    }

    // same math as quantize_q8_1: a warp holds 32 consecutive values, one q8_1 block
    if (q8 != nullptr) {
#pragma unroll
        for (int i = 0; i < NE; ++i) {
            const float xi   = reg[i];
            const float amax = warp_size == 32 ? fwht_warp_max(fabsf(xi)) : warp_reduce_max<QK8_1>(fabsf(xi));
            const float sum  = warp_size == 32 ? fwht_warp_sum(xi)        : warp_reduce_sum<QK8_1>(xi);
            const float  d = amax / 127.0f;
            const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

            const int64_t idx = r * N + i * NT + tid;
            block_q8_1 & b = q8[idx / QK8_1];
            b.qs[idx % QK8_1] = q;
            if (idx % QK8_1 == 0) {
                b.ds = make_half2(d, sum);
            }
        }
    }
}

template <int N, int NT, typename T, bool has_signs>
__launch_bounds__(NT, 1)
__global__ void fwht_cuda_block(const T * src, float * dst, const int64_t n_rows, const float scale,
                                const float * signs, const int n_blk, block_q8_1 * q8) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int NE        = N / NT;
    static_assert(NE >= 1 && N % NT == 0 && NT % warp_size == 0, "bad FWHT block shape");

    __shared__ float s[N];

    const int64_t r = blockIdx.x;
    if (r >= n_rows) {
        return;
    }

    src += r * N;

    const int tid = threadIdx.x;

    ggml_cuda_pdl_sync();
    const float * signs_row = has_signs ? signs + (r % n_blk) * N : nullptr;

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        reg[i] = fwht_load(src[i * NT + tid]) * scale;
        if (has_signs) {
            reg[i] *= signs_row[i * NT + tid];
        }
    }

    fwht_block_finish<N, NT>(reg, s, r, dst, q8);
}

// rms_norm * weight -> norm_dst, then sign flip + FWHT of it, one block per N-wide chunk of a row.
// Each block reads the full row for the norm sum and writes only its own chunk.
template <int N, int NT>
__launch_bounds__(NT, 1)
__global__ void fwht_rms_norm_block(const float * x, const float * w, float * norm_dst, float * dst,
                                    const int ncols, const float eps, const float scale,
                                    const float * signs, const int n_blk, block_q8_1 * q8) {
    constexpr int NE = N / NT;

    __shared__ float s[N];
    __shared__ float s_sum[32];

    const int64_t r     = blockIdx.x;
    const int     chunk = r % (ncols / N);
    const int64_t row   = r / (ncols / N);
    const int     tid   = threadIdx.x;

    x        += row * ncols;
    norm_dst += row * ncols;

    ggml_cuda_pdl_sync();
    // This chunk's inputs do not depend on the norm: load them first so their latency overlaps the reduction.
    const float * signs_row = signs + (r % n_blk) * N;
    float xr[NE], wr[NE], sr[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int col = chunk * N + i * NT + tid;
        xr[i] = x[col];
        wr[i] = w[col];
        sr[i] = signs_row[i * NT + tid];
    }
    // Sum in the same order as rms_norm_f32, which runs vt = 256 or 1024 threads: thread tid acts as the
    // virtual threads tid + k*NT, and each warp of them gives the same warp sum as there.
    const int vt = ncols < 1024 ? 256 : 1024;
    for (int k = 0; k < vt / NT; ++k) {
        const int v = tid + k * NT;
        float tmp = 0.0f;
        for (int col = v; col < ncols; col += vt) {
            const float xi = x[col];
            tmp += xi * xi;
        }
        tmp = WARP_SIZE == 32 ? fwht_warp_sum(tmp) : warp_reduce_sum(tmp);
        if (v % WARP_SIZE == 0) {
            s_sum[v / WARP_SIZE] = tmp;
        }
    }
    __syncthreads();
    const int   lane   = tid % WARP_SIZE;
    const float sum_in = lane < vt / WARP_SIZE ? s_sum[lane] : 0.0f;
    const float sum    = WARP_SIZE == 32 ? fwht_warp_sum(sum_in) : warp_reduce_sum(sum_in);
    const float rscale = rsqrtf(sum / ncols + eps);

    float reg[NE];
#pragma unroll
    for (int i = 0; i < NE; ++i) {
        const int   col = chunk * N + i * NT + tid;
        const float y   = rscale * xr[i] * wr[i];
        norm_dst[col] = y;
        reg[i] = y * scale * sr[i];
    }

    fwht_block_finish<N, NT>(reg, s, r, dst, q8);
}

bool ggml_cuda_op_rms_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                                const ggml_tensor * signs, ggml_tensor * dst) {
    static const bool disabled = getenv("GGML_CUDA_NO_RMS_FWHT") != nullptr || getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;
    if (disabled) {
        return false;
    }
    const ggml_tensor * x = rms_norm->src[0];
    const ggml_tensor * w = mul->src[0] == rms_norm ? mul->src[1] : mul->src[0];
    const int     n     = dst->ne[0];
    const int     ncols = x->ne[0];
    const int64_t rows  = ggml_nrows(x);

    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || signs->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32 ||
        !ggml_is_contiguous(x) || !ggml_is_contiguous(w) || !ggml_is_contiguous(signs) ||
        !ggml_is_contiguous(mul) || !ggml_is_contiguous(dst) ||
        ggml_nrows(w) != 1 || w->ne[0] != ncols || signs->ne[0] != ncols || ggml_nrows(signs) != 1 ||
        ncols % n != 0 || ggml_nelements(dst) != rows * ncols || ggml_nelements(mul) != rows * ncols) {
        return false;
    }

    float eps = 0.0f;
    memcpy(&eps, rms_norm->op_params, sizeof(float));
    const float scale = 1 / sqrtf(n);
    const int   n_blk = ncols / n;

    const dim3 g((unsigned) (rows * n_blk), 1, 1), b(FWHT_BLOCK_THREADS, 1, 1);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, ctx.stream());

    const float * x_d = (const float *) x->data;
    const float * w_d = (const float *) w->data;
    const float * s_d = (const float *) signs->data;
    float * norm_d = (float *) mul->data;
    float * dst_d  = (float *) dst->data;

    switch (n) {
#define RMS_FWHT_CASE(NN) \
        case NN: { \
            block_q8_1 * q8 = ggml_cuda_q8_1_cache_put(ctx, dst); \
            ggml_cuda_kernel_launch(fwht_rms_norm_block<NN, FWHT_BLOCK_THREADS>, lp, x_d, w_d, norm_d, dst_d, ncols, eps, scale, s_d, n_blk, q8); \
            return true; \
        }
        RMS_FWHT_CASE(512)
        RMS_FWHT_CASE(1024)
        RMS_FWHT_CASE(2048)
        RMS_FWHT_CASE(4096)
        RMS_FWHT_CASE(8192)
#undef RMS_FWHT_CASE
        default:
            return false;
    }
}

template <typename T>
static bool fwht_launch(ggml_backend_cuda_context & ctx, const T * src_d, float * dst_d,
                        const int n, const int64_t rows, const float scale,
                        const float * signs, const int n_blk, const ggml_tensor * dst) {
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = 4;
    const int64_t num_blocks = (rows + rows_per_block - 1) / rows_per_block;
    cudaStream_t stream = ctx.stream();
    dim3 grid_dims(num_blocks, 1, 1);
    dim3 block_dims(warp_size, rows_per_block, 1);
    const ggml_cuda_kernel_launch_params launch_params =
        ggml_cuda_kernel_launch_params(grid_dims, block_dims, 0, stream);

    switch (n) {
#define FWHT_CASE(NN) \
        case NN: \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, true>,  launch_params, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda<NN, T, false>, launch_params, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true;
        FWHT_CASE(64)
        FWHT_CASE(128)
        FWHT_CASE(256)
        default:
            break;
    }
    // From 512 up, one block of FWHT_BLOCK_THREADS per row (fwht_cuda_block).
    // The older kernels were the largest single kernel of a decode step at these widths.
    // GGML_CUDA_FWHT_LEGACY=1 restores them for A/B.
#define FWHT_SMEM_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_SMEM_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_smem<NN, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1); \
            } \
            return true; \
        }
#define FWHT_BLOCK_CASE(NN) \
        case NN: { \
            const dim3 g((unsigned) rows, 1, 1), b(FWHT_BLOCK_THREADS, 1, 1); \
            const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(g, b, 0, stream); \
            block_q8_1 * q8 = ggml_cuda_q8_1_cache_put(ctx, dst); \
            if (signs) { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, true>,  lp, src_d, dst_d, rows, scale, signs, n_blk, q8); \
            } else { \
                ggml_cuda_kernel_launch(fwht_cuda_block<NN, FWHT_BLOCK_THREADS, T, false>, lp, src_d, dst_d, rows, scale, nullptr, 1, q8); \
            } \
            return true; \
        }
    static const bool legacy = getenv("GGML_CUDA_FWHT_LEGACY") != nullptr;
    if (legacy) {
        switch (n) {
            FWHT_CASE(512)
            FWHT_CASE(1024)
            FWHT_CASE(2048)
            FWHT_SMEM_CASE(4096)
            FWHT_SMEM_CASE(8192)
            default:
                return false;
        }
    }
    switch (n) {
        FWHT_BLOCK_CASE(512)
        FWHT_BLOCK_CASE(1024)
        FWHT_BLOCK_CASE(2048)
        FWHT_BLOCK_CASE(4096)
        FWHT_BLOCK_CASE(8192)
#undef FWHT_CASE
#undef FWHT_SMEM_CASE
#undef FWHT_BLOCK_CASE
        default:
            return false;
    }
}

static bool fwht_dispatch(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst,
                          const ggml_tensor * signs_t) {
    GGML_ASSERT(ggml_nelements(src) == ggml_nelements(dst));
    if (!ggml_is_contiguous(src) || !ggml_is_contiguous(dst)) {
        return false;
    }
    const int     n    = dst->ne[0];
    const int64_t rows = ggml_nelements(dst) / n;

    if ((src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_F16) || dst->type != GGML_TYPE_F32) {
        return false;
    }

    const float * signs = nullptr;
    int n_blk = 1;
    if (signs_t) {
        if (signs_t->type != GGML_TYPE_F32 || !ggml_is_contiguous(signs_t) || signs_t->ne[0] % n != 0) {
            return false;
        }
        signs = (const float *) signs_t->data;
        n_blk = signs_t->ne[0] / n;
    }

    float * dst_d = (float *) dst->data;
    const float scale = 1 / sqrtf(n);

    if (src->type == GGML_TYPE_F32) {
        return fwht_launch<float>(ctx, (const float *) src->data, dst_d, n, rows, scale, signs, n_blk, dst);
    }
    return fwht_launch<half>(ctx, (const half *) src->data, dst_d, n, rows, scale, signs, n_blk, dst);
}

bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst) {
    GGML_ASSERT(ggml_are_same_shape(src, dst));
    return fwht_dispatch(ctx, src, dst, nullptr);
}

bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst) {
    return fwht_dispatch(ctx, src, dst, signs);
}

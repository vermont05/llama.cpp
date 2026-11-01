#include "ptq1-wmma.cuh"
#include "vecdotq.cuh"

// Layout (RDNA3 wave32 WMMA): both operands hold 16 int8 of one row per lane (lane % 16), mirrored in both half waves;
// element v of the int32 result of a lane is (row lane % 16 of the second operand, row 2*v + lane / 16 of the first).
// The first operand is the activation columns (rows past ncols_dst are 0), the second a 16-row tile of the weights, so
// a lane's results all belong to its own weight row (one weight scale) and to columns 2*v + lane / 16 (those < NC kept).
//
// Each 128-weight PTQ1_0 block decodes to 32 ints of 4 signed trits in k order (as the MMQ tile loader decodes it):
// ints 4t + g (t < 5, g < 4) are trit t of qs word g, ints 20 + 2t + g trit t of qs word 4 + g, ints 30, 31 the qh
// trits. A K chunk of 16 is ints 4c .. 4c + 3, a q8_1 block of 32 is two chunks.
//
// The activations are first repacked from q8_1 so that a lane loads its 16 bytes of a chunk, and a q8_1 block's scales
// of all columns, with one aligned load each: qs [K/16][8 columns][16 int8], d [K/32][8 columns] (columns past
// ncols_dst are 0).

static constexpr int ptq1_wmma_ncols_pad = 8;

static __global__ void ptq1_0_wmma_pack_y(
        const block_q8_1 * __restrict__ y, int8_t * __restrict__ yq, float * __restrict__ yd,
        const int nblocks_y, const int ncols, const int stride_col_y) {
    const int ib = blockIdx.x*blockDim.y + threadIdx.y; // q8_1 block along K
    const int j  = threadIdx.x / 4;                      // column, 0..7
    const int w  = threadIdx.x % 4;                      // 8-byte quarter of the 32 values
    if (ib >= nblocks_y) {
        return;
    }
    int2 v = make_int2(0, 0);
    float d = 0.0f;
    if (j < ncols) {
        const block_q8_1 * by = y + (int64_t) j*stride_col_y + ib;
        v = make_int2(get_int_b4(by->qs, 2*w), get_int_b4(by->qs, 2*w + 1));
        d = __low2float(by->ds);
    }
    // chunk 2*ib + w/2, column j, bytes 8*(w%2)
    *(int2 *) (yq + ((int64_t) (2*ib + w/2)*ptq1_wmma_ncols_pad + j)*16 + 8*(w % 2)) = v;
    if (w == 0) {
        yd[(int64_t) ib*ptq1_wmma_ncols_pad + j] = d;
    }
}

#if defined(GGML_USE_HIP) && defined(RDNA3)
typedef int ptq1_v4i __attribute__((ext_vector_type(4)));
typedef int ptq1_v8i __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ void ptq1_0_wmma_decode_qs4(uint32_t packed, int * __restrict__ dst, const int stride) {
    uint32_t v_lo = ptq1_widen_lo(packed);
    uint32_t v_hi = ptq1_widen_hi(packed);
#pragma unroll
    for (int t = 0; t < 5; ++t) {
        const uint32_t w_lo = v_lo * 3;
        const uint32_t w_hi = v_hi * 3;
        v_lo = w_lo & 0x00FF00FF;
        v_hi = w_hi & 0x00FF00FF;
        dst[t*stride] = ptq1_trits4_signed(w_lo, w_hi);
    }
}

static __device__ __forceinline__ void ptq1_0_wmma_decode(const block_ptq1_0 * __restrict__ bx, int * __restrict__ a) {
#pragma unroll
    for (int g = 0; g < 4; ++g) {
        ptq1_0_wmma_decode_qs4(get_int_b4(bx->qs, g), a + g, 4);
    }
#pragma unroll
    for (int g = 0; g < 2; ++g) {
        ptq1_0_wmma_decode_qs4(get_int_b4(bx->qs + 16, g), a + 20 + g, 2);
    }
    uint32_t v = (uint32_t) bx->qh[0] | ((uint32_t) bx->qh[1] << 16);
#pragma unroll
    for (int t = 0; t < 4; t += 2) {
        const uint32_t w0 = v * 3;
        v = w0 & 0x00FF00FF;
        const uint32_t w1 = v * 3;
        v = w1 & 0x00FF00FF;
        a[30 + t/2] = ptq1_trits4_signed(w0, w1);
    }
}
#endif // defined(GGML_USE_HIP) && defined(RDNA3)

template <int NC, int NWK>
static __global__ void __launch_bounds__(32*NWK, 1) mul_mat_ptq1_0_wmma(
        const block_ptq1_0 * __restrict__ x, const int8_t * __restrict__ yq, const float * __restrict__ yd,
        float * __restrict__ dst, const int nblocks_k, const int nrows, const int stride_row_x, const int stride_col_dst) {
#if defined(GGML_USE_HIP) && defined(RDNA3)
    constexpr int NV = (NC + 1) / 2; // results of a lane that can hold a column < NC

    const int lane = threadIdx.x;
    const int wk   = threadIdx.y;
    const int r    = lane % 16;
    const int half = lane / 16;
    const int row  = min((int) blockIdx.x*16 + r, nrows - 1);

    const block_ptq1_0 * xr = x + (int64_t) row*stride_row_x;
    const bool           bc = r < NC; // this lane's activation row is a real column

    float acc[NV] = {0.0f};

    for (int ib = wk; ib < nblocks_k; ib += NWK) {
        int a[32];
        ptq1_0_wmma_decode(xr + ib, a);
        const float dw = __half2float(xr[ib].d);

        float accb[NV] = {0.0f};
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const int    ic = 8*ib + 2*q; // first K chunk of the q8_1 block
            ptq1_v4i b0 = {0, 0, 0, 0};
            ptq1_v4i b1 = {0, 0, 0, 0};
            if (bc) {
                b0 = *(const ptq1_v4i *) (yq + ((int64_t) (ic + 0)*ptq1_wmma_ncols_pad + r)*16);
                b1 = *(const ptq1_v4i *) (yq + ((int64_t) (ic + 1)*ptq1_wmma_ncols_pad + r)*16);
            }
            const float4 d0 = *(const float4 *) (yd + (int64_t) (4*ib + q)*ptq1_wmma_ncols_pad);
            float4 d1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (NC > 4) {
                d1 = *(const float4 *) (yd + (int64_t) (4*ib + q)*ptq1_wmma_ncols_pad + 4);
            }
            const float dy[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};

            const ptq1_v4i a0 = ptq1_v4i{a[8*q + 0], a[8*q + 1], a[8*q + 2], a[8*q + 3]};
            const ptq1_v4i a1 = ptq1_v4i{a[8*q + 4], a[8*q + 5], a[8*q + 6], a[8*q + 7]};
            ptq1_v8i c = {0, 0, 0, 0, 0, 0, 0, 0};
            c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b0, true, a0, c, false);
            c = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, b1, true, a1, c, false);
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                if (2*v + 1 < NC || half == 0) {
                    accb[v] += (float) c[v] * (half ? dy[2*v + 1] : dy[2*v]);
                }
            }
        }
#pragma unroll
        for (int v = 0; v < NV; ++v) {
            acc[v] += accb[v] * dw;
        }
    }

    __shared__ float red[NWK > 1 ? NWK - 1 : 1][32][NV];
    if (NWK > 1) {
        if (wk > 0) {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                red[wk - 1][lane][v] = acc[v];
            }
        }
        __syncthreads();
        if (wk > 0) {
            return;
        }
#pragma unroll
        for (int w = 0; w < NWK - 1; ++w) {
#pragma unroll
            for (int v = 0; v < NV; ++v) {
                acc[v] += red[w][lane][v];
            }
        }
    }

    const int row_out = blockIdx.x*16 + r;
    if (row_out >= nrows) {
        return;
    }
#pragma unroll
    for (int v = 0; v < NV; ++v) {
        const int j = 2*v + half;
        if (j < NC) {
            dst[(int64_t) j*stride_col_dst + row_out] = acc[v];
        }
    }
#else
    GGML_UNUSED_VARS(x, yq, yd, dst, nblocks_k, nrows, stride_row_x, stride_col_dst);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(RDNA3)
}

template <int NC>
static void launch_mul_mat_ptq1_0_wmma(
        const void * vx, const int8_t * yq, const float * yd, float * dst, const int ncols_x, const int nrows_x,
        const int stride_row_x, const int stride_col_dst, cudaStream_t stream) {
    // GGML_CUDA_PTQ1_0_WMMA_NWK: waves along K per block (A/B)
    static const int nwk = getenv("GGML_CUDA_PTQ1_0_WMMA_NWK") ? atoi(getenv("GGML_CUDA_PTQ1_0_WMMA_NWK")) : 8;
    const dim3 grid((nrows_x + 15) / 16);
    const int  nblocks_k = ncols_x / QK_PTQ1_0;
    const block_ptq1_0 * x = (const block_ptq1_0 *) vx;
    switch (nwk) {
        case 1: mul_mat_ptq1_0_wmma<NC, 1><<<grid, dim3(32, 1), 0, stream>>>(x, yq, yd, dst, nblocks_k, nrows_x, stride_row_x, stride_col_dst); break;
        case 2: mul_mat_ptq1_0_wmma<NC, 2><<<grid, dim3(32, 2), 0, stream>>>(x, yq, yd, dst, nblocks_k, nrows_x, stride_row_x, stride_col_dst); break;
        case 4: mul_mat_ptq1_0_wmma<NC, 4><<<grid, dim3(32, 4), 0, stream>>>(x, yq, yd, dst, nblocks_k, nrows_x, stride_row_x, stride_col_dst); break;
        default: mul_mat_ptq1_0_wmma<NC, 8><<<grid, dim3(32, 8), 0, stream>>>(x, yq, yd, dst, nblocks_k, nrows_x, stride_row_x, stride_col_dst); break;
    }
}

bool ggml_cuda_mul_mat_ptq1_0_wmma(
        ggml_backend_cuda_context & ctx, const void * vx, const void * vy, float * dst,
        const int ncols_x, const int nrows_x, const int ncols_dst,
        const int stride_row_x, const int stride_col_y, const int stride_col_dst, cudaStream_t stream) {
    static const bool disabled = getenv("GGML_CUDA_NO_PTQ1_0_WMMA") != nullptr;
    // GGML_CUDA_PTQ1_0_WMMA_MIN: the fewest columns that take this kernel
    static const int ncols_min = getenv("GGML_CUDA_PTQ1_0_WMMA_MIN") ? atoi(getenv("GGML_CUDA_PTQ1_0_WMMA_MIN")) : 2;
    if (disabled || ncols_dst < std::max(ncols_min, 2) || ncols_dst > ptq1_wmma_ncols_pad || ncols_x % QK_PTQ1_0 != 0) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (!GGML_CUDA_CC_IS_RDNA3(cc)) {
        return false;
    }

    const int nblocks_y = ncols_x / QK8_1;
    ggml_cuda_pool_alloc<int8_t> yq(ctx.pool(), (size_t) ncols_x*ptq1_wmma_ncols_pad);
    ggml_cuda_pool_alloc<float>  yd(ctx.pool(), (size_t) nblocks_y*ptq1_wmma_ncols_pad);
    {
        const dim3 block(32, 8);
        const dim3 grid((nblocks_y + 7) / 8);
        ptq1_0_wmma_pack_y<<<grid, block, 0, stream>>>((const block_q8_1 *) vy, yq.get(), yd.get(), nblocks_y, ncols_dst, stride_col_y);
    }

    switch (ncols_dst) {
        case 2: launch_mul_mat_ptq1_0_wmma<2>(vx, yq.get(), yd.get(), dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, stream); break;
        case 3: launch_mul_mat_ptq1_0_wmma<3>(vx, yq.get(), yd.get(), dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, stream); break;
        case 4: launch_mul_mat_ptq1_0_wmma<4>(vx, yq.get(), yd.get(), dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, stream); break;
        case 5: launch_mul_mat_ptq1_0_wmma<5>(vx, yq.get(), yd.get(), dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, stream); break;
        case 6: launch_mul_mat_ptq1_0_wmma<6>(vx, yq.get(), yd.get(), dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, stream); break;
        case 7: launch_mul_mat_ptq1_0_wmma<7>(vx, yq.get(), yd.get(), dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, stream); break;
        default: launch_mul_mat_ptq1_0_wmma<8>(vx, yq.get(), yd.get(), dst, ncols_x, nrows_x, stride_row_x, stride_col_dst, stream); break;
    }
    return true;
}

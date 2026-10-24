#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-decode-q4.cuh"

// Single-token decode attention for a Q4_0 K/V cache with grouped-query attention (head size 256).
//
// The generic vector kernel runs one block per Q head, so with GQA every K/V row is fetched and unpacked once per
// Q head, through 2-byte loads (q4_0 blocks are 18 bytes). On RDNA3 that kernel is bound by load-instruction issue
// and latency rather than by DRAM bandwidth. Here one block serves all G Q heads of a K/V head:
//   - each warp stages its own 16-row K and V tiles into shared memory with 16-byte coalesced loads,
//   - QK: 8 lanes per K row, one q4_0 block per lane, dp4a against a q8_1 copy of Q for all G heads,
//   - PV: 32 lanes per V row, 8 output dims per lane, f32 accumulators for all G heads,
//   - online softmax per warp; the warps of a block are merged at the end, and blocks over the K/V length are
//     merged by flash_attn_combine_results.
// GGML_CUDA_NO_FA_DECODE_Q4=1 disables it.

static constexpr int fattn_dq4_D       = 256;
static constexpr int fattn_dq4_nwarps  = 4;
static constexpr int fattn_dq4_rows    = 16;                                // K/V rows per warp per tile
static constexpr int fattn_dq4_tile    = fattn_dq4_nwarps*fattn_dq4_rows;   // K/V rows per block per tile
static constexpr int fattn_dq4_nb      = fattn_dq4_D/QK4_0;                 // q4_0 blocks per row
static constexpr int fattn_dq4_row_b   = fattn_dq4_nb*sizeof(block_q4_0);   // 144 bytes per row
static constexpr int fattn_dq4_row_u4  = fattn_dq4_row_b/16;                // 9 x 16 bytes per row

static_assert(fattn_dq4_row_b % 16 == 0, "bad q4_0 row size");

static __device__ __forceinline__ void fattn_dq4_wave_sync() {
#ifdef GGML_USE_HIP
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
#else
    __syncwarp();
#endif // GGML_USE_HIP
}

// 4 bytes starting at byte offset off (a multiple of 2) of a 4-byte aligned shared buffer.
static __device__ __forceinline__ uint32_t fattn_dq4_load_u32_a2(const uint32_t * buf, const int off) {
    const uint32_t lo = buf[off/4];
    const uint32_t hi = buf[off/4 + 1];
    return (off & 2) ? __builtin_amdgcn_alignbyte(hi, lo, 2) : lo;
}

template <int G>
__launch_bounds__(fattn_dq4_nwarps*WARP_SIZE, 1)
static __global__ void flash_attn_ext_decode_q4_0(
        const char * __restrict__ Q,
        const char * __restrict__ K,
        const char * __restrict__ V,
        const char * __restrict__ mask,
        const char * __restrict__ sinks,
        const int  * __restrict__ KV_max,
        float      * __restrict__ dst,
        float2     * __restrict__ dst_meta,
        const float scale,
        const float max_bias,
        const float m0,
        const float m1,
        const uint32_t n_head_log2,
        const float logit_softcap,
        const int32_t ne00, const uint3   ne01, const int32_t ne02, const int32_t ne03,
                            const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne10, const int32_t ne11, const int32_t ne12, const int32_t ne13,
                            const int32_t nb11, const int32_t nb12, const int64_t nb13,
                            const int32_t nb21, const int32_t nb22, const int64_t nb23,
                            const int32_t ne31, const int32_t ne32, const int32_t ne33,
                            const int32_t nb31, const int32_t nb32, const int64_t nb33) {
#if defined(GGML_USE_HIP) && defined(RDNA)
    GGML_UNUSED_VARS(sinks, max_bias, m0, m1, n_head_log2, logit_softcap, ne00, ne03, nb01, nb03,
        ne10, ne13, ne31, ne32, nb32);

    constexpr int D      = fattn_dq4_D;
    constexpr int nwarps = fattn_dq4_nwarps;
    constexpr int R      = fattn_dq4_rows;
    constexpr int NB     = fattn_dq4_nb;

    const int warp = threadIdx.y;
    const int lane = threadIdx.x;
    const int tid  = warp*WARP_SIZE + lane;

    const int sequence = blockIdx.z / ne12;
    const int head_kv  = blockIdx.z - sequence*ne12;
    const int head0    = head_kv*G; // gqa_ratio == G, checked on the host

    Q += nb03*sequence + int64_t(nb02)*head0;
    K += nb13*sequence + int64_t(nb12)*head_kv;
    V += nb23*sequence + int64_t(nb22)*head_kv;
    const half * maskh = (const half *) (mask + nb33*(sequence % ne33));

    __shared__ int      Q_qs[G][NB][QI8_1];
    __shared__ float2   Q_ds[G][NB];
    // Per-warp staging; one extra 16 bytes so the unaligned 4-byte reads of the last block stay in bounds.
    __shared__ uint4    KV_s[nwarps][2][R*fattn_dq4_row_u4 + 1];
    __shared__ float    P_s[nwarps][R][G];
    __shared__ float    O_s[G][D];
    __shared__ float2   ml_s[nwarps][G];

    // Q * scale -> q8_1, one 32-value block per thread.
    for (int t = tid; t < G*NB; t += nwarps*WARP_SIZE) {
        const int h = t / NB;
        const int b = t - h*NB;
        const float4 * x4 = (const float4 *) ((const float *) (Q + int64_t(nb02)*h) + b*QK8_1);
        float v[QK8_1];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < QK8_1/4; ++i) {
            const float4 x = x4[i];
            v[4*i + 0] = x.x*scale; v[4*i + 1] = x.y*scale; v[4*i + 2] = x.z*scale; v[4*i + 3] = x.w*scale;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                amax = fmaxf(amax, fabsf(v[4*i + k]));
            }
        }
        const float d  = amax / 127.0f;
        const float id = d > 0.0f ? 1.0f/d : 0.0f;
        int sum = 0;
#pragma unroll
        for (int i = 0; i < QI8_1; ++i) {
            uint32_t packed = 0;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int q = roundf(v[4*i + k]*id);
                sum += q;
                packed |= uint32_t(q & 0xFF) << (8*k);
            }
            Q_qs[h][b][i] = packed;
        }
        Q_ds[h][b] = make_float2(d, d*sum);
    }
    for (int t = tid; t < G*D/4; t += nwarps*WARP_SIZE) {
        ((float4 *) O_s)[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    __syncthreads();

    // QK role: lane handles q4_0 block kb of row kr of each group of 4 rows.
    const int kb = lane % NB;
    const int kr = lane / NB;
    int    q_reg[G][QI8_1];
    float2 qd_reg[G];
#pragma unroll
    for (int h = 0; h < G; ++h) {
#pragma unroll
        for (int i = 0; i < QI8_1; ++i) {
            q_reg[h][i] = Q_qs[h][kb][i];
        }
        qd_reg[h] = Q_ds[h][kb];
    }

    // PV role: lane handles dims 32*vb + 4*vj + {0..3} and 32*vb + 16 + 4*vj + {0..3}.
    const int vb = lane / 4;
    const int vj = lane % 4;
    float acc[G][8];
    float csum[G]; // sum of P*d_V, for the -8 offset of q4_0
    float KQ_max[G];
    float KQ_sum[G];
#pragma unroll
    for (int h = 0; h < G; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[h][i] = 0.0f;
        }
        csum[h]   = 0.0f;
        KQ_max[h] = -FLT_MAX/2.0f;
        KQ_sum[h] = 0.0f;
    }

    uint4 * K_s = KV_s[warp][0];
    uint4 * V_s = KV_s[warp][1];
    const uint32_t * K_s32 = (const uint32_t *) K_s;
    const uint32_t * V_s32 = (const uint32_t *) V_s;

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    for (int t0 = blockIdx.y*fattn_dq4_tile; t0 < k_VKQ_max; t0 += gridDim.y*fattn_dq4_tile) {
        const int row0 = t0 + warp*R;

        // Stage this warp's K and V rows.
        fattn_dq4_wave_sync(); // the previous tile's reads are done
#pragma unroll
        for (int i0 = 0; i0 < R*fattn_dq4_row_u4; i0 += WARP_SIZE) {
            const int i = i0 + lane;
            if (i0 + WARP_SIZE <= R*fattn_dq4_row_u4 || i < R*fattn_dq4_row_u4) {
                const int r = i / fattn_dq4_row_u4;
                const int c = i - r*fattn_dq4_row_u4;
                K_s[i] = ((const uint4 *) (K + int64_t(row0 + r)*nb11))[c];
                V_s[i] = ((const uint4 *) (V + int64_t(row0 + r)*nb21))[c];
            }
        }
        fattn_dq4_wave_sync();

        // S = Q K^T for the warp's R rows; S[p][h] is row p*4 + kr, replicated over the 8 lanes of the row.
        float S[R/4][G];
#pragma unroll
        for (int p = 0; p < R/4; ++p) {
            const int rr  = p*4 + kr;
            const int off = rr*fattn_dq4_row_b + kb*int(sizeof(block_q4_0));
            const float dK = __half2float(__ushort_as_half((unsigned short) (fattn_dq4_load_u32_a2(K_s32, off) & 0xFFFF)));
            int qs[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                qs[i] = fattn_dq4_load_u32_a2(K_s32, off + 2 + 4*i);
            }
            const float mval = __half2float(maskh[row0 + rr]);
#pragma unroll
            for (int h = 0; h < G; ++h) {
                int sumi = 0;
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    sumi = ggml_cuda_dp4a( qs[i]       & 0x0F0F0F0F, q_reg[h][i],     sumi);
                    sumi = ggml_cuda_dp4a((qs[i] >> 4) & 0x0F0F0F0F, q_reg[h][i + 4], sumi);
                }
                float s = dK*(qd_reg[h].x*sumi - 8.0f*qd_reg[h].y);
                s = warp_reduce_sum<NB>(s);
                S[p][h] = s + mval;
            }
        }

        // Online softmax over the tile.
#pragma unroll
        for (int h = 0; h < G; ++h) {
            float tmax = S[0][h];
#pragma unroll
            for (int p = 1; p < R/4; ++p) {
                tmax = fmaxf(tmax, S[p][h]);
            }
#pragma unroll
            for (int offset = NB; offset < WARP_SIZE; offset <<= 1) {
                tmax = fmaxf(tmax, __shfl_xor_sync(0xFFFFFFFF, tmax, offset, WARP_SIZE));
            }
            const float m_new = fmaxf(KQ_max[h], tmax);
            const float alpha = expf(KQ_max[h] - m_new);
            KQ_max[h] = m_new;

            float tsum = 0.0f;
#pragma unroll
            for (int p = 0; p < R/4; ++p) {
                S[p][h] = expf(S[p][h] - m_new);
                tsum += S[p][h];
            }
#pragma unroll
            for (int offset = NB; offset < WARP_SIZE; offset <<= 1) {
                tsum += __shfl_xor_sync(0xFFFFFFFF, tsum, offset, WARP_SIZE);
            }
            KQ_sum[h] = KQ_sum[h]*alpha + tsum;

#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[h][i] *= alpha;
            }
            csum[h] *= alpha;
        }
        if (kb == 0) {
#pragma unroll
            for (int p = 0; p < R/4; ++p) {
#pragma unroll
                for (int h = 0; h < G; ++h) {
                    P_s[warp][p*4 + kr][h] = S[p][h];
                }
            }
        }
        fattn_dq4_wave_sync();

        // O += P V
#pragma unroll 4
        for (int rr = 0; rr < R; ++rr) {
            const int off = rr*fattn_dq4_row_b + vb*int(sizeof(block_q4_0));
            const float dV = __half2float(__ushort_as_half((unsigned short) (fattn_dq4_load_u32_a2(V_s32, off) & 0xFFFF)));
            const uint32_t q = fattn_dq4_load_u32_a2(V_s32, off + 2 + 4*vj);
            float vf[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                vf[k]     = float((q >> (8*k))     & 0x0F);
                vf[4 + k] = float((q >> (8*k + 4)) & 0x0F);
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float w = P_s[warp][rr][h]*dV;
                csum[h] += w;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[h][i] += w*vf[i];
                }
            }
        }
    }

    // Merge the warps of the block.
    if (lane == 0) {
#pragma unroll
        for (int h = 0; h < G; ++h) {
            ml_s[warp][h] = make_float2(KQ_max[h], KQ_sum[h]);
        }
    }
    __syncthreads();
    float M[G];
    float L[G];
#pragma unroll
    for (int h = 0; h < G; ++h) {
        M[h] = ml_s[0][h].x;
#pragma unroll
        for (int w = 1; w < nwarps; ++w) {
            M[h] = fmaxf(M[h], ml_s[w][h].x);
        }
        L[h] = 0.0f;
#pragma unroll
        for (int w = 0; w < nwarps; ++w) {
            L[h] += ml_s[w][h].y*expf(ml_s[w][h].x - M[h]);
        }
        const float f = expf(KQ_max[h] - M[h]);
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            atomicAdd(&O_s[h][32*vb +      4*vj + k], (acc[h][k]     - 8.0f*csum[h])*f);
            atomicAdd(&O_s[h][32*vb + 16 + 4*vj + k], (acc[h][4 + k] - 8.0f*csum[h])*f);
        }
    }
    __syncthreads();

    for (int i = tid; i < G*D; i += nwarps*WARP_SIZE) {
        const int h   = i / D;
        const int dim = i - h*D;
        float val = O_s[h][dim];
        if (gridDim.y == 1) {
            val /= L[h];
        }
        dst[((int64_t(sequence)*int(ne01.z)*ne02 + head0 + h)*gridDim.y + blockIdx.y)*D + dim] = val;
    }
    if (gridDim.y != 1 && tid < G) {
        dst_meta[(int64_t(sequence)*int(ne01.z)*ne02 + head0 + tid)*gridDim.y + blockIdx.y] = make_float2(M[tid], L[tid]);
    }
#else
    GGML_UNUSED_VARS(Q, K, V, mask, sinks, KV_max, dst, dst_meta, scale,
        max_bias, m0, m1, n_head_log2, logit_softcap,
        ne00, ne01, ne02, ne03,
              nb01, nb02, nb03,
        ne10, ne11, ne12, ne13,
              nb11, nb12, nb13,
              nb21, nb22, nb23,
        ne31, ne32, ne33,
              nb31, nb32, nb33);
    NO_DEVICE_CODE;
#endif // defined(GGML_USE_HIP) && defined(RDNA)
}

bool ggml_cuda_flash_attn_ext_decode_q4_supported(const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    static const bool disabled = getenv("GGML_CUDA_NO_FA_DECODE_Q4") != nullptr;
    if (disabled || !GGML_CUDA_CC_IS_RDNA3_0(ggml_cuda_info().devices[ggml_cuda_get_device()].cc)) {
        return false;
    }
    const ggml_tensor * Q     = dst->src[0];
    const ggml_tensor * K     = dst->src[1];
    const ggml_tensor * V     = dst->src[2];
    const ggml_tensor * mask  = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];

    float max_bias      = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&max_bias,      (const float *) dst->op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) dst->op_params + 2, sizeof(float));

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_Q4_0 || V->type != GGML_TYPE_Q4_0) {
        return false;
    }
    if (Q->ne[0] != fattn_dq4_D || K->ne[0] != fattn_dq4_D || V->ne[0] != fattn_dq4_D || Q->ne[1] != 1) {
        return false;
    }
    if (!mask || sinks || max_bias != 0.0f || logit_softcap != 0.0f || mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (K->ne[1] % fattn_dq4_tile != 0 || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio != 4 && gqa_ratio != 6 && gqa_ratio != 8) {
        return false;
    }
    if ((uintptr_t) K->data % 16 != 0 || (uintptr_t) V->data % 16 != 0 || (uintptr_t) Q->data % 16 != 0) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (K->nb[i] % 16 != 0 || V->nb[i] % 16 != 0 || Q->nb[i] % 16 != 0) {
            return false;
        }
    }
    return true;
#else
    GGML_UNUSED(dst);
    return false;
#endif // GGML_USE_HIP
}

template <int G>
static void ggml_cuda_flash_attn_ext_decode_q4_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    fattn_kernel_t fattn_kernel = flash_attn_ext_decode_q4_0<G>;
    launch_fattn<fattn_dq4_D, 1, G>(ctx, dst, fattn_kernel, fattn_dq4_nwarps, 0, fattn_dq4_tile, false, false, false);
}

void ggml_cuda_flash_attn_ext_decode_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int gqa_ratio = dst->src[0]->ne[2] / dst->src[1]->ne[2];
    switch (gqa_ratio) {
        case 4: ggml_cuda_flash_attn_ext_decode_q4_case<4>(ctx, dst); break;
        case 6: ggml_cuda_flash_attn_ext_decode_q4_case<6>(ctx, dst); break;
        case 8: ggml_cuda_flash_attn_ext_decode_q4_case<8>(ctx, dst); break;
        default: GGML_ABORT("unsupported gqa ratio");
    }
}

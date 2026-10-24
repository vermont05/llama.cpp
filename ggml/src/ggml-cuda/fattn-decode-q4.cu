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

#if defined(GGML_USE_HIP) && defined(RDNA)
// Lane exchanges without the LDS crossbar: DPP row_xmask for xor 1..8 within a row of 16, permlanex16 for xor 16.
template <int mask>
static __device__ __forceinline__ float fattn_dq4_xor(const float x) {
    if constexpr (mask == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), 0x160 | mask, 0xF, 0xF, false));
    }
}
#endif // defined(GGML_USE_HIP) && defined(RDNA)

#if defined(GGML_USE_HIP) && defined(RDNA)
// Byte k of x as a float in one instruction; the compiler emits v_bfe_u32 + v_cvt_f32_ubyte0 instead.
template <int k>
static __device__ __forceinline__ float fattn_dq4_ubyte(const uint32_t x) {
    float f;
    if constexpr (k == 0) {
        asm("v_cvt_f32_ubyte0 %0, %1" : "=v"(f) : "v"(x));
    } else if constexpr (k == 1) {
        asm("v_cvt_f32_ubyte1 %0, %1" : "=v"(f) : "v"(x));
    } else if constexpr (k == 2) {
        asm("v_cvt_f32_ubyte2 %0, %1" : "=v"(f) : "v"(x));
    } else {
        asm("v_cvt_f32_ubyte3 %0, %1" : "=v"(f) : "v"(x));
    }
    return f;
}
#endif // defined(GGML_USE_HIP) && defined(RDNA)

// 4 bytes starting at byte offset off (a multiple of 2) of a 4-byte aligned shared buffer.
static __device__ __forceinline__ uint32_t fattn_dq4_load_u32_a2(const uint32_t * buf, const int off) {
    const uint32_t lo = buf[off/4];
    const uint32_t hi = buf[off/4 + 1];
    return (off & 2) ? __builtin_amdgcn_alignbyte(hi, lo, 2) : lo;
}

#ifndef PRISM_FA_DQ4_WAVES_PER_EU
#define PRISM_FA_DQ4_WAVES_PER_EU 4
#endif

template <int G>
#ifdef GGML_USE_HIP
__attribute__((amdgpu_flat_work_group_size(fattn_dq4_nwarps*WARP_SIZE, fattn_dq4_nwarps*WARP_SIZE), amdgpu_waves_per_eu(PRISM_FA_DQ4_WAVES_PER_EU)))
#else
__launch_bounds__(fattn_dq4_nwarps*WARP_SIZE, 1)
#endif // GGML_USE_HIP
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

    __shared__ __align__(16) int Q_qs[G][NB][QI8_1];
    __shared__ float2   Q_ds[G][NB];
    // Per-warp K/V staging (one extra 16 bytes so the unaligned 4-byte reads of the last block stay in bounds).
    // The merged output O_s reuses the same memory after the main loop.
    constexpr int ne_stage = nwarps*2*(R*fattn_dq4_row_u4 + 1);
    static_assert(ne_stage*16 >= G*D*4, "staging too small for O_s");
    __shared__ uint4    KV_s_raw[ne_stage];
    __shared__ float    P_s[nwarps][R][G];
    __shared__ float2   ml_s[nwarps][G];
    float (*O_s)[D] = (float (*)[D]) KV_s_raw;

    // Q * scale * log2(e) -> q8_1, one 32-value block per thread. The softmax below works in base 2.
    const float scale_l2 = scale*1.4426950408889634f;
    for (int t = tid; t < G*NB; t += nwarps*WARP_SIZE) {
        const int h = t / NB;
        const int b = t - h*NB;
        const float4 * x4 = (const float4 *) ((const float *) (Q + int64_t(nb02)*h) + b*QK8_1);
        float v[QK8_1];
        float amax = 0.0f;
#pragma unroll
        for (int i = 0; i < QK8_1/4; ++i) {
            const float4 x = x4[i];
            v[4*i + 0] = x.x*scale_l2; v[4*i + 1] = x.y*scale_l2; v[4*i + 2] = x.z*scale_l2; v[4*i + 3] = x.w*scale_l2;
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
    __syncthreads();

    // QK role: lane handles q4_0 block kb of row kr of each group of 4 rows.
    const int kb = lane % NB;
    const int kr = lane / NB;

    // PV role: lane handles dims 32*vb + 4*vj + {0..3} and 32*vb + 16 + 4*vj + {0..3}.
    const int vb = lane / 4;
    const int vj = lane % 4;
    float acc[G][8];
    float csum[G]; // sum of P*d_V, for the -8 offset of q4_0
#pragma unroll
    for (int h = 0; h < G; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[h][i] = 0.0f;
        }
        csum[h] = 0.0f;
    }
    // Softmax role: after the reduce-scatter below, lane holds the scores of head hs = lane % 8 (heads >= G are
    // padding) for rows p*4 + kr; it keeps the running max and sum of that head.
    static_assert(G <= NB, "too many Q heads per K/V head");
    const int hs = lane % NB;
    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    uint4 * K_s = KV_s_raw + (2*warp + 0)*(R*fattn_dq4_row_u4 + 1);
    uint4 * V_s = KV_s_raw + (2*warp + 1)*(R*fattn_dq4_row_u4 + 1);
    const uint32_t * K_s32 = (const uint32_t *) K_s;
    const uint32_t * V_s32 = (const uint32_t *) V_s;

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    // The lane's share of a warp tile, prefetched into registers one tile ahead.
    // Named registers, not arrays: the compiler kept arrays of uint4 here in scratch memory.
    static_assert(R*fattn_dq4_row_u4 > 4*WARP_SIZE && R*fattn_dq4_row_u4 <= 5*WARP_SIZE, "prefetch assumes 5 loads per lane");
    const bool ld4 = 4*WARP_SIZE + lane < R*fattn_dq4_row_u4;
    uint4 pk0, pk1, pk2, pk3, pk4 = make_uint4(0, 0, 0, 0);
    uint4 pv0, pv1, pv2, pv3, pv4 = make_uint4(0, 0, 0, 0);
#define FATTN_DQ4_LD(j, row0_load, dk, dv) {                                                    \
        const int i = (j)*WARP_SIZE + lane;                                                     \
        const int r = i / fattn_dq4_row_u4;                                                     \
        const int c = i - r*fattn_dq4_row_u4;                                                   \
        dk = ((const uint4 *) (K + int64_t((row0_load) + r)*nb11))[c];                          \
        dv = ((const uint4 *) (V + int64_t((row0_load) + r)*nb21))[c];                          \
    }
#define FATTN_DQ4_LOAD_TILE(row0_load) {                                                        \
        FATTN_DQ4_LD(0, row0_load, pk0, pv0); FATTN_DQ4_LD(1, row0_load, pk1, pv1);             \
        FATTN_DQ4_LD(2, row0_load, pk2, pv2); FATTN_DQ4_LD(3, row0_load, pk3, pv3);             \
        if (ld4) { FATTN_DQ4_LD(4, row0_load, pk4, pv4); }                                      \
    }

    int t0 = blockIdx.y*fattn_dq4_tile;
    if (t0 < k_VKQ_max) {
        FATTN_DQ4_LOAD_TILE(t0 + warp*R);
    }
    for (; t0 < k_VKQ_max; t0 += gridDim.y*fattn_dq4_tile) {
        const int row0 = t0 + warp*R;

        // Stage this warp's K and V rows, then start loading the next tile.
        fattn_dq4_wave_sync(); // the previous tile's reads are done
        K_s[0*WARP_SIZE + lane] = pk0; V_s[0*WARP_SIZE + lane] = pv0;
        K_s[1*WARP_SIZE + lane] = pk1; V_s[1*WARP_SIZE + lane] = pv1;
        K_s[2*WARP_SIZE + lane] = pk2; V_s[2*WARP_SIZE + lane] = pv2;
        K_s[3*WARP_SIZE + lane] = pk3; V_s[3*WARP_SIZE + lane] = pv3;
        if (ld4) {
            K_s[4*WARP_SIZE + lane] = pk4; V_s[4*WARP_SIZE + lane] = pv4;
        }
        fattn_dq4_wave_sync();
        if (t0 + gridDim.y*fattn_dq4_tile < k_VKQ_max) {
            FATTN_DQ4_LOAD_TILE(row0 + gridDim.y*fattn_dq4_tile);
        }

        // S = Q K^T for the warp's R rows. Each lane computes the partial dot products of its q4_0 block for all
        // heads; a reduce-scatter over the 8 lanes of the row leaves the full score of head hs in each lane.
        float S[R/4];
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
            float v[NB];
#pragma unroll
            for (int h = 0; h < NB; ++h) {
                if (h >= G) {
                    v[h] = 0.0f;
                    continue;
                }
                const int4 qa = ((const int4 *) Q_qs[h][kb])[0];
                const int4 qb = ((const int4 *) Q_qs[h][kb])[1];
                const float2 qd = Q_ds[h][kb];
                int sumi = 0;
                sumi = ggml_cuda_dp4a( qs[0]       & 0x0F0F0F0F, qa.x, sumi);
                sumi = ggml_cuda_dp4a( qs[1]       & 0x0F0F0F0F, qa.y, sumi);
                sumi = ggml_cuda_dp4a( qs[2]       & 0x0F0F0F0F, qa.z, sumi);
                sumi = ggml_cuda_dp4a( qs[3]       & 0x0F0F0F0F, qa.w, sumi);
                sumi = ggml_cuda_dp4a((qs[0] >> 4) & 0x0F0F0F0F, qb.x, sumi);
                sumi = ggml_cuda_dp4a((qs[1] >> 4) & 0x0F0F0F0F, qb.y, sumi);
                sumi = ggml_cuda_dp4a((qs[2] >> 4) & 0x0F0F0F0F, qb.z, sumi);
                sumi = ggml_cuda_dp4a((qs[3] >> 4) & 0x0F0F0F0F, qb.w, sumi);
                v[h] = dK*(qd.x*sumi - 8.0f*qd.y);
            }
            // Reduce-scatter: at each step a lane keeps half of its heads and adds the partner's copy of them.
            const bool b4 = lane & 4;
            float u4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float keep = b4 ? v[i + 4] : v[i];
                const float send = b4 ? v[i]     : v[i + 4];
                u4[i] = keep + fattn_dq4_xor<4>(send);
            }
            const bool b2 = lane & 2;
            float u2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float keep = b2 ? u4[i + 2] : u4[i];
                const float send = b2 ? u4[i]     : u4[i + 2];
                u2[i] = keep + fattn_dq4_xor<2>(send);
            }
            const bool b1 = lane & 1;
            const float sc = (b1 ? u2[1] : u2[0]) + fattn_dq4_xor<1>(b1 ? u2[0] : u2[1]);
            S[p] = sc + __half2float(maskh[row0 + rr])*1.4426950408889634f;
        }

        // Online softmax over the tile, for head hs.
        float tmax = S[0];
#pragma unroll
        for (int p = 1; p < R/4; ++p) {
            tmax = fmaxf(tmax, S[p]);
        }
        tmax = fmaxf(tmax, fattn_dq4_xor<8>(tmax));
        tmax = fmaxf(tmax, fattn_dq4_xor<16>(tmax));
        const float m_new = fmaxf(KQ_max, tmax);
        const float alpha = __builtin_amdgcn_exp2f(KQ_max - m_new);
        KQ_max = m_new;
        float tsum = 0.0f;
#pragma unroll
        for (int p = 0; p < R/4; ++p) {
            S[p] = __builtin_amdgcn_exp2f(S[p] - m_new);
            tsum += S[p];
        }
        tsum += fattn_dq4_xor<8>(tsum);
        tsum += fattn_dq4_xor<16>(tsum);
        KQ_sum = KQ_sum*alpha + tsum;
        if (hs < G) {
#pragma unroll
            for (int p = 0; p < R/4; ++p) {
                P_s[warp][p*4 + kr][hs] = S[p];
            }
        }
        // Rescale the accumulators of every head with the alpha of lane h (which holds head h).
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float alpha_h = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(alpha), h));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[h][i] *= alpha_h;
            }
            csum[h] *= alpha_h;
        }
        fattn_dq4_wave_sync();

        // O += P V
#pragma unroll 4
        for (int rr = 0; rr < R; ++rr) {
            const int off = rr*fattn_dq4_row_b + vb*int(sizeof(block_q4_0));
            const float dV = __half2float(__ushort_as_half((unsigned short) (fattn_dq4_load_u32_a2(V_s32, off) & 0xFFFF)));
            const uint32_t q = fattn_dq4_load_u32_a2(V_s32, off + 2 + 4*vj);
            const uint32_t qlo =  q       & 0x0F0F0F0F;
            const uint32_t qhi = (q >> 4) & 0x0F0F0F0F;
            const float vf[8] = {
                fattn_dq4_ubyte<0>(qlo), fattn_dq4_ubyte<1>(qlo), fattn_dq4_ubyte<2>(qlo), fattn_dq4_ubyte<3>(qlo),
                fattn_dq4_ubyte<0>(qhi), fattn_dq4_ubyte<1>(qhi), fattn_dq4_ubyte<2>(qhi), fattn_dq4_ubyte<3>(qhi),
            };
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

#undef FATTN_DQ4_LOAD_TILE
#undef FATTN_DQ4_LD

    // Merge the warps of the block. O_s aliases the staging buffers, so clear it once every warp is done with them.
    __syncthreads();
    for (int t = tid; t < G*D/4; t += nwarps*WARP_SIZE) {
        ((float4 *) O_s)[t] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    if (lane < G) {
        ml_s[warp][lane] = make_float2(KQ_max, KQ_sum);
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
            L[h] += ml_s[w][h].y*__builtin_amdgcn_exp2f(ml_s[w][h].x - M[h]);
        }
    }
    // Add the warps' outputs in a fixed order, so that results are deterministic (float atomics are not).
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        if (warp == w) {
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float f = __builtin_amdgcn_exp2f(ml_s[warp][h].x - M[h]);
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    O_s[h][32*vb +      4*vj + k] += (acc[h][k]     - 8.0f*csum[h])*f;
                    O_s[h][32*vb + 16 + 4*vj + k] += (acc[h][4 + k] - 8.0f*csum[h])*f;
                }
            }
        }
        __syncthreads();
    }

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
        dst_meta[(int64_t(sequence)*int(ne01.z)*ne02 + head0 + tid)*gridDim.y + blockIdx.y] = make_float2(M[tid]*0.6931471805599453f, L[tid]); // combine works in base e
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

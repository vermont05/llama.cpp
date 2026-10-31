#include "common.cuh"
#include "fattn-common.cuh"
#include "fattn-decode-turbo3.cuh"

// Single-token decode attention for a TURBO3_0 K/V cache with grouped-query attention (head size 256), on the
// pattern of flash_attn_ext_decode_q4_0. The cache holds K and V rotated (each 128-value group by M = D(s2) H/sqrt(128)
// D(s1), see ggml-common.h), and M is orthogonal, so the kernel works in the rotated domain:
//   q.k = (Mq).(Mk), and sum_i p_i v_i = M^T sum_i p_i (M v_i).
// Q is rotated once per block, the output once after the warps are merged; the split over the K/V length is merged
// by flash_attn_combine_results, which is linear, so it can stay in the original domain.
//   - each warp stages its own 8-row K and V tiles (100 bytes per row, 4-byte loads) into shared memory,
//   - QK: 16 lanes per K row, 16 values per lane; pairs of values are decoded through a 64-entry half2 table and
//     multiplied with v_dot2_f32_f16 against a f16 copy of the rotated Q of all G heads in shared memory,
//   - PV: 32 lanes per V row, 8 dims per lane, decoded through a 64-entry float2 table, f32 accumulators,
//   - online softmax per warp, warps merged at the end as in the q4_0 kernel.
// GGML_CUDA_NO_FA_DECODE_TURBO3=1 disables it (attention then goes through the f16 copy and the tile kernel).

static constexpr int fattn_dt3_D      = 256;
static constexpr int fattn_dt3_nwarps = 4;
static constexpr int fattn_dt3_rows   = 8;                                   // K/V rows per warp per tile (16 spills at G = 6)
static constexpr int fattn_dt3_tile   = fattn_dt3_nwarps*fattn_dt3_rows;     // K/V rows per block per tile
static constexpr int fattn_dt3_row_b  = (fattn_dt3_D/QK_TURBO3)*sizeof(block_turbo3_0); // 100 bytes per row
static constexpr int fattn_dt3_row_w  = fattn_dt3_row_b/4;                   // 25 dwords per row
static constexpr int fattn_dt3_tile_w = fattn_dt3_rows*fattn_dt3_row_w;      // 400 dwords per warp tile
static constexpr int fattn_dt3_nld    = (fattn_dt3_tile_w + WARP_SIZE - 1)/WARP_SIZE; // dword loads per lane

static_assert(fattn_dt3_row_b % 4 == 0, "bad turbo3 row size");
static_assert(sizeof(block_turbo3_0) == 50, "the offsets below assume the 50-byte turbo3 block");

typedef _Float16 fattn_dt3_h2 __attribute__((ext_vector_type(2)));

static __device__ __forceinline__ void fattn_dt3_wave_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

#if defined(GGML_USE_HIP) && defined(RDNA)
template <int mask>
static __device__ __forceinline__ float fattn_dt3_xor(const float x) {
    if constexpr (mask == 16) {
        return __int_as_float(__builtin_amdgcn_permlanex16(__float_as_int(x), __float_as_int(x), 0x76543210, 0xfedcba98, false, false));
    } else {
        return __int_as_float(__builtin_amdgcn_update_dpp(__float_as_int(x), __float_as_int(x), 0x160 | mask, 0xF, 0xF, false));
    }
}
#endif // defined(GGML_USE_HIP) && defined(RDNA)

// 4 bytes starting at byte offset off (a multiple of 2) of a 4-byte aligned shared buffer.
static __device__ __forceinline__ uint32_t fattn_dt3_load_u32_a2(const uint32_t * buf, const int off) {
    const uint32_t lo = buf[off/4];
    const uint32_t hi = buf[off/4 + 1];
    return (off & 2) ? __builtin_amdgcn_alignbyte(hi, lo, 2) : lo;
}

// In place on n_grp groups of 128 floats in shared memory, all 128 threads of the block (tid < 128 work):
// x = D(sb) * H/sqrt(128) * D(sa) * x.
static __device__ __forceinline__ void fattn_dt3_wht(float * x, const int n_grp, const float * sa, const float * sb, const int tid) {
    if (tid < QK_TURBO3) {
        for (int g = 0; g < n_grp; ++g) {
            x[g*QK_TURBO3 + tid] *= sa[tid];
        }
    }
    __syncthreads();
#pragma unroll
    for (int h = 1; h < QK_TURBO3; h <<= 1) {
        if (tid < QK_TURBO3 && (tid & h) == 0) {
            for (int g = 0; g < n_grp; ++g) {
                const float a = x[g*QK_TURBO3 + tid];
                const float b = x[g*QK_TURBO3 + tid + h];
                x[g*QK_TURBO3 + tid]     = a + b;
                x[g*QK_TURBO3 + tid + h] = a - b;
            }
        }
        __syncthreads();
    }
    if (tid < QK_TURBO3) {
        for (int g = 0; g < n_grp; ++g) {
            x[g*QK_TURBO3 + tid] *= 0.08838834764831845f*sb[tid];
        }
    }
    __syncthreads();
}

#ifndef PRISM_FA_DT3_WAVES_PER_EU
#define PRISM_FA_DT3_WAVES_PER_EU 4
#endif

template <int G>
#ifdef GGML_USE_HIP
__attribute__((amdgpu_flat_work_group_size(fattn_dt3_nwarps*WARP_SIZE, fattn_dt3_nwarps*WARP_SIZE), amdgpu_waves_per_eu(PRISM_FA_DT3_WAVES_PER_EU)))
#else
__launch_bounds__(fattn_dt3_nwarps*WARP_SIZE, 1)
#endif // GGML_USE_HIP
static __global__ void flash_attn_ext_decode_turbo3_0(
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

    constexpr int D      = fattn_dt3_D;
    constexpr int nwarps = fattn_dt3_nwarps;
    constexpr int R      = fattn_dt3_rows;
    constexpr int RW     = fattn_dt3_row_w;
    static_assert(G <= 8, "too many Q heads per K/V head");
    static_assert(nwarps*WARP_SIZE == QK_TURBO3, "the rotations use one thread per value of a group");

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

    // Per-warp K/V staging (+1 dword: the 2-byte aligned reads of the last row stay in bounds); the rotated Q and
    // after the main loop the merged output reuse the same memory.
    constexpr int ne_stage = nwarps*2*(fattn_dt3_tile_w + 1) > G*D ? nwarps*2*(fattn_dt3_tile_w + 1) : G*D;
    __shared__ uint32_t KV_s_raw[ne_stage];
    __shared__ float    P_s[nwarps][R][G];
    __shared__ float2   ml_s[nwarps][G];
    __shared__ fattn_dt3_h2 lutK[64];
    __shared__ __align__(16) fattn_dt3_h2 Q_s[G][D/2]; // rotated Q in f16 pairs
    __shared__ float2       lutV[64];
    float (*X_s)[D] = (float (*)[D]) KV_s_raw;

    // pair tables: index = lo2(a) | lo2(b) << 2 | hi(a) << 4 | hi(b) << 5
    if (tid < 64) {
        const int ia = (tid & 3)        | ((tid >> 4) & 1) << 2;
        const int ib = ((tid >> 2) & 3) | ((tid >> 5) & 1) << 2;
        lutK[tid] = fattn_dt3_h2{(_Float16) turbo3_centroids[ia], (_Float16) turbo3_centroids[ib]};
        lutV[tid] = make_float2(turbo3_centroids[ia], turbo3_centroids[ib]);
    }

    // Q * scale * log2(e), rotated per 128-value group. The softmax below works in base 2.
    const float scale_l2 = scale*1.4426950408889634f;
    for (int t = tid; t < G*D; t += nwarps*WARP_SIZE) {
        const int h = t / D;
        const int d = t - h*D;
        X_s[h][d] = ((const float *) (Q + int64_t(nb02)*h))[d]*scale_l2;
    }
    __syncthreads();
    fattn_dt3_wht(&X_s[0][0], G*D/QK_TURBO3, turbo3_wht_s1, turbo3_wht_s2, tid);

    // QK role: lane handles values 16*l16 .. 16*l16 + 15 of row 2*p + kr; they are in turbo3 block l16/8.
    const int l16 = lane % 16;
    const int kr  = lane / 16;
    for (int t = tid; t < G*D/2; t += nwarps*WARP_SIZE) {
        const int h = t / (D/2);
        const int i = t - h*(D/2);
        Q_s[h][i] = fattn_dt3_h2{(_Float16) X_s[h][2*i], (_Float16) X_s[h][2*i + 1]};
    }
    const int kblk = l16 / 8;          // turbo3 block of the lane's values
    const int ksub = l16 % 8;          // 16-value slice within the block
    const int k_off_norm  = kblk*50;
    const int k_off_qs    = kblk*50 + 2  + 4*ksub;
    const int k_off_signs = kblk*50 + 34 + 2*ksub;

    // PV role: lane handles dims 8*lane .. 8*lane + 7, in block lane/16 at offset 8*(lane%16).
    const int vblk = lane / 16;
    const int vsub = lane % 16;
    const int v_off_norm  = vblk*50;
    const int v_off_qs    = vblk*50 + 2  + 2*vsub;
    const int v_off_signs = vblk*50 + 34 + vsub;
    float acc[G][8];
#pragma unroll
    for (int h = 0; h < G; ++h) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            acc[h][i] = 0.0f;
        }
    }

    // Softmax role: after the reduce-scatter, lane holds the score of head hs = lane % 8 (heads >= G are padding)
    // of row 2*p + kr; lanes l and l + 8 hold the same values.
    const int hs = lane % 8;
    float KQ_max = -FLT_MAX/2.0f;
    float KQ_sum = 0.0f;

    __syncthreads(); // X_s (the rotated Q) is no longer needed, the staging buffers take its place

    uint32_t * K_s = KV_s_raw + (2*warp + 0)*(fattn_dt3_tile_w + 1);
    uint32_t * V_s = KV_s_raw + (2*warp + 1)*(fattn_dt3_tile_w + 1);

    const int k_VKQ_max = KV_max ? KV_max[sequence*gridDim.x + blockIdx.x] : ne11;

    uint32_t pk[fattn_dt3_nld];
    uint32_t pv[fattn_dt3_nld];
    auto load_tile = [&](const int row0_load) {
#pragma unroll
        for (int j = 0; j < fattn_dt3_nld; ++j) {
            const int i = j*WARP_SIZE + lane;
            if (i < fattn_dt3_tile_w) {
                const int r = i / RW;
                const int c = i - r*RW;
                pk[j] = ((const uint32_t *) (K + int64_t(row0_load + r)*nb11))[c];
                pv[j] = ((const uint32_t *) (V + int64_t(row0_load + r)*nb21))[c];
            }
        }
    };

    int t0 = blockIdx.y*fattn_dt3_tile;
    if (t0 < k_VKQ_max) {
        load_tile(t0 + warp*R);
    }
    for (; t0 < k_VKQ_max; t0 += gridDim.y*fattn_dt3_tile) {
        const int row0 = t0 + warp*R;

        fattn_dt3_wave_sync(); // the previous tile's reads are done
#pragma unroll
        for (int j = 0; j < fattn_dt3_nld; ++j) {
            const int i = j*WARP_SIZE + lane;
            if (i < fattn_dt3_tile_w) {
                K_s[i] = pk[j];
                V_s[i] = pv[j];
            }
        }
        fattn_dt3_wave_sync();
        if (t0 + gridDim.y*fattn_dt3_tile < k_VKQ_max) {
            load_tile(row0 + gridDim.y*fattn_dt3_tile);
        }

        // S = Q K^T for the warp's R rows, 2 rows at a time.
        float S[R/2];
#pragma unroll
        for (int p = 0; p < R/2; ++p) {
            const int rr   = 2*p + kr;
            const int base = rr*fattn_dt3_row_b;
            const float    nK = __half2float(__ushort_as_half((unsigned short) (fattn_dt3_load_u32_a2(K_s, base + k_off_norm) & 0xFFFF)));
            const uint32_t qs = fattn_dt3_load_u32_a2(K_s, base + k_off_qs);
            const uint32_t sg = fattn_dt3_load_u32_a2(K_s, base + k_off_signs) & 0xFFFF;
            fattn_dt3_h2 c[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                c[i] = lutK[((qs >> (4*i)) & 0xF) | (((sg >> (2*i)) & 3) << 4)];
            }
            float v[8];
#pragma unroll
            for (int h = 0; h < 8; ++h) {
                v[h] = 0.0f;
                if (h >= G) {
                    continue;
                }
                const uint4 qa = ((const uint4 *) &Q_s[h][8*l16])[0];
                const uint4 qb = ((const uint4 *) &Q_s[h][8*l16])[1];
                const uint32_t qw[8] = {qa.x, qa.y, qa.z, qa.w, qb.x, qb.y, qb.z, qb.w};
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v[h] = __builtin_amdgcn_fdot2(c[i], __builtin_bit_cast(fattn_dt3_h2, qw[i]), v[h], false);
                }
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                v[h] *= nK;
            }
            // Reduce-scatter over lane bits 4, 2, 1 (heads), then add the other half of the row (lane bit 8).
            const bool b4 = lane & 4;
            float u4[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float keep = b4 ? v[i + 4] : v[i];
                const float send = b4 ? v[i]     : v[i + 4];
                u4[i] = keep + fattn_dt3_xor<4>(send);
            }
            const bool b2 = lane & 2;
            float u2[2];
#pragma unroll
            for (int i = 0; i < 2; ++i) {
                const float keep = b2 ? u4[i + 2] : u4[i];
                const float send = b2 ? u4[i]     : u4[i + 2];
                u2[i] = keep + fattn_dt3_xor<2>(send);
            }
            const bool b1 = lane & 1;
            float sc = (b1 ? u2[1] : u2[0]) + fattn_dt3_xor<1>(b1 ? u2[0] : u2[1]);
            sc += fattn_dt3_xor<8>(sc);
            S[p] = sc + __half2float(maskh[row0 + rr])*1.4426950408889634f;
        }

        // Online softmax over the tile, for head hs.
        float tmax = S[0];
#pragma unroll
        for (int p = 1; p < R/2; ++p) {
            tmax = fmaxf(tmax, S[p]);
        }
        tmax = fmaxf(tmax, fattn_dt3_xor<16>(tmax));
        const float m_new = fmaxf(KQ_max, tmax);
        const float alpha = __builtin_amdgcn_exp2f(KQ_max - m_new);
        KQ_max = m_new;
        float tsum = 0.0f;
#pragma unroll
        for (int p = 0; p < R/2; ++p) {
            S[p] = __builtin_amdgcn_exp2f(S[p] - m_new);
            tsum += S[p];
        }
        tsum += fattn_dt3_xor<16>(tsum);
        KQ_sum = KQ_sum*alpha + tsum;
        if (hs < G && (lane & 8) == 0) {
#pragma unroll
            for (int p = 0; p < R/2; ++p) {
                P_s[warp][2*p + kr][hs] = S[p];
            }
        }
#pragma unroll
        for (int h = 0; h < G; ++h) {
            const float alpha_h = __int_as_float(__builtin_amdgcn_readlane(__float_as_int(alpha), h));
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[h][i] *= alpha_h;
            }
        }
        fattn_dt3_wave_sync();

        // O += P V (rotated domain)
#pragma unroll 4
        for (int rr = 0; rr < R; ++rr) {
            const int base = rr*fattn_dt3_row_b;
            const float    nV = __half2float(__ushort_as_half((unsigned short) (fattn_dt3_load_u32_a2(V_s, base + v_off_norm) & 0xFFFF)));
            const uint32_t qs = fattn_dt3_load_u32_a2(V_s, base + v_off_qs);
            const uint32_t sg = fattn_dt3_load_u32_a2(V_s, base + (v_off_signs & ~1)) >> (8*(v_off_signs & 1));
            float vf[8];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int idx = ((qs >> (4*i)) & 0xF) | (((sg >> (2*i)) & 3) << 4);
                const float2 c = lutV[idx];
                vf[2*i + 0] = c.x;
                vf[2*i + 1] = c.y;
            }
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float w = P_s[warp][rr][h]*nV;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[h][i] += w*vf[i];
                }
            }
        }
    }

    // Merge the warps of the block, as in the q4_0 kernel.
    __syncthreads();
    for (int t = tid; t < G*D; t += nwarps*WARP_SIZE) {
        X_s[0][t] = 0.0f;
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
#pragma unroll
    for (int w = 0; w < nwarps; ++w) {
        if (warp == w) {
#pragma unroll
            for (int h = 0; h < G; ++h) {
                const float f = __builtin_amdgcn_exp2f(ml_s[warp][h].x - M[h]);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    X_s[h][8*lane + i] += acc[h][i]*f;
                }
            }
        }
        __syncthreads();
    }

    // back to the original domain: M^T = D(s1) H/sqrt(128) D(s2)
    fattn_dt3_wht(&X_s[0][0], G*D/QK_TURBO3, turbo3_wht_s2, turbo3_wht_s1, tid);

    for (int i = tid; i < G*D; i += nwarps*WARP_SIZE) {
        const int h   = i / D;
        const int dim = i - h*D;
        float val = X_s[h][dim];
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

bool ggml_cuda_flash_attn_ext_decode_turbo3_supported(const ggml_tensor * dst) {
#ifdef GGML_USE_HIP
    static const bool disabled = getenv("GGML_CUDA_NO_FA_DECODE_TURBO3") != nullptr;
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

    if (Q->type != GGML_TYPE_F32 || K->type != GGML_TYPE_TURBO3_0 || V->type != GGML_TYPE_TURBO3_0) {
        return false;
    }
    if (Q->ne[0] != fattn_dt3_D || K->ne[0] != fattn_dt3_D || V->ne[0] != fattn_dt3_D || Q->ne[1] != 1) {
        return false;
    }
    if (!mask || sinks || max_bias != 0.0f || logit_softcap != 0.0f || mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (K->ne[1] % fattn_dt3_tile != 0 || Q->ne[2] % K->ne[2] != 0 || Q->ne[3] != K->ne[3]) {
        return false;
    }
    const int gqa_ratio = Q->ne[2] / K->ne[2];
    if (gqa_ratio != 4 && gqa_ratio != 6 && gqa_ratio != 8) {
        return false;
    }
    if ((uintptr_t) K->data % 4 != 0 || (uintptr_t) V->data % 4 != 0 || (uintptr_t) Q->data % 16 != 0) {
        return false;
    }
    for (int i = 1; i < GGML_MAX_DIMS; ++i) {
        if (K->nb[i] % 4 != 0 || V->nb[i] % 4 != 0 || Q->nb[i] % 16 != 0) {
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
static void ggml_cuda_flash_attn_ext_decode_turbo3_case(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    fattn_kernel_t fattn_kernel = flash_attn_ext_decode_turbo3_0<G>;
    launch_fattn<fattn_dt3_D, 1, G>(ctx, dst, fattn_kernel, fattn_dt3_nwarps, 0, fattn_dt3_tile, false, false, false);
}

void ggml_cuda_flash_attn_ext_decode_turbo3(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const int gqa_ratio = dst->src[0]->ne[2] / dst->src[1]->ne[2];
    switch (gqa_ratio) {
        case 4: ggml_cuda_flash_attn_ext_decode_turbo3_case<4>(ctx, dst); break;
        case 6: ggml_cuda_flash_attn_ext_decode_turbo3_case<6>(ctx, dst); break;
        case 8: ggml_cuda_flash_attn_ext_decode_turbo3_case<8>(ctx, dst); break;
        default: GGML_ABORT("unsupported gqa ratio");
    }
}

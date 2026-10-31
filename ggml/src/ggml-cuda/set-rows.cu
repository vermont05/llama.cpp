#include "set-rows.cuh"
#include "cpy-utils.cuh"

#include <cinttypes>
#include <cstring>

typedef void (*set_rows_kernel_t)(const char * src, char * dst);

// Generic quantized set_rows kernel template
template <typename idx_t, typename block_type, int qk, void (*quantize_func)(const float *, block_type *)>
static __global__ void k_set_rows_quant(const float * __restrict__ src0,
                                        const idx_t * __restrict__ src1,
                                        block_type * __restrict__ dst,
                                        const int64_t ne_total,
                                        const int64_t ne10,
                                        const int64_t ne11,
                                        const int64_t ne12,
                                        const int64_t ne13,
                                        const int64_t s01,
                                        const int64_t s02,
                                        const int64_t s03,
                                        const int64_t s10,
                                        const int64_t s11,
                                        const int64_t s12,
                                        const int64_t s1,
                                        const int64_t s2,
                                        const int64_t s3,
                                        const uint3   ne00,
                                        const uint3   ne01,
                                        const uint3   ne02,
                                        const uint3   ne11_fd,
                                        const uint3   ne12_fd) {
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    const int64_t i_base = i * qk;
    uint32_t      tmp    = (uint32_t) i_base;
    uint2         div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);

    const float * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    block_type * dst_row_ptr = dst + (dst_row*s1 + i02*s2 + i03*s3) / sizeof(block_type);

    const float * src_block = src0_row + i00;
    block_type * dst_block = dst_row_ptr + i00 / qk;

    quantize_func(src_block, dst_block);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

// Template dispatch function for quantized set_rows
template<typename idx_t, typename block_type, int qk, void (*quantize_func)(const float*, block_type*)>
static void set_rows_cuda_quant(
        const float * src0_d, const idx_t * src1_d, block_type * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    GGML_ASSERT(ne00 % qk == 0);
    const int64_t ne_total = (ne00 * ne01 * ne02 * ne03) / qk;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);

    const int64_t s01 = nb01/sizeof(float);
    const int64_t s02 = nb02/sizeof(float);
    const int64_t s03 = nb03/sizeof(float);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1;
    const int64_t s2  = nb2;
    const int64_t s3  = nb3;

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        k_set_rows_quant<idx_t, block_type, qk, quantize_func><<<grid_size, block_size, 0, stream>>>(
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01, s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd,
            ne01_fd, ne02_fd, ne11_fd, ne12_fd);
    }
}

template <typename src_t, typename idx_t, typename dst_t>
static __global__ void k_set_rows(const src_t * src0_ptr,
                                  const idx_t * src1_ptr,
                                  dst_t * dst_ptr,
                                  const int64_t ne_total,
                                  const int64_t ne10,
                                  const int64_t ne11,
                                  const int64_t ne12,
                                  const int64_t ne13,
                                  const int64_t s01,
                                  const int64_t s02,
                                  const int64_t s03,
                                  const int64_t s10,
                                  const int64_t s11,
                                  const int64_t s12,
                                  const int64_t s1,
                                  const int64_t s2,
                                  const int64_t s3,
                                  const uint3   ne00,
                                  const uint3   ne01,
                                  const uint3   ne02,
                                  const uint3   ne11_fd,
                                  const uint3   ne12_fd) {
    const src_t * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const idx_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t       * GGML_CUDA_RESTRICT dst  = dst_ptr;
    const int64_t i = int64_t(blockDim.x) * blockIdx.x + threadIdx.x;

    if (i >= ne_total) {
        return;
    }

    uint32_t tmp = (uint32_t) i;
    uint2    div_mod;

    div_mod           = fast_div_modulo(tmp, ne00);
    const int64_t i00 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne01);
    const int64_t i01 = div_mod.y;
    tmp               = div_mod.x;

    div_mod           = fast_div_modulo(tmp, ne02);
    const int64_t i02 = div_mod.y;
    const int64_t i03 = div_mod.x;

    const int64_t i12 = fastmodulo((uint32_t) i03, ne12_fd);
    const int64_t i11 = fastmodulo((uint32_t) i02, ne11_fd);
    const int64_t i10 = i01;

    ggml_cuda_pdl_sync();
    const int64_t dst_row = *(src1 + i10*s10 + i11*s11 + i12*s12);
    ggml_cuda_pdl_lc();

    const src_t * src0_row = src0 + i01*s01 + i02*s02 + i03*s03;
    dst_t * dst_row_ptr    = dst + dst_row*s1 + i02*s2 + i03*s3;

    dst_row_ptr[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);

    GGML_UNUSED(ne10);
    GGML_UNUSED(ne11);
    GGML_UNUSED(ne12);
    GGML_UNUSED(ne13);
}

template<typename src_t, typename idx_t, typename dst_t>
static void set_rows_cuda(
        const src_t * src0_d, const idx_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const int64_t ne01, const int64_t ne02, const int64_t ne03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const int64_t ne13,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {

    const int64_t ne_total = ne00 * ne01 * ne02 * ne03;
    const int num_blocks = (ne_total + CUDA_SET_ROWS_BLOCK_SIZE - 1) / CUDA_SET_ROWS_BLOCK_SIZE;
    const dim3 block_size(CUDA_SET_ROWS_BLOCK_SIZE);
    const dim3 grid_size(num_blocks);


    const int64_t s01 = nb01/sizeof(src_t);
    const int64_t s02 = nb02/sizeof(src_t);
    const int64_t s03 = nb03/sizeof(src_t);
    const int64_t s10 = nb10/sizeof(idx_t);
    const int64_t s11 = nb11/sizeof(idx_t);
    const int64_t s12 = nb12/sizeof(idx_t);
    const int64_t s1  = nb1/sizeof(dst_t);
    const int64_t s2  = nb2/sizeof(dst_t);
    const int64_t s3  = nb3/sizeof(dst_t);

    if (ne_total > 0 && ne00 > 0 && ne01 > 0 && ne02 > 0 && ne11 > 0 && ne12 > 0) {
        const uint3 ne00_fd = init_fastdiv_values((uint32_t) ne00);
        const uint3 ne01_fd = init_fastdiv_values((uint32_t) ne01);
        const uint3 ne02_fd = init_fastdiv_values((uint32_t) ne02);
        const uint3 ne11_fd = init_fastdiv_values((uint32_t) ne11);
        const uint3 ne12_fd = init_fastdiv_values((uint32_t) ne12);

        const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(grid_size, block_size, 0, stream);
        ggml_cuda_kernel_launch(k_set_rows<src_t, idx_t, dst_t>, launch_params,
            src0_d, src1_d, dst_d, ne_total, ne10, ne11, ne12, ne13, s01,
            s02, s03, s10, s11, s12, s1, s2, s3, ne00_fd, ne01_fd, ne02_fd,
            ne11_fd, ne12_fd);
    }
}

template<typename src_t, typename idx_t>
static void set_rows_cuda(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const src_t * src0_d = (const src_t *)src0->data;
    const idx_t * src1_d = (const idx_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F32) {
        set_rows_cuda(
            src0_d, src1_d, (float*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_BF16) {
        set_rows_cuda(
            src0_d, src1_d, (nv_bfloat16*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_0) {
        set_rows_cuda_quant<idx_t, block_q4_0, QK4_0, quantize_f32_q4_0_block>(
            src0_d, src1_d, (block_q4_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q4_1) {
        set_rows_cuda_quant<idx_t, block_q4_1, QK4_1, quantize_f32_q4_1_block>(
            src0_d, src1_d, (block_q4_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_0) {
        set_rows_cuda_quant<idx_t, block_q5_0, QK5_0, quantize_f32_q5_0_block>(
            src0_d, src1_d, (block_q5_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q5_1) {
        set_rows_cuda_quant<idx_t, block_q5_1, QK5_1, quantize_f32_q5_1_block>(
            src0_d, src1_d, (block_q5_1*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_Q8_0) {
        set_rows_cuda_quant<idx_t, block_q8_0, QK8_0, quantize_f32_q8_0_block>(
            src0_d, src1_d, (block_q8_0*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else if (dst->type == GGML_TYPE_IQ4_NL) {
        set_rows_cuda_quant<idx_t, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl_block>(
            src0_d, src1_d, (block_iq4_nl*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int32_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int32_t * src1_d = (const int32_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}

template<>
void set_rows_cuda<half, int64_t>(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const half    * src0_d = (const half *)src0->data;
    const int64_t * src1_d = (const int64_t *)src1->data;

    GGML_TENSOR_BINARY_OP_LOCALS

    cudaStream_t stream = ctx.stream();


    if (dst->type == GGML_TYPE_F16) {
        set_rows_cuda(
            src0_d, src1_d, (half*)dst->data,
            ne00, ne01, ne02, ne03,
            ne10, ne11, ne12, ne13,
            nb01, nb02, nb03,
            nb10, nb11, nb12,
            nb1, nb2, nb3,
            stream
        );
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(dst->type));
    }
}



// ---- KV cache quantization emulation (see set-rows.cuh) ----

enum kv_emu_mode { KV_EMU_NONE = 0, KV_EMU_Q4_0 = 1, KV_EMU_TURBO3 = 2 };

static kv_emu_mode kv_emu_parse(const char * name) {
    const char * v = getenv(name);
    if (v == nullptr || v[0] == 0 || strcmp(v, "none") == 0) {
        return KV_EMU_NONE;
    }
    if (strcmp(v, "q4_0") == 0) {
        return KV_EMU_Q4_0;
    }
    if (strcmp(v, "turbo3") == 0) {
        return KV_EMU_TURBO3;
    }
    GGML_ABORT("%s: unknown mode '%s' (q4_0, turbo3, none)", name, v);
}

static kv_emu_mode kv_emu_mode_k() { static const kv_emu_mode m = kv_emu_parse("PRISM_KV_EMU_K"); return m; }
static kv_emu_mode kv_emu_mode_v() { static const kv_emu_mode m = kv_emu_parse("PRISM_KV_EMU_V"); return m; }

bool ggml_cuda_kv_emu_active() {
    return kv_emu_mode_k() != KV_EMU_NONE || kv_emu_mode_v() != KV_EMU_NONE;
}

// sign diagonals of the WHT, seed 42 (TURBO_WHT_SIGNS1/2 of the fork, = turbo_cpu_s1/s2)
static __constant__ float kv_emu_s1[128] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1
};
static __constant__ float kv_emu_s2[128] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1
};
// 3-bit Lloyd-Max centroids for N(0, 1/128)
static __constant__ float kv_emu_c3[8] = {
    -0.190207f, -0.118786f, -0.066822f, -0.021663f, 0.021663f, 0.066822f, 0.118786f, 0.190207f
};

static __device__ int kv_emu_nearest3(const float v) {
    if (v < -0.154496f) return 0;
    if (v < -0.092804f) return 1;
    if (v < -0.044243f) return 2;
    if (v <  0.000000f) return 3;
    if (v <  0.044243f) return 4;
    if (v <  0.092804f) return 5;
    if (v <  0.154496f) return 6;
    return 7;
}

// sum of v over the 128 threads of the block
static __device__ float kv_emu_sum128(float v, float * red) {
    const int j = threadIdx.x;
    red[j] = v;
    __syncthreads();
    for (int h = 64; h > 0; h >>= 1) {
        if (j < h) {
            red[j] += red[j + h];
        }
        __syncthreads();
    }
    const float res = red[0];
    __syncthreads();
    return res;
}

// x = D(sb) * H/sqrt(128) * D(sa) * x for the 128 values in x (in place, all threads)
static __device__ void kv_emu_wht128(float * x, const float * sa, const float * sb) {
    const int j = threadIdx.x;
    x[j] *= sa[j];
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
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

// one block of 128 threads per 128-value group of a row; dst is contiguous
static __global__ void k_kv_emu(const char * __restrict__ src, float * __restrict__ dst, const int mode,
        const int64_t ne00, const int64_t ne01, const int64_t ne02,
        const int64_t nb01, const int64_t nb02, const int64_t nb03) {
    const int     j     = threadIdx.x;
    const int64_t n_grp = ne00 / 128;
    int64_t       r     = blockIdx.x;
    const int64_t i_grp = r % n_grp; r /= n_grp;
    const int64_t i01   = r % ne01;  r /= ne01;
    const int64_t i02   = r % ne02;
    const int64_t i03   = r / ne02;

    const float * s = (const float *) (src + i01*nb01 + i02*nb02 + i03*nb03) + i_grp*128;
    float       * d = dst + ((i03*ne02 + i02)*ne01 + i01)*ne00 + i_grp*128;

    __shared__ float x[128];
    __shared__ float red[128];

    x[j] = s[j];
    __syncthreads();

    if (mode == KV_EMU_Q4_0) {
        // quantize_f32_q4_0_block and back, one thread per 32-value block
        if (j < 4) {
            block_q4_0 q;
            quantize_f32_q4_0_block(x + j*QK4_0, &q);
            const float qd = __half2float(q.d);
            for (int k = 0; k < QK4_0/2; ++k) {
                x[j*QK4_0 + k]           = ((q.qs[k] & 0x0F) - 8)*qd;
                x[j*QK4_0 + k + QK4_0/2] = ((q.qs[k] >>   4) - 8)*qd;
            }
        }
        __syncthreads();
        d[j] = x[j];
        return;
    }

    // turbo3: normalize, rotate, 3-bit centroids, corrected norm (f16), rotate back
    const float norm     = sqrtf(kv_emu_sum128(x[j]*x[j], red));
    const float inv_norm = norm > 1e-10f ? 1.0f/norm : 0.0f;
    x[j] *= inv_norm;
    __syncthreads();

    kv_emu_wht128(x, kv_emu_s1, kv_emu_s2);

    const float c        = kv_emu_c3[kv_emu_nearest3(x[j])];
    const float rnorm    = sqrtf(kv_emu_sum128(c*c, red));
    const float corr     = __half2float(__float2half(rnorm > 1e-10f ? norm/rnorm : norm));
    x[j] = c*corr;
    __syncthreads();

    // inverse: D(s1) * H/sqrt(128) * D(s2)
    kv_emu_wht128(x, kv_emu_s2, kv_emu_s1);

    d[j] = x[j];
}

// true if the write went through the emulation
static bool kv_emu_set_rows(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    if (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16 || !ggml_cuda_kv_emu_active()) {
        return false;
    }
    const ggml_tensor * base = dst;
    while (base->view_src) {
        base = base->view_src;
    }
    const bool is_k = strncmp(base->name, "cache_k", 7) == 0;
    const bool is_v = strncmp(base->name, "cache_v", 7) == 0;
    const kv_emu_mode mode = is_k ? kv_emu_mode_k() : is_v ? kv_emu_mode_v() : KV_EMU_NONE;
    if (mode == KV_EMU_NONE) {
        return false;
    }
    // rows of whole heads (FA layout, V not transposed); 128-value groups never straddle a head of 128 or 256
    GGML_ASSERT(src0->ne[0] % 128 == 0 && src0->nb[0] == sizeof(float) && "KV emulation needs rows of 128k values (flash attention)");

    static bool logged[2] = { false, false };
    if (!logged[is_v]) {
        logged[is_v] = true;
        GGML_LOG_WARN("%s: emulating %s for %s (row of %" PRId64 " values)\n", __func__,
                mode == KV_EMU_Q4_0 ? "q4_0" : "turbo3", is_k ? "K" : "V", src0->ne[0]);
    }

    const int64_t n = ggml_nelements(src0);
    ggml_cuda_pool_alloc<float> tmp(ctx.pool(), n);

    const int64_t n_blocks = n / 128;
    k_kv_emu<<<n_blocks, 128, 0, ctx.stream()>>>((const char *) src0->data, tmp.get(), mode,
            src0->ne[0], src0->ne[1], src0->ne[2], src0->nb[1], src0->nb[2], src0->nb[3]);
    CUDA_CHECK(cudaGetLastError());

    ggml_tensor src0c = *src0;
    src0c.data  = tmp.get();
    src0c.nb[0] = sizeof(float);
    src0c.nb[1] = src0c.nb[0]*src0->ne[0];
    src0c.nb[2] = src0c.nb[1]*src0->ne[1];
    src0c.nb[3] = src0c.nb[2]*src0->ne[2];

    if (src1->type == GGML_TYPE_I64) {
        set_rows_cuda<float, int64_t>(ctx, &src0c, src1, dst);
    } else {
        set_rows_cuda<float, int32_t>(ctx, &src0c, src1, dst);
    }
    return true;
}

void ggml_cuda_op_set_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || (src0->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16));
    GGML_ASSERT(src1->type == GGML_TYPE_I64 || src1->type == GGML_TYPE_I32);

    if (kv_emu_set_rows(ctx, src0, src1, dst)) {
        return;
    }

    if (src0->type == GGML_TYPE_F32) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<float, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<float, int32_t>(ctx, src0, src1, dst);
        }
    } else if (src0->type == GGML_TYPE_F16) {
        if (src1->type == GGML_TYPE_I64) {
            set_rows_cuda<half, int64_t>(ctx, src0, src1, dst);
        } else {
            set_rows_cuda<half, int32_t>(ctx, src0, src1, dst);
        }
    } else {
        GGML_ABORT("unsupported type %s", ggml_type_name(src0->type));
    }
}

// Two single-row SET_ROWS into Q4_0 caches (the K and V cache writes of one attention layer) as one launch.
// Each thread quantizes one 32-value block with quantize_f32_q4_0_block, as k_set_rows_quant does.
// GGML_CUDA_NO_SET_ROWS_PAIR=1 disables it.
template <typename idx_t>
static __global__ void k_set_rows_q4_0_pair(
        const float * __restrict__ x0, const idx_t * __restrict__ idx0, char * __restrict__ dst0, const int64_t nb01, const int nblk0,
        const float * __restrict__ x1, const idx_t * __restrict__ idx1, char * __restrict__ dst1, const int64_t nb11, const int nblk1) {
    // One code path for both tensors (pick the pointers first): with a branch per tensor the compiler contracted
    // x*id + 8.5 into an fma for only some of the values, which rounds differently from k_set_rows_quant.
    const int nblocks0 = (nblk0 + blockDim.x - 1) / blockDim.x;
    const bool second  = (int) blockIdx.x >= nblocks0;
    const int  i       = (second ? blockIdx.x - nblocks0 : blockIdx.x)*blockDim.x + threadIdx.x;
    const float * x    = second ? x1 : x0;
    const idx_t * idx  = second ? idx1 : idx0;
    char        * dst  = second ? dst1 : dst0;
    const int64_t nb1  = second ? nb11 : nb01;
    const int     nblk = second ? nblk1 : nblk0;
    if (i >= nblk) {
        return;
    }
    block_q4_0 * row = (block_q4_0 *) (dst + int64_t(idx[0])*nb1);
    quantize_f32_q4_0_block(x + i*QK4_0, row + i);
}

bool ggml_cuda_op_set_rows_q4_0_pair(ggml_backend_cuda_context & ctx, ggml_tensor * a, ggml_tensor * b) {
    static const bool disabled = getenv("GGML_CUDA_NO_SET_ROWS_PAIR") != nullptr;
    auto ok = [](const ggml_tensor * t) {
        const ggml_tensor * x   = t->src[0];
        const ggml_tensor * idx = t->src[1];
        return t->type == GGML_TYPE_Q4_0 && x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[1] == 1 &&
            x->ne[2] == 1 && x->ne[3] == 1 && x->ne[0] == t->ne[0] && x->ne[0] % QK4_0 == 0 &&
            (idx->type == GGML_TYPE_I64 || idx->type == GGML_TYPE_I32) && ggml_nelements(idx) == 1 &&
            t->ne[2] == 1 && t->ne[3] == 1;
    };
    if (disabled || !ok(a) || !ok(b) || a->src[1]->type != b->src[1]->type) {
        return false;
    }
    const int nblk0 = a->ne[0] / QK4_0;
    const int nblk1 = b->ne[0] / QK4_0;
    const int nthreads = 32;
    const int nblocks  = (nblk0 + nthreads - 1) / nthreads + (nblk1 + nthreads - 1) / nthreads;
    if (a->src[1]->type == GGML_TYPE_I64) {
        k_set_rows_q4_0_pair<int64_t><<<nblocks, nthreads, 0, ctx.stream()>>>(
            (const float *) a->src[0]->data, (const int64_t *) a->src[1]->data, (char *) a->data, a->nb[1], nblk0,
            (const float *) b->src[0]->data, (const int64_t *) b->src[1]->data, (char *) b->data, b->nb[1], nblk1);
    } else {
        k_set_rows_q4_0_pair<int32_t><<<nblocks, nthreads, 0, ctx.stream()>>>(
            (const float *) a->src[0]->data, (const int32_t *) a->src[1]->data, (char *) a->data, a->nb[1], nblk0,
            (const float *) b->src[0]->data, (const int32_t *) b->src[1]->data, (char *) b->data, b->nb[1], nblk1);
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

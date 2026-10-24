#include "common.cuh"
#include "unary.cuh"
#include "ssm-conv-step.cuh"

// Single-token decode of the gated delta net input convolution as one kernel. The graph does
//   conv_states = get_rows(cache, ids)          [3*C]
//   conv_input  = concat(conv_states, x^T)      [4, C]
//   cache[row]  = conv_input[1..3]              (state update)
//   y           = silu(ssm_conv(conv_input, w)) [C]
//   qk          = l2_norm(y[0 .. n_l2))         [128, n_l2/128]
// which is five launches per layer. Here each thread handles one channel; a block of 128 channels is one l2 head.
// The arithmetic matches ssm_conv_f32 and l2_norm_f32<WARP_SIZE> exactly. GGML_CUDA_NO_CONV_STEP=1 disables it.

static constexpr int conv_step_head = 128;

static __global__ void ssm_conv_step_f32(
        const float * __restrict__ state_src, const int32_t * __restrict__ ids, const int64_t state_src_row,
        const float * __restrict__ x, const float * __restrict__ w, const int w_stride,
        float * state_dst, float * __restrict__ silu_dst, float * __restrict__ l2_dst, const int n_l2, const float eps,
        const int ncols_l2) {
    const int tid = threadIdx.x;
    const int c   = blockIdx.x*conv_step_head + tid;

    const float * s = state_src + int64_t(ids[0])*state_src_row + 3*c;
    const float x0 = s[0];
    const float x1 = s[1];
    const float x2 = s[2];
    const float x3 = x[c];

    float wv[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        wv[j] = w[c*w_stride + j];
    }
    const float b = 0.0f;

    float sumf = 0.0f;
    sumf += x0*wv[0];
    sumf += x1*wv[1];
    sumf += x2*wv[2];
    sumf += x3*wv[3];
    sumf += b;
    const float y = ggml_cuda_op_silu_single(sumf);

    // The state row may be the one just read (in place); each thread reads its channel before writing it.
    state_dst[3*c + 0] = x1;
    state_dst[3*c + 1] = x2;
    state_dst[3*c + 2] = x3;
    silu_dst[c] = y;

    if (blockIdx.x*conv_step_head < n_l2) {
        // Same code shape as l2_norm_f32<WARP_SIZE> (one warp per row, runtime column loop), so the rounding matches.
        __shared__ float ys[conv_step_head];
        ys[tid] = y;
        __syncthreads();
        if (tid < WARP_SIZE) {
            float tmp = 0.0f;
            for (int col = tid; col < ncols_l2; col += WARP_SIZE) {
                const float xi = ys[col];
                tmp += xi * xi;
            }
            tmp = block_reduce<block_reduce_method::SUM, WARP_SIZE>(tmp, (float *) nullptr);
            const float scale = rsqrtf(fmaxf(tmp, eps * eps));
            for (int col = tid; col < ncols_l2; col += WARP_SIZE) {
                l2_dst[blockIdx.x*conv_step_head + col] = scale * ys[col];
            }
        }
    }
}

// Debug: GGML_CUDA_CONV_STEP_CHECK=1 runs the fused kernel into scratch buffers, lets the original ops run, and then
// compares the three outputs bit for bit (counts of differing values and the max abs difference, per graph).
static __global__ void conv_step_compare(const float * a, const float * b, const int n, unsigned int * stats) {
    const int k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < n && __float_as_uint(a[k]) != __float_as_uint(b[k])) {
        atomicAdd(&stats[0], 1u);
        atomicMax(&stats[1], __float_as_uint(fabsf(a[k] - b[k])));
    }
}

struct conv_step_check {
    const ggml_tensor * l2 = nullptr;
    int l2_idx = -1;
    const float * real[3] = { nullptr, nullptr, nullptr };
    float * tmp[3] = { nullptr, nullptr, nullptr };
    int n[3] = { 0, 0, 0 };
};
static conv_step_check conv_step_pending;
static float * conv_step_scratch = nullptr;
static unsigned int * conv_step_stats = nullptr; // [3 outputs][count, max]

void ggml_cuda_conv_step_check_node(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    // The view nodes right after the L2_NORM never reach the fusion hook, so compare at the first node past it.
    if (conv_step_pending.l2 && i > conv_step_pending.l2_idx) {
        for (int k = 0; k < 3; ++k) {
            conv_step_compare<<<(conv_step_pending.n[k] + 255)/256, 256, 0, ctx.stream()>>>(
                conv_step_pending.tmp[k], conv_step_pending.real[k], conv_step_pending.n[k], conv_step_stats + 2*k);
        }
        conv_step_pending.l2 = nullptr;
    }
}

void ggml_cuda_conv_step_check_report(ggml_backend_cuda_context & ctx) {
    if (!conv_step_stats) {
        return;
    }
    unsigned int h[6];
    CUDA_CHECK(cudaMemcpyAsync(h, conv_step_stats, sizeof(h), cudaMemcpyDeviceToHost, ctx.stream()));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    float mx[3];
    memcpy(mx, h + 1, 4); memcpy(mx + 1, h + 3, 4); memcpy(mx + 2, h + 5, 4);
    GGML_LOG_INFO("conv step check: differing values state %u (max %g), silu %u (max %g), l2 %u (max %g)\n",
        h[0], mx[0], h[2], mx[1], h[4], mx[2]);
    CUDA_CHECK(cudaMemsetAsync(conv_step_stats, 0, sizeof(h), ctx.stream()));
}

static bool conv_step_is_empty(const ggml_tensor * t) {
    return ggml_nelements(t) == 0;
}

int ggml_cuda_try_fuse_conv_step(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_CONV_STEP") != nullptr;
    if (disabled) {
        return 0;
    }
    const ggml_tensor * g = cgraph->nodes[i];
    if (g->op != GGML_OP_GET_ROWS || g->type != GGML_TYPE_F32 || g->src[0]->type != GGML_TYPE_F32 ||
            g->ne[1] != 1 || g->ne[2] != 1 || g->ne[3] != 1 || g->src[1]->type != GGML_TYPE_I32 || g->src[1]->ne[0] != 1 ||
            g->src[0]->nb[0] != sizeof(float)) {
        return 0;
    }

    // Walk the expected op sequence, skipping empty nodes (the copies for other sequences).
    const ggml_op expected[] = {
        GGML_OP_RESHAPE, GGML_OP_TRANSPOSE, GGML_OP_CONCAT, GGML_OP_VIEW, GGML_OP_VIEW, GGML_OP_CPY,
        GGML_OP_SSM_CONV, GGML_OP_UNARY, GGML_OP_VIEW, GGML_OP_L2_NORM,
    };
    constexpr int n_expected = sizeof(expected)/sizeof(expected[0]);
    const ggml_tensor * t[n_expected];
    int j = i + 1;
    for (int k = 0; k < n_expected; ++k) {
        while (j < cgraph->n_nodes && conv_step_is_empty(cgraph->nodes[j])) {
            ++j;
        }
        if (j >= cgraph->n_nodes || cgraph->nodes[j]->op != expected[k]) {
            return 0;
        }
        t[k] = cgraph->nodes[j++];
    }
    const ggml_tensor * resh = t[0];
    const ggml_tensor * tr   = t[1];
    const ggml_tensor * cat  = t[2];
    const ggml_tensor * last = t[3];
    const ggml_tensor * upd  = t[4];
    const ggml_tensor * cpy  = t[5];
    const ggml_tensor * conv = t[6];
    const ggml_tensor * silu = t[7];
    const ggml_tensor * vqk  = t[8];
    const ggml_tensor * l2   = t[9];

    const ggml_tensor * x = tr->src[0];
    const ggml_tensor * w = conv->src[1];
    const int64_t C = cat->ne[1];

    const bool ok =
        resh->src[0] == g && cat->src[0] == resh && cat->src[1] == tr && ggml_get_op_params_i32(cat, 0) == 0 &&
        cat->ne[0] == 4 && resh->ne[0] == 3 && resh->ne[1] == C && cat->ne[2] == 1 && cat->ne[3] == 1 &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && ggml_nelements(x) == C &&
        last->src[0] == cat && last->view_offs == sizeof(float) && last->ne[0] == 3 && last->ne[1] == C &&
        cpy->src[0] == last && cpy->src[1] == upd && upd->type == GGML_TYPE_F32 && ggml_is_contiguous(upd) &&
        ggml_nelements(upd) == 3*C &&
        conv->src[0] == cat && w->type == GGML_TYPE_F32 && w->ne[0] == 4 && w->ne[1] == C && w->nb[0] == sizeof(float) &&
        conv->src[2] == nullptr && conv->ne[0] == C && conv->ne[1] == 1 &&
        ggml_get_unary_op(silu) == GGML_UNARY_OP_SILU && silu->src[0] == conv && ggml_is_contiguous(silu) &&
        vqk->src[0] == silu && vqk->view_offs == 0 && vqk->ne[0] == conv_step_head && vqk->nb[1] == conv_step_head*sizeof(float) &&
        vqk->ne[2] == 1 && vqk->ne[3] == 1 &&
        l2->src[0] == vqk && ggml_is_contiguous(l2) &&
        C % conv_step_head == 0 && vqk->ne[1]*conv_step_head <= C;
    if (!ok) {
        return 0;
    }

    // The graph allocator may place the outputs in memory freed by the earlier nodes of the chain (e.g. the qkv
    // projection). The separate ops are fine with that; one kernel is not, as other blocks may still be reading.
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    // silu exactly in place of x is fine: each thread reads x[c] before it writes silu[c].
    for (const ggml_tensor * out : {silu, l2}) {
        for (const ggml_tensor * in : {x, w, (const ggml_tensor *) g->src[0], (const ggml_tensor *) g->src[1]}) {
            if (out == silu && in == x && silu->data == x->data && ggml_nbytes(silu) == ggml_nbytes(x)) {
                continue;
            }
            if (overlaps(out, in)) {
                static int n_logged = 0;
                if (n_logged++ < 4) {
                    GGML_LOG_INFO("%s: %s [%p, %zu B] overlaps %s [%p, %zu B], not fusing\n", __func__, out->name, out->data,
                        ggml_nbytes(out), in->name, in->data, ggml_nbytes(in));
                }
                return 0;
            }
        }
    }
    if (overlaps(silu, l2) || overlaps(upd, x) || overlaps(upd, silu) || overlaps(upd, l2)) {
        return 0;
    }

    float eps;
    memcpy(&eps, l2->op_params, sizeof(float));

    const int n_l2 = vqk->ne[1]*conv_step_head;
    static const bool check = getenv("GGML_CUDA_CONV_STEP_CHECK") != nullptr;
    if (check) {
        if (!conv_step_scratch) {
            CUDA_CHECK(cudaMalloc(&conv_step_scratch, 5*C*sizeof(float)));
            CUDA_CHECK(cudaMalloc(&conv_step_stats, 6*sizeof(unsigned int)));
            CUDA_CHECK(cudaMemset(conv_step_stats, 0, 6*sizeof(unsigned int)));
        }
        conv_step_pending.l2 = l2;
        for (int k = i + 1; k < cgraph->n_nodes; ++k) {
            if (cgraph->nodes[k] == l2) {
                conv_step_pending.l2_idx = k;
                break;
            }
        }
        conv_step_pending.tmp[0] = conv_step_scratch;       conv_step_pending.real[0] = (const float *) upd->data;  conv_step_pending.n[0] = 3*C;
        conv_step_pending.tmp[1] = conv_step_scratch + 3*C; conv_step_pending.real[1] = (const float *) silu->data; conv_step_pending.n[1] = C;
        conv_step_pending.tmp[2] = conv_step_scratch + 4*C; conv_step_pending.real[2] = (const float *) l2->data;   conv_step_pending.n[2] = n_l2;
        ssm_conv_step_f32<<<C/conv_step_head, conv_step_head, 0, ctx.stream()>>>(
            (const float *) g->src[0]->data, (const int32_t *) g->src[1]->data, g->src[0]->nb[1]/sizeof(float),
            (const float *) x->data, (const float *) w->data, w->nb[1]/sizeof(float),
            conv_step_pending.tmp[0], conv_step_pending.tmp[1], conv_step_pending.tmp[2], n_l2, eps, conv_step_head);
        return 0; // the original ops run and are compared after the L2_NORM node
    }
    ssm_conv_step_f32<<<C/conv_step_head, conv_step_head, 0, ctx.stream()>>>(
        (const float *) g->src[0]->data, (const int32_t *) g->src[1]->data, g->src[0]->nb[1]/sizeof(float),
        (const float *) x->data, (const float *) w->data, w->nb[1]/sizeof(float),
        (float *) upd->data, (float *) silu->data, (float *) l2->data, n_l2, eps, conv_step_head);
    CUDA_CHECK(cudaGetLastError());

    // Index of the L2_NORM node relative to i.
    int last_idx = i;
    for (int k = i + 1; k < cgraph->n_nodes; ++k) {
        if (cgraph->nodes[k] == l2) {
            last_idx = k;
            break;
        }
    }
    return last_idx - i;
}

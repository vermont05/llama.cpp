#include "common.cuh"
#include "unary.cuh"
#include "gated-norm.cuh"
#include "fwht.cuh"

// rms_norm(x) * w, then silu(gate) * that, then a head permutation made contiguous, as one kernel. In Qwen3.5 this is
// the gated norm after the gated delta net (5 nodes, 3 launches: rms_norm+mul, swiglu, cont). One block per output
// row; the arithmetic matches rms_norm_f32<256, true> and unary_gated_op_kernel<op_silu> exactly.
// GGML_CUDA_NO_GATED_NORM=1 disables it.

static constexpr int gated_norm_block = 256;

static __global__ void gated_norm_f32(
        const float * __restrict__ x, const float * __restrict__ gate, const float * __restrict__ w, float * __restrict__ dst,
        const int ncols, const int ne1, const int64_t nb1, const int64_t nb2, const float eps) {
    const int r   = blockIdx.x;
    const int i1  = r % ne1;
    const int i2  = r / ne1;
    const int h   = (i1*nb1 + i2*nb2) / (ncols*int64_t(sizeof(float))); // source row of output row r
    const int tid = threadIdx.x;

    x    += int64_t(h)*ncols;
    gate += int64_t(h)*ncols;
    dst  += int64_t(r)*ncols;

    float tmp = 0.0f;
    for (int col = tid; col < ncols; col += gated_norm_block) {
        const float xi = x[col];
        tmp += xi * xi;
    }
    __shared__ float s_sum[32];
    tmp = block_reduce<block_reduce_method::SUM, gated_norm_block>(tmp, s_sum);

    const float mean  = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    for (int col = tid; col < ncols; col += gated_norm_block) {
        const float n = scale * x[col] * w[col];
        dst[col] = ggml_cuda_op_silu_single(gate[col]) * n;
    }
}

int ggml_cuda_try_fuse_gated_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool disabled = getenv("GGML_CUDA_NO_GATED_NORM") != nullptr;
    if (disabled || i + 6 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * rn   = cgraph->nodes[i];
    const ggml_tensor * mul  = cgraph->nodes[i + 1];
    const ggml_tensor * glu  = cgraph->nodes[i + 2];
    const ggml_tensor * r1   = cgraph->nodes[i + 3];
    const ggml_tensor * r2   = cgraph->nodes[i + 4];
    const ggml_tensor * perm = cgraph->nodes[i + 5];
    const ggml_tensor * cont = cgraph->nodes[i + 6];
    if (rn->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || glu->op != GGML_OP_GLU || r1->op != GGML_OP_RESHAPE ||
            r2->op != GGML_OP_RESHAPE || perm->op != GGML_OP_PERMUTE || cont->op != GGML_OP_CONT) {
        return 0;
    }
    const ggml_tensor * x = rn->src[0];
    const ggml_tensor * w = mul->src[0] == rn ? mul->src[1] : mul->src[0];
    const int64_t ncols = x->ne[0];
    if (ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || glu->src[1] == nullptr || (glu->src[0] != mul && glu->src[1] != mul)) {
        return 0;
    }
    const ggml_tensor * gate = glu->src[0] == mul ? glu->src[1] : glu->src[0];
    const bool ok =
        x->type == GGML_TYPE_F32 && ncols <= gated_norm_block && x->nb[0] == sizeof(float) && x->nb[1] == ncols*sizeof(float) &&
        x->ne[2] == 1 && x->ne[3] == 1 &&
        (mul->src[0] == rn || mul->src[1] == rn) && w->type == GGML_TYPE_F32 && ggml_nelements(w) == ncols && ggml_is_contiguous(w) &&
        ggml_are_same_shape(mul, rn) &&
        gate->type == GGML_TYPE_F32 && ggml_are_same_shape(gate, rn) && ggml_is_contiguous(gate) &&
        glu->src[0] == gate && glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) &&
        r1->src[0] == glu && r2->src[0] == r1 && perm->src[0] == r2 && cont->src[0] == perm &&
        perm->ne[0] == ncols && perm->nb[0] == sizeof(float) && perm->ne[3] == 1 &&
        perm->nb[1] % (ncols*sizeof(float)) == 0 && perm->nb[2] % (ncols*sizeof(float)) == 0 &&
        perm->ne[1]*perm->ne[2] == x->ne[1] &&
        cont->type == GGML_TYPE_F32 && ggml_is_contiguous(cont);
    if (!ok) {
        return 0;
    }
    // No partial overlap between the output and the inputs (the separate ops never ran concurrently).
    auto overlaps = [](const ggml_tensor * a, const ggml_tensor * b) {
        const char * a0 = (const char *) a->data; const char * a1 = a0 + ggml_nbytes(a);
        const char * b0 = (const char *) b->data; const char * b1 = b0 + ggml_nbytes(b);
        return a0 < b1 && b0 < a1;
    };
    if (overlaps(cont, x) || overlaps(cont, gate) || overlaps(cont, w)) {
        return 0;
    }

    float eps;
    memcpy(&eps, rn->op_params, sizeof(float));

    // + the sign flip and FWHT that follow (RESHAPE, MUL(signs), RESHAPE, FWHT-hint MUL_MAT): one kernel for both
    if (i + 10 < cgraph->n_nodes) {
        const ggml_tensor * r1b  = cgraph->nodes[i + 7];
        const ggml_tensor * smul = cgraph->nodes[i + 8];
        const ggml_tensor * r2b  = cgraph->nodes[i + 9];
        ggml_tensor *       mm   = cgraph->nodes[i + 10];
        if (r1b->op == GGML_OP_RESHAPE && r1b->src[0] == cont && smul->op == GGML_OP_MUL && smul->src[0] == r1b &&
                r2b->op == GGML_OP_RESHAPE && r2b->src[0] == smul && mm->op == GGML_OP_MUL_MAT && mm->src[1] == r2b &&
                ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD && ggml_is_contiguous(smul) &&
                smul->src[1]->ne[0] == smul->ne[0] && ggml_nrows(smul->src[1]) == 1 &&
                !overlaps(mm, x) && !overlaps(mm, gate) && !overlaps(mm, w) && ncols == 128 &&
                // x and gate are views: the rest of their parent buffers may still be read later (e.g. the GDN state)
                !overlaps(mm, x->view_src ? x->view_src : x) && !overlaps(mm, gate->view_src ? gate->view_src : gate) &&
                ggml_cuda_op_gated_norm_fwht(ctx, x, gate, w, eps, perm, smul->src[1], mm, cont)) {
            return 10;
        }
    }

    gated_norm_f32<<<perm->ne[1]*perm->ne[2], gated_norm_block, 0, ctx.stream()>>>(
        (const float *) x->data, (const float *) gate->data, (const float *) w->data, (float *) cont->data,
        ncols, perm->ne[1], perm->nb[1], perm->nb[2], eps);
    CUDA_CHECK(cudaGetLastError());
    return 6;
}

#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// Fused CONCAT(conv state, qkv^T) -> SSM_CONV -> SILU with the conv state CPYs (Gated DeltaNet). One CPY normally,
// n_rs_seq+1 of them (state snapshots for partial rollback) when MTP drafting is on.
#define SSM_CONV_CONCAT_MAX_SNAPS 16
struct ssm_conv_concat_snaps {
    float * dst[SSM_CONV_CONCAT_MAX_SNAPS];
    int     col[SSM_CONV_CONCAT_MAX_SNAPS];
    int     n;
};
bool ggml_cuda_ssm_conv_concat_supported(const ggml_tensor * concat, ggml_tensor * const * cpys, int n_cpy,
                                         const ggml_tensor * ssm_conv, const ggml_tensor * silu);
void ggml_cuda_op_ssm_conv_concat(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * const * cpys,
                                  int n_cpy, ggml_tensor * ssm_conv, ggml_tensor * silu);

#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// Fused CONCAT(conv state, qkv^T) -> SSM_CONV -> SILU with the conv state CPY (Gated DeltaNet prompt path).
bool ggml_cuda_ssm_conv_concat_supported(const ggml_tensor * concat, const ggml_tensor * cpy, const ggml_tensor * ssm_conv,
                                         const ggml_tensor * silu);
void ggml_cuda_op_ssm_conv_concat(ggml_backend_cuda_context & ctx, ggml_tensor * concat, ggml_tensor * cpy,
                                  ggml_tensor * ssm_conv, ggml_tensor * silu);

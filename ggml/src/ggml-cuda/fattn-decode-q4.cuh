#include "common.cuh"

// Single-token decode attention for a Q4_0 K/V cache with GQA (RDNA3 only, see fattn-decode-q4.cu).
bool ggml_cuda_flash_attn_ext_decode_q4_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_decode_q4(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

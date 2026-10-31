#include "common.cuh"

// Single-token decode attention for a TURBO3_0 K/V cache with GQA (RDNA3 only, see fattn-decode-turbo3.cu).
bool ggml_cuda_flash_attn_ext_decode_turbo3_supported(const ggml_tensor * dst);
void ggml_cuda_flash_attn_ext_decode_turbo3(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

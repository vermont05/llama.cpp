#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// rms_norm -> mul(weight) -> mul(signs) -> reshape -> FWHT-hint matmul as one kernel. Writes mul and dst.
bool ggml_cuda_op_rms_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                                const ggml_tensor * signs, ggml_tensor * dst);

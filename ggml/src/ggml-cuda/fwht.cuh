#include "common.cuh"

// Returns whether the Fast Walsh-Hadamard transform could be used.
bool ggml_cuda_op_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * src, ggml_tensor * dst);
bool ggml_cuda_op_fwht_signed(ggml_backend_cuda_context & ctx, const ggml_tensor * src,
                              const ggml_tensor * signs, ggml_tensor * dst);

// rms_norm -> mul(weight) -> mul(signs) -> reshape -> FWHT-hint matmul as one kernel. Writes mul and dst.
bool ggml_cuda_op_rms_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * rms_norm, const ggml_tensor * mul,
                                const ggml_tensor * signs, ggml_tensor * dst);

// gated norm (see gated-norm.cu) + sign flip + FWHT-hint matmul (N = 1024) as one kernel; false if not applicable
bool ggml_cuda_op_gated_norm_fwht(ggml_backend_cuda_context & ctx, const ggml_tensor * x, const ggml_tensor * gate,
        const ggml_tensor * w, float eps, const ggml_tensor * perm, const ggml_tensor * signs, ggml_tensor * mm,
        const ggml_tensor * cont_dbg = nullptr);

// debug for GGML_CUDA_GN_FWHT_CHECK=1
struct gn_fwht_pending { const ggml_tensor * mm = nullptr; float * scratch = nullptr; int n = 0; const ggml_tensor * cont = nullptr; const ggml_tensor * x = nullptr; };
extern gn_fwht_pending ggml_cuda_gn_fwht_pending;
void ggml_cuda_gn_fwht_check(ggml_backend_cuda_context & ctx);
void ggml_cuda_gn_fwht_check_pre(ggml_backend_cuda_context & ctx);

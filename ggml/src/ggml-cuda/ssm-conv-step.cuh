#include "common.cuh"

// Fuses the single-token gated delta net conv chain starting at node i (a GET_ROWS of the conv state).
// Returns the number of following nodes it covered (0: no match, nothing launched).
int ggml_cuda_try_fuse_conv_step(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

// Debug hooks for GGML_CUDA_CONV_STEP_CHECK=1.
void ggml_cuda_conv_step_check_node(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);
void ggml_cuda_conv_step_check_report(ggml_backend_cuda_context & ctx);

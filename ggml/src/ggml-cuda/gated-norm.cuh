#include "common.cuh"

// Fuses rms_norm -> mul -> swiglu(gate) -> reshape -> reshape -> permute -> cont starting at node i (the RMS_NORM).
// Returns the number of following nodes it covered (0: no match, nothing launched).
int ggml_cuda_try_fuse_gated_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i);

#include "turbo3.cuh"

template <bool inverse>
static __global__ void k_turbo3_rotate_rows(const char * __restrict__ x, float * __restrict__ y,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const size_t nb1, const size_t nb2, const size_t nb3) {
    __shared__ float v[QK_TURBO3];

    const int     j  = threadIdx.x;
    const int64_t i1 = blockIdx.y;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i0 = int64_t(blockIdx.x)*QK_TURBO3 + j;

    v[j] = ((const float *) (x + i1*nb1 + i2*nb2 + i3*nb3))[i0];
    __syncthreads();
    if (inverse) {
        turbo3_wht(v, turbo3_wht_s2, turbo3_wht_s1);
    } else {
        turbo3_wht(v, turbo3_wht_s1, turbo3_wht_s2);
    }
    y[((i3*ne2 + i2)*ne1 + i1)*ne0 + i0] = v[j];
}

void ggml_cuda_turbo3_rotate_rows(const char * x, float * y, int64_t ne0, int64_t ne1, int64_t ne2, int64_t ne3,
        size_t nb1, size_t nb2, size_t nb3, bool inverse, cudaStream_t stream) {
    GGML_ASSERT(ne0 % QK_TURBO3 == 0 && ne1 <= 65535 && ne2*ne3 <= 65535);
    const dim3 grid(ne0/QK_TURBO3, ne1, ne2*ne3);
    if (inverse) {
        k_turbo3_rotate_rows<true><<<grid, QK_TURBO3, 0, stream>>>(x, y, ne0, ne1, ne2, nb1, nb2, nb3);
    } else {
        k_turbo3_rotate_rows<false><<<grid, QK_TURBO3, 0, stream>>>(x, y, ne0, ne1, ne2, nb1, nb2, nb3);
    }
    CUDA_CHECK(cudaGetLastError());
}

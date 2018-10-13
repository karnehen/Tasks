#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 32

__kernel void matrix_multiplication(__global float* a, __global float* b, __global float* c,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    // TODO
    size_t m = get_global_id(1);
    size_t n = get_global_id(0);
    size_t local_m = get_local_id(1);
    size_t local_n = get_local_id(0);

    __local float tileA[TILE_SIZE][TILE_SIZE+1];
    __local float tileB[TILE_SIZE][TILE_SIZE+1];

    float sum = 0.0;

    for (size_t tileK = 0; tileK < K; tileK += TILE_SIZE) {
        tileA[local_n][local_m] = 0.0;
        tileB[local_n][local_m] = 0.0;

        size_t k = tileK + local_n;
        if (m < M && k < K) {
            tileA[local_n][local_m] = a[m * K + k];
        }

        k = tileK + local_m;
        if (n < N && k < K) {
            tileB[local_n][local_m] = b[k * N + n];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[k][local_m] * tileB[local_n][k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (m < M && n < N) {
        c[m * N + n] = sum;
    }
}
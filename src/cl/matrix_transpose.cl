#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 32

__kernel void matrix_transpose(__global float* a, __global float* at,
                               unsigned int width, unsigned int height)
{
    size_t x = 2 * get_global_id(0);
    size_t y = 2 * get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE+1];

    size_t local_x = 2 * get_local_id(0);
    size_t local_y = 2 * get_local_id(1);
    tile[local_y][local_x] = 0;
    tile[local_y][local_x+1] = 0;
    tile[local_y+1][local_x] = 0;
    tile[local_y+1][local_x+1] = 0;

    if (x < width && y < height) {
        tile[local_y][local_x] = a[y * width + x];
    }
    if (x + 1 < width && y < height) {
        tile[local_y][local_x+1] = a[y * width + x + 1];
    }
    if (x < width && y + 1 < height) {
        tile[local_y+1][local_x] = a[(y + 1) * width + x];
    }
    if (x + 1 < width && y + 1 < height) {
        tile[local_y+1][local_x+1] = a[(y + 1) * width + x + 1];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    x = get_group_id(1) * get_local_size(1) * 2 + local_x;
    y = get_group_id(0) * get_local_size(0) * 2 + local_y;

    if (x < height && y < width) {
        at[y * height + x] = tile[local_x][local_y];
    }
    if (x + 1 < height && y < width) {
        at[y * height + x + 1] = tile[local_x+1][local_y];
    }
    if (x < height && y + 1 < width) {
        at[(y + 1) * height + x] = tile[local_x][local_y+1];
    }
    if (x + 1 < height && y + 1 < width) {
        at[(y + 1) * height + x + 1] = tile[local_x+1][local_y+1];
    }
}
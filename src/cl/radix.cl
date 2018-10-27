#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_SIZE 128
#define VALUES_PER_DIGIT 4

__kernel void radix(__global unsigned int* as, __global unsigned int* indexes,
        __global unsigned int* sums, unsigned int mask, unsigned int n)
{
    unsigned int second_mask = mask * 2;
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_size = get_local_size(0);
    unsigned int max_local_id = min(group_size, n - group_size * group_id);

    __local unsigned int local_as[WORK_SIZE];
    __local unsigned int local_sums_0[WORK_SIZE];
    __local unsigned int local_sums_1[WORK_SIZE];
    __local unsigned int local_sums_2[WORK_SIZE];
    __local unsigned int buffer_0[WORK_SIZE];
    __local unsigned int buffer_1[WORK_SIZE];
    __local unsigned int buffer_2[WORK_SIZE];
    bool first_digit;
    bool second_digit;

    if (global_id < n) {
        local_as[local_id] = as[global_id];
        first_digit = local_as[local_id] & mask;
        second_digit = local_as[local_id] & second_mask;

        if (local_id < max_local_id - 1) {
            local_sums_0[local_id + 1] = (!first_digit && !second_digit) ? 1 : 0;
            buffer_0[local_id + 1] = local_sums_0[local_id + 1];

            local_sums_1[local_id + 1] = (first_digit && !second_digit) ? 1 : 0;
            buffer_1[local_id + 1] = local_sums_1[local_id + 1];

            local_sums_2[local_id + 1] = (!first_digit && second_digit) ? 1 : 0;
            buffer_2[local_id + 1] = local_sums_2[local_id + 1];
        } else {
            local_sums_0[0] = 0;
            buffer_0[0] = 0;

            local_sums_1[0] = 0;
            buffer_1[0] = 0;

            local_sums_2[0] = 0;
            buffer_2[0] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int step = 1; step < group_size; step *= 2) {
        if (local_id < max_local_id && local_id >= step) {
            buffer_0[local_id] = local_sums_0[local_id - step] + local_sums_0[local_id];
            buffer_1[local_id] = local_sums_1[local_id - step] + local_sums_1[local_id];
            buffer_2[local_id] = local_sums_2[local_id - step] + local_sums_2[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (local_id < max_local_id && local_id >= step) {
            local_sums_0[local_id] = buffer_0[local_id];
            local_sums_1[local_id] = buffer_1[local_id];
            local_sums_2[local_id] = buffer_2[local_id];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        bool first_digit = local_as[max_local_id - 1] & mask;
        bool second_digit = local_as[max_local_id - 1] & second_mask;
        unsigned int total_zeros = local_sums_0[max_local_id - 1] + (!first_digit && !second_digit ? 1 : 0);
        unsigned int total_ones  = local_sums_1[max_local_id - 1] + (first_digit && !second_digit ? 1 : 0);
        unsigned int total_twos = local_sums_2[max_local_id - 1] + (!first_digit && second_digit ? 1 : 0);

        unsigned int last_index = VALUES_PER_DIGIT * (group_id + 1);
        sums[last_index] = total_zeros;
        sums[last_index + 1] = total_ones;
        sums[last_index + 2] = total_twos;
        sums[last_index + 3] = max_local_id - total_zeros - total_ones - total_twos;
    }

    if (global_id < n) {
        unsigned int local_index;
        if (!first_digit && !second_digit) {
            local_index = local_sums_0[local_id];
        } else if (first_digit && !second_digit) {
            local_index = local_sums_1[local_id];
        } else if (!first_digit && second_digit) {
            local_index = local_sums_2[local_id];
        } else {
            local_index = local_id - local_sums_0[local_id] - local_sums_1[local_id] - local_sums_2[local_id];
        }

        indexes[global_id] = local_index;
    }
}

__kernel void permute(__global unsigned int* as, __global unsigned int* indexes,
        __global unsigned int* sums, __global unsigned int* as_output,
        unsigned int mask, unsigned int n) {
    unsigned int second_mask = mask * 2;
    unsigned int global_id = get_global_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_count = get_num_groups(0);
    unsigned int last_index = VALUES_PER_DIGIT * group_count;
    unsigned int total_zeros = sums[last_index];
    unsigned int total_ones = sums[last_index + 1];
    unsigned int total_twos = sums[last_index + 2];

    if (global_id < n) {
        unsigned int new_index = indexes[global_id];
        unsigned int value = as[global_id];
        bool first_digit = value & mask;
        bool second_digit = value & second_mask;

        if (first_digit && second_digit) {
            new_index += sums[VALUES_PER_DIGIT * group_id + 3] + total_zeros + total_ones + total_twos;
        } else if (!first_digit && second_digit) {
            new_index += sums[VALUES_PER_DIGIT * group_id + 2] + total_zeros + total_ones;
        } else if (first_digit && ! second_digit) {
            new_index += sums[VALUES_PER_DIGIT * group_id + 1] + total_zeros;
        } else {
            new_index += sums[VALUES_PER_DIGIT * group_id];
        }

        as_output[new_index] = value;
    }
}
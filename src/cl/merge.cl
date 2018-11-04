#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define LOCAL_SIZE 128

unsigned int diagonal_first(unsigned int diagonal_index, unsigned int diagonal_number) {
    return diagonal_number + 1 - diagonal_index;
}

unsigned int diagonal_second(unsigned int diagonal_index, unsigned int diagonal_number) {
    return diagonal_index;
}

unsigned int local_diagonal_value(__local float* local_a, unsigned int diagonal_index,
                            unsigned int diagonal_number, unsigned int first_size) {
    unsigned int first = diagonal_first(diagonal_index, diagonal_number);
    unsigned int second = diagonal_second(diagonal_index, diagonal_number);

    if (first == 0) {
        return 0;
    } else if (local_a[first - 1] < local_a[second + first_size]) {
        return 0;
    } else {
        return 1;
    }
}

unsigned int diagonal_value(__global float* a, unsigned int diagonal_index,
                            unsigned int diagonal_number, unsigned int split_size,
                            unsigned int n) {
    unsigned int first = diagonal_first(diagonal_index, diagonal_number);
    unsigned int second = diagonal_second(diagonal_index, diagonal_number);

    if (first == 0) {
        return 0;
    }

    float first_value = (first - 1 < n) ? a[first - 1] : +INFINITY;
    float second_value = (second + split_size < n) ? a[second + split_size] : +INFINITY;

    if (first_value < second_value) {
        return 0;
    } else {
        return 1;
    }
}

void local_merge(__local float* local_a, unsigned int first_size, unsigned int second_size,
                 unsigned int diagonal_number, unsigned int n) {
    unsigned int globalId = get_global_id(0);
    unsigned int localId = get_local_id(0);

    unsigned int left = (diagonal_number < first_size) ? 0 : (diagonal_number - first_size + 1);
    unsigned int right = (diagonal_number < second_size) ? (diagonal_number + 1) : second_size;

    while (left < right) {
        unsigned int middle = (left + right) / 2;
        unsigned int middle_value = local_diagonal_value(local_a, middle, diagonal_number, first_size);

        if (middle_value == 1) {
            left = middle + 1;
        } else {
            right = middle;
        }
    }

    unsigned int diagonal_first_ = diagonal_first(left, diagonal_number);
    unsigned int diagonal_second_ = diagonal_second(left, diagonal_number);
    unsigned int groupId = get_group_id(0);
    unsigned int from = 0;
    if (diagonal_first_ == 0) {
        from = first_size + diagonal_second_ - 1;
    } else if (diagonal_second_ == 0) {
        from = diagonal_first_ - 1;
    } else {
        if (local_a[diagonal_first_ - 1] >= local_a[first_size + diagonal_second_ - 1]) {
            from = diagonal_first_ - 1;
        } else {
            from = first_size + diagonal_second_ - 1;
        }
    }

    float temp = 0.0;
    if (globalId < n) {
        temp = local_a[from];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (globalId < n) {
        local_a[diagonal_number] = temp;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
}

__kernel void merge(__global float* a, __global float* a_out, unsigned int n, unsigned int split_size,
                    __global unsigned int* diagonal_global_first, __global unsigned int* diagonal_global_second) {
    unsigned int globalId = get_global_id(0);
    unsigned int groupId = get_group_id(0);
    unsigned int localId = get_local_id(0);

    __local float local_a[LOCAL_SIZE];

    if (split_size == 1) {
        if (globalId < n) {
            local_a[localId] = a[globalId];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (localId % 2 == 0 && globalId + 1 < n) {
            if (local_a[localId + 1] < local_a[localId]) {
                float temp = local_a[localId];
                local_a[localId] = local_a[localId + 1];
                local_a[localId + 1] = temp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (globalId < n) {
            a_out[globalId] = local_a[localId];
        }
    } else if (2 * split_size <= LOCAL_SIZE) {
        if (globalId < n) {
            local_a[localId] = a[globalId];
        } else {
            local_a[localId] = +INFINITY;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int first_index = localId / split_size / 2 * split_size * 2;
        unsigned int diagonal_number = localId - first_index;
        local_merge(local_a + first_index, split_size, split_size, diagonal_number, n);

        if (globalId < n) {
            a_out[globalId] = local_a[localId];
        }
    } else {
        __local unsigned int first_begin;
        __local unsigned int first_end;
        __local unsigned int second_begin;
        __local unsigned int second_end;

        if (localId == 0) {
            first_begin = diagonal_global_first[2 * groupId];
            first_end = diagonal_global_first[2 * groupId + 1];
            second_begin = diagonal_global_second[2 * groupId];
            second_end = diagonal_global_second[2 * groupId + 1];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int index = n;
        if (localId < first_end - first_begin) {
            index = first_begin + localId;
        } else if (localId + first_begin - first_end < second_end - second_begin) {
            index = second_begin + localId + first_begin - first_end;
        }

        if (index < n) {
            local_a[localId] = a[index];
        } else {
            local_a[localId] = +INFINITY;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        local_merge(local_a, first_end - first_begin, second_end - second_begin, localId, n);

        if (globalId < n) {
            a_out[globalId] = local_a[localId];
        }
    }

}

__kernel void find_diagonal_indexes(__global float* a, unsigned int n, unsigned int split_size,
        __global unsigned int* diagonal_global_first, __global unsigned int* diagonal_global_second) {
    unsigned int globalId = get_global_id(0);
    unsigned int groupId = get_group_id(0);
    unsigned int localId = get_local_id(0);
    unsigned int first_index = globalId / split_size / 2 * split_size * 2;
    unsigned int second_index = first_index + split_size;

    if (globalId == first_index) {
        diagonal_global_first[2 * groupId] = first_index;
        diagonal_global_second[2 * groupId] = second_index;
    }

    if (localId == 0) {
        unsigned int diagonal_number = (globalId - first_index) + LOCAL_SIZE - 1;
        unsigned int left = (diagonal_number < split_size) ? 0 : (diagonal_number - split_size + 1);
        unsigned int right = (diagonal_number < split_size) ? (diagonal_number + 1) : split_size;

        while (left < right) {
            unsigned int middle = (left + right) / 2;
            unsigned int middle_value = diagonal_value(a + first_index, middle, diagonal_number,
                                                       split_size, n - first_index);

            if (middle_value == 1) {
                left = middle + 1;
            } else {
                right = middle;
            }
        }

        unsigned int diagonal_first_ = diagonal_first(left, diagonal_number);
        unsigned int diagonal_second_ = diagonal_second(left, diagonal_number);

        diagonal_global_first[2 * groupId + 1] = diagonal_first_ + first_index;
        diagonal_global_second[2 * groupId + 1] = diagonal_second_ + second_index;
        if (globalId + LOCAL_SIZE < first_index + 2 * split_size) {
            diagonal_global_first[2 * groupId + 2] = diagonal_first_ + first_index;
            diagonal_global_second[2 * groupId + 2] = diagonal_second_ + second_index;
        }
    }
}
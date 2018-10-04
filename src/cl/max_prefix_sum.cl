#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define MAX_LOCAL_SIZE 256

__kernel void max_prefix_sum(__global int* as, int size, __global int* max_sums, __global int* prefixes,
        __global int* group_sums) {
    int localId = get_local_id(0);
    int globalId = get_global_id(0);
    int groupId = get_group_id(0);
    int groupSize = get_local_size(0);

    __local int local_as[MAX_LOCAL_SIZE];
    if (globalId < size) {
        local_as[localId] = as[globalId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        int group_sum = local_as[0];
        int max_sum = group_sum;
        int prefix = globalId + 1;
        groupSize = min(groupSize, max(size, globalId) - globalId);

        for (int i = 1; i < groupSize; ++i) {
            group_sum += local_as[i];

            if (group_sum > max_sum) {
                max_sum = group_sum;
                prefix = globalId + i + 1;
            }
        }

        max_sums[groupId] = max_sum;
        prefixes[groupId] = prefix;
        group_sums[groupId] = group_sum;
    }
}

__kernel void max_prefix_sum_fast(__global int* as, int size, __global int* max_sums, __global int* prefixes,
                             __global int* group_sums) {
    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);
    unsigned int groupId = get_group_id(0);
    unsigned int groupSize = get_local_size(0);

    __local int local_as[MAX_LOCAL_SIZE * 4];
    __local int buffer[MAX_LOCAL_SIZE * 4];

    unsigned int blockGlobalId = groupId * groupSize * 4;
    unsigned int realGroupSize = min(4 * groupSize, size - blockGlobalId);
    while (2 * groupSize >= realGroupSize && groupSize > 1) {
        groupSize >>= 1;
    }

    unsigned int localIdSecond = localId + groupSize;
    unsigned int localIdThird = localIdSecond + groupSize;
    unsigned int localIdFourth = localIdThird + groupSize;

    if (localId < groupSize) {
        if (localId < realGroupSize) {
            local_as[localId] = as[blockGlobalId + localId];
            buffer[localId] = local_as[localId];
        }
        if (localIdSecond < realGroupSize) {
            local_as[localIdSecond] = as[blockGlobalId + localIdSecond];
            buffer[localIdSecond] = local_as[localIdSecond];
        }
        if (localIdThird < realGroupSize) {
            local_as[localIdThird] = as[blockGlobalId + localIdThird];
            buffer[localIdThird] = local_as[localIdThird];
        }
        if (localIdFourth < realGroupSize) {
            local_as[localIdFourth] = as[blockGlobalId + localIdFourth];
            buffer[localIdFourth] = local_as[localIdFourth];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId < groupSize) {
        int step;
        for (step = 1; step < groupSize; step *= 2) {
            if (localId >= step) {
                buffer[localId] = local_as[localId - step] + local_as[localId];
            }
            if (localIdSecond < realGroupSize) {
                buffer[localIdSecond] = local_as[localIdSecond - step] + local_as[localIdSecond];
            }
            if (localIdThird < realGroupSize) {
                buffer[localIdThird] = local_as[localIdThird - step] + local_as[localIdThird];
            }
            if (localIdFourth < realGroupSize) {
                buffer[localIdFourth] = local_as[localIdFourth - step] + local_as[localIdFourth];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
            local_as[localId] = buffer[localId];
            if (localIdSecond < realGroupSize) {
                local_as[localIdSecond] = buffer[localIdSecond];
            }
            if (localIdThird < realGroupSize) {
                local_as[localIdThird] = buffer[localIdThird];
            }
            if (localIdFourth < realGroupSize) {
                local_as[localIdFourth] = buffer[localIdFourth];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (localIdSecond < realGroupSize) {
            unsigned int maxStep = min(groupSize * 2, realGroupSize);

            for (; step < maxStep; step *= 2) {
                buffer[localIdSecond] = local_as[localIdSecond - step] + local_as[localIdSecond];
                if (localIdThird < realGroupSize) {
                    buffer[localIdThird] = local_as[localIdThird - step] + local_as[localIdThird];
                }
                if (localIdFourth < realGroupSize) {
                    buffer[localIdFourth] = local_as[localIdFourth - step] + local_as[localIdFourth];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                local_as[localIdSecond] = buffer[localIdSecond];
                if (localIdThird < realGroupSize) {
                    local_as[localIdThird] = buffer[localIdThird];
                }
                if (localIdFourth < realGroupSize) {
                    local_as[localIdFourth] = buffer[localIdFourth];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        if (localIdThird < realGroupSize) {
            unsigned int maxStep = min(groupSize * 3, realGroupSize);

            for (; step < maxStep; step *= 2) {
                buffer[localIdThird] = local_as[localIdThird - step] + local_as[localIdThird];
                if (localIdFourth < realGroupSize) {
                    buffer[localIdFourth] = local_as[localIdFourth - step] + local_as[localIdFourth];
                }

                barrier(CLK_LOCAL_MEM_FENCE);
                local_as[localIdThird] = buffer[localIdThird];
                if (localIdFourth < realGroupSize) {
                    local_as[localIdFourth] = buffer[localIdFourth];
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        }

        if (localIdFourth < realGroupSize) {
            for (; step < realGroupSize; step *= 2) {
                buffer[localIdFourth] = local_as[localIdFourth - step] + local_as[localIdFourth];

                if (realGroupSize - step >= realGroupSize % 32) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                local_as[localIdFourth] = buffer[localIdFourth];
                if (realGroupSize - step >= realGroupSize % 32) {
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        }
    }

    // local_as is used for max sum, buffer ist used for prefixes
    if (localId < groupSize) {
        buffer[localId] = blockGlobalId + localId;
        if (localIdSecond < realGroupSize) {
            int a = local_as[localId];
            int b = local_as[localIdSecond];

            if (a < b) {
                local_as[localId] = b;
                buffer[localId] = blockGlobalId + localIdSecond;
            }
        }
        if (localIdThird < realGroupSize) {
            int a = local_as[localId];
            int b = local_as[localIdThird];

            if (a < b) {
                local_as[localId] = b;
                buffer[localId] = blockGlobalId + localIdThird;
            }
        }
        if (localIdFourth < realGroupSize) {
            int a = local_as[localId];
            int b = local_as[localIdFourth];

            if (a < b) {
                local_as[localId] = b;
                buffer[localId] = blockGlobalId + localIdFourth;
            }
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int length = groupSize / 2; length >= 1; length /= 2) {
        if (localId < length) {
            int a = local_as[localId];
            int b = local_as[localId + length];

            if (a < b) {
                local_as[localId] = b;
                buffer[localId] = buffer[localId + length];
            }
        }

        if (length > 32) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    if (localId == 0) {
        max_sums[groupId] = local_as[0];
        prefixes[groupId] = buffer[0] + 1;
        group_sums[groupId] = local_as[realGroupSize - 1];
    }
}

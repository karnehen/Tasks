#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define MAX_LOCAL_SIZE 256

__kernel void sum(__global unsigned int* as, __global unsigned int* result, unsigned int size) {
    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);
    unsigned int groupSize = get_local_size(0);
    unsigned int groupId = get_group_id(0);

    __local unsigned int local_as[MAX_LOCAL_SIZE];

    if (globalId < size) {
        local_as[localId] = as[globalId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int sum = 0;
        groupSize = min(groupSize, max(size, globalId) - globalId);

        for (int i = 0; i < groupSize; ++i) {
            sum += local_as[i];
        }

        atomic_add(result, sum);
    }
}

__kernel void sum_fast(__global unsigned int* as, __global unsigned int* result, unsigned int size) {
    unsigned int localId = get_local_id(0);
    unsigned int globalId = get_global_id(0);
    unsigned int groupSize = get_local_size(0);
    unsigned int groupId = get_group_id(0);

    __local unsigned int local_as[MAX_LOCAL_SIZE * 4];

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
        }
        if (localIdSecond < realGroupSize) {
            local_as[localIdSecond] = as[blockGlobalId + localIdSecond];
        }
        if (localIdThird < realGroupSize) {
            local_as[localIdThird] = as[blockGlobalId + localIdThird];
        }
        if (localIdFourth < realGroupSize) {
            local_as[localIdFourth] = as[blockGlobalId + localIdFourth];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId < groupSize) {
        if (localIdSecond < realGroupSize) {
            local_as[localId] += local_as[localIdSecond];
        }
        if (localIdThird < realGroupSize) {
            local_as[localId] += local_as[localIdThird];
        }
        if (localIdFourth < realGroupSize) {
            local_as[localId] += local_as[localIdFourth];
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int length = groupSize / 2; length >= 1; length /= 2) {
        if (localId < length) {
            local_as[localId] += local_as[localId + length];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(result, local_as[0]);
    }
}
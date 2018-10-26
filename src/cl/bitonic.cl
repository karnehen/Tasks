#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_SIZE 256

__kernel void bitonic(__global float* as, unsigned int n,
        unsigned int max_step, unsigned int step)
{
    __local float local_as[WORK_SIZE];

    unsigned int global_id = get_global_id(0);

    if (step * 2 <= WORK_SIZE) {
        unsigned int local_id = get_local_id(0);
        unsigned int group_first = 2 * get_group_id(0) * get_local_size(0);

        local_as[local_id] = as[group_first + local_id];
        local_as[local_id + WORK_SIZE / 2] = as[group_first + local_id + WORK_SIZE / 2];

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int max_step_limit = min(n, (unsigned  int) WORK_SIZE);

        do {
            bool ascend = global_id % (max_step * 2) < max_step;

            for (; step > 0; step /= 2) {
                unsigned int first = (global_id % step) + (global_id / step * step * 2);
                unsigned int second = first + step;

                unsigned int local_first = (local_id % step) + (local_id / step * step * 2);
                unsigned int local_second = local_first + step;

                if (first < n && second < n) {
                    float a = local_as[local_first];
                    float b = local_as[local_second];

                    if (ascend == (a > b)) {
                        local_as[local_first] = b;
                        local_as[local_second] = a;
                    }
                }

                barrier(CLK_LOCAL_MEM_FENCE);
            }

            max_step *= 2;
            step = max_step;
        } while (max_step < max_step_limit);

        as[group_first + local_id] = local_as[local_id];
        as[group_first + local_id + WORK_SIZE / 2] = local_as[local_id + WORK_SIZE / 2];
    } else {
        bool ascend = global_id % (max_step * 2) < max_step;
        unsigned int first = (global_id % step) + (global_id / step * step * 2);
        unsigned int second = first + step;

        if (first < n && second < n) {
            float a = as[first];
            float b = as[second];

            if (ascend == (a > b)) {
                as[first] = b;
                as[second] = a;
            }
        }
    }
}

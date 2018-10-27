#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)
#define VALUES_PER_DIGIT 4
#define DIGITS_PER_STEP 2


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel permute(radix_kernel, radix_kernel_length, "permute");
        permute.compile();

        unsigned int workGroupSize = 128;
        unsigned int groups_count = (n + workGroupSize - 1) / workGroupSize;
        unsigned int global_work_size = groups_count * workGroupSize;
        unsigned int sums_size = VALUES_PER_DIGIT * (groups_count + 1);

        gpu::gpu_mem_32u sums_gpu;
        sums_gpu.resizeN(sums_size);

        gpu::gpu_mem_32u indexes;
        indexes.resizeN(n);

        gpu::gpu_mem_32u as_buffer;
        as_buffer.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            std::vector<unsigned int> sums(sums_size, 0);
            sums_gpu.writeN(sums.data(), sums_size);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных

            unsigned int mask = 1;
            for (unsigned int digit = 0; digit < 32 / DIGITS_PER_STEP; ++digit) {
                if (digit % 2) {
                    radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                               as_buffer, indexes, sums_gpu, mask, n);
                } else {
                    radix.exec(gpu::WorkSize(workGroupSize, global_work_size),
                               as_gpu, indexes, sums_gpu, mask, n);
                }

                sums_gpu.readN(sums.data(), sums_size);

                for (unsigned int i = VALUES_PER_DIGIT; i < sums_size; ++i) {
                    sums[i] += sums[i - VALUES_PER_DIGIT];
                }

                sums_gpu.writeN(sums.data(), sums_size);

                if (digit % 2) {
                    permute.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_buffer, indexes, sums_gpu, as_gpu, mask, n);
                } else {
                    permute.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                 as_gpu, indexes, sums_gpu, as_buffer, mask, n);
                }

                mask <<= DIGITS_PER_STEP;
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000/1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}

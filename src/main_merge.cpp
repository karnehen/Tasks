#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

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


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32*1024*1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
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

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32f as_buffer_gpu;
    as_buffer_gpu.resizeN(n);

    unsigned int workGroupSize = 128;
    unsigned int groupCount = (n + workGroupSize - 1) / workGroupSize;
    unsigned int global_work_size = groupCount * workGroupSize;

    gpu::gpu_mem_32u diagonal_first;
    diagonal_first.resizeN(groupCount * 2);
    gpu::gpu_mem_32u diagonal_second;
    diagonal_second.resizeN(groupCount * 2);

    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();

        ocl::Kernel find_diagonal_indexes(merge_kernel, merge_kernel_length, "find_diagonal_indexes");
        find_diagonal_indexes.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart(); // Запускаем секундомер после прогрузки данных чтобы замерять время работы кернела, а не трансфер данных
            bool use_buffer = false;

            for (unsigned int split_size = 1; split_size < n; split_size *= 2) {
                if (split_size >= workGroupSize) {
                    if (use_buffer) {
                        find_diagonal_indexes.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                                   as_buffer_gpu, n, split_size, diagonal_first, diagonal_second);
                    } else {
                        find_diagonal_indexes.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                                   as_gpu, n, split_size, diagonal_first, diagonal_second);
                    }
                }
                if (use_buffer) {
                    merge.exec(gpu::WorkSize(workGroupSize, global_work_size),
                               as_buffer_gpu, as_gpu, n, split_size, diagonal_first, diagonal_second);
                } else {
                    merge.exec(gpu::WorkSize(workGroupSize, global_work_size),
                               as_gpu, as_buffer_gpu, n, split_size, diagonal_first, diagonal_second);
                }
                use_buffer = !use_buffer;
            }
            if (use_buffer) {
                as_gpu.write(as_buffer_gpu, n * sizeof(float));
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
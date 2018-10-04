#include <libclew/ocl_init.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <fstream>
#include "commons.h"

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
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        OpenCLWrapper openCLWrapper(argc, argv);
        ProgramWrapper programWrapper(openCLWrapper, "../src/cl/sum.cl");
        KernelWrapper kernelWrapper(programWrapper, "sum");
        KernelWrapper kernelWrapperFast(programWrapper, "sum_fast");

        cl_mem as_gpu = openCLWrapper.createInputBuffer(n, as.data());
        cl_mem sum_gpu = openCLWrapper.createOutputBuffer<unsigned int>(1);

        unsigned int sum = 0;
        size_t groupSize = 256;
        size_t workSize = (n + groupSize - 1) / groupSize * groupSize;

        {
            timer t;

            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                openCLWrapper.writeMemoryBuffer(sum_gpu, &sum, 1);
                kernelWrapper.runKernel(1, &workSize, &groupSize, as_gpu, sum_gpu, n);
                openCLWrapper.readMemoryBuffer(sum_gpu, &sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                sum = 0;
                t.nextLap();
            }

            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        workSize = ((n + 3) / 4 + groupSize - 1) / groupSize * groupSize;

        {
            timer t;

            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                openCLWrapper.writeMemoryBuffer(sum_gpu, &sum, 1);
                kernelWrapperFast.runKernel(1, &workSize, &groupSize, as_gpu, sum_gpu, n);
                openCLWrapper.readMemoryBuffer(sum_gpu, &sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                sum = 0;
                t.nextLap();
            }

            std::cout << "GPU fast: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU fast: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }

    return 0;
}
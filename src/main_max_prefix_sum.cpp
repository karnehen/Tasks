#include <fstream>

#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
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
    int max_n = (1 << 24);

    for (int n = 2; n <= max_n; n *= 2) {
        std::cout << "______________________________________________" << std::endl;
        int values_range = std::min(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << (-values_range) << "; " << values_range << "]" << std::endl;

        std::vector<int> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = (unsigned int) r.next(-values_range, values_range);
        }

        int reference_max_sum;
        int reference_result;
        {
            int max_sum = as[0];
            int sum = max_sum;
            int result = 1;
            for (int i = 1; i < n; ++i) {
                sum += as[i];
                if (sum > max_sum) {
                    max_sum = sum;
                    result = i + 1;
                }
            }
            reference_max_sum = max_sum;
            reference_result = result;
        }
        std::cout << "Max prefix sum: " << reference_max_sum << " on prefix [0; " << reference_result << ")" << std::endl;

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                int max_sum = as[0];
                int sum = max_sum;
                int result = 1;
                for (int i = 1; i < n; ++i) {
                    sum += as[i];
                    if (sum > max_sum) {
                        max_sum = sum;
                        result = i + 1;
                    }
                }
                EXPECT_THE_SAME(reference_max_sum, max_sum, "CPU result should be consistent!");
                EXPECT_THE_SAME(reference_result, result, "CPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            // TODO: implement on OpenCL
            OpenCLWrapper openCLWrapper(argc, argv);
            ProgramWrapper programWrapper(openCLWrapper, "../src/cl/max_prefix_sum.cl");
            KernelWrapper kernelWrapper(programWrapper, "max_prefix_sum");
            KernelWrapper kernelWrapperFast(programWrapper, "max_prefix_sum_fast");

            size_t groupSize = 256;
            cl_mem as_gpu = openCLWrapper.createInputBuffer(n, as.data());

            {
                size_t groupCount = (n + groupSize - 1) / groupSize;
                size_t workSize = groupCount * groupSize;

                std::vector<int> max_sums(groupCount, 0);
                std::vector<int> prefixes(groupCount, 0);
                std::vector<int> group_sums(groupCount, 0);

                cl_mem max_sums_gpu = openCLWrapper.createOutputBuffer<int>(groupCount);
                cl_mem prefixes_gpu = openCLWrapper.createOutputBuffer<int>(groupCount);
                cl_mem group_sums_gpu = openCLWrapper.createOutputBuffer<int>(groupCount);

                timer t;

                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    kernelWrapper.runKernel(1, &workSize, &groupSize, as_gpu, n, max_sums_gpu, prefixes_gpu,
                                            group_sums_gpu);

                    openCLWrapper.readMemoryBuffer(max_sums_gpu, max_sums.data(), groupCount);
                    openCLWrapper.readMemoryBuffer(prefixes_gpu, prefixes.data(), groupCount);
                    openCLWrapper.readMemoryBuffer(group_sums_gpu, group_sums.data(), groupCount);

                    int max_sum = max_sums[0];
                    int result = prefixes[0];
                    int cummulative_sum = group_sums[0];

                    for (int i = 1; i < groupCount; ++i) {
                        int sum = cummulative_sum + max_sums[i];

                        if (sum > max_sum) {
                            max_sum = sum;
                            result = prefixes[i];
                        }

                        cummulative_sum += group_sums[i];
                    }

                    EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent!");
                    EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent!");
                    t.nextLap();
                }

                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }

            {
                size_t groupCount = ((n + 3) / 4 + groupSize - 1) / groupSize;
                size_t workSize = groupCount * groupSize;

                std::vector<int> max_sums(groupCount, 0);
                std::vector<int> prefixes(groupCount, 0);
                std::vector<int> group_sums(groupCount, 0);

                cl_mem max_sums_gpu = openCLWrapper.createOutputBuffer<int>(groupCount);
                cl_mem prefixes_gpu = openCLWrapper.createOutputBuffer<int>(groupCount);
                cl_mem group_sums_gpu = openCLWrapper.createOutputBuffer<int>(groupCount);

                timer t;

                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    kernelWrapperFast.runKernel(1, &workSize, &groupSize, as_gpu, n, max_sums_gpu, prefixes_gpu,
                                            group_sums_gpu);

                    openCLWrapper.readMemoryBuffer(max_sums_gpu, max_sums.data(), groupCount);
                    openCLWrapper.readMemoryBuffer(prefixes_gpu, prefixes.data(), groupCount);
                    openCLWrapper.readMemoryBuffer(group_sums_gpu, group_sums.data(), groupCount);

                    int max_sum = max_sums[0];
                    int result = prefixes[0];
                    int cummulative_sum = group_sums[0];

                    for (int i = 1; i < groupCount; ++i) {
                        int sum = cummulative_sum + max_sums[i];

                        if (sum > max_sum) {
                            max_sum = sum;
                            result = prefixes[i];
                        }

                        cummulative_sum += group_sums[i];
                    }

                    EXPECT_THE_SAME(reference_max_sum, max_sum, "GPU result should be consistent! 1");
                    EXPECT_THE_SAME(reference_result, result, "GPU result should be consistent! 2");
                    t.nextLap();
                }

                std::cout << "GPU fast: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU fast: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}

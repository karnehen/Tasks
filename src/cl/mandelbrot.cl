#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results, unsigned int width, unsigned int height,
        float fromX, float fromY, float sizeX, float sizeY, unsigned int iters,
        unsigned int smoothing, unsigned int antialiasing)
{
    // TODO если хочется избавиться от зернистости и дрожжания при интерактивном погружении - добавьте anti-aliasing:
    // грубо говоря при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int col = get_global_id(0);
    const unsigned int row = get_global_id(1);

    if (col >= width || row >= height) {
        return;
    }

    float total_result = 0;

    for (unsigned int sub_col = 1; sub_col <= antialiasing; ++sub_col) {
        for (unsigned int sub_row = 1; sub_row <= antialiasing; ++sub_row) {
            float x0 = fromX + (col + sub_col / (1.0 + antialiasing)) * sizeX / width;
            float y0 = fromY + (row + sub_row / (1.0 + antialiasing)) * sizeY / height;

            float x = x0;
            float y = y0;

            int iter = 0;
            for (; iter < iters; ++iter) {
                float xPrev = x;
                x = x * x - y * y + x0;
                y = 2.0f * xPrev * y + y0;
                if ((x * x + y * y) > threshold2) {
                    break;
                }
            }

            float result = iter;
            if (smoothing && iter != iters) {
                result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
            }

            total_result += result;
        }
    }

    total_result = 1.0f * total_result / iters / antialiasing / antialiasing;
    results[row * width + col] = total_result;
}

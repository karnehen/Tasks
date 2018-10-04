#pragma once

#include <set>

#include <libgpu/device.h>
#include <libutils/misc.h>
#include <CL/cl.h>

class OpenCLWrapper {
public:
    OpenCLWrapper(int argc, char** argv);
    ~OpenCLWrapper();

    template <class T>
    cl_mem createInputBuffer(size_t size, T* ptr) {
        return createMemoryBuffer<T>(size, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, ptr);
    }

    template <class T>
    cl_mem createOutputBuffer(size_t size) {
        return createMemoryBuffer<T>(size, CL_MEM_WRITE_ONLY, nullptr);
    }

    template <class T>
    void readMemoryBuffer(cl_mem buffer, T* ptr, size_t size) {
        OCL_SAFE_CALL(clEnqueueReadBuffer(command_queue, buffer, CL_TRUE, 0, size * sizeof(T), ptr,
                0, nullptr, nullptr));
    }

    template <class T>
    void writeMemoryBuffer(cl_mem buffer, T* ptr, size_t size) {
        OCL_SAFE_CALL(clEnqueueWriteBuffer(command_queue, buffer, CL_TRUE, 0, size * sizeof(T), ptr,
                0, nullptr, nullptr));
    }

    void releaseMemoryBuffer(cl_mem buffer);

    cl_command_queue commandQueue() const;
    cl_context context() const;
    cl_device_id deviceId() const;
    const cl_device_id* deviceIdPtr() const;

private:
    cl_command_queue command_queue;
    cl_context context_;
    gpu::Device device;
    std::set<cl_mem> memory_buffers;

    template <class T>
    cl_mem createMemoryBuffer(size_t size, cl_mem_flags flags, void* ptr) {
        cl_int error_code;
        cl_mem result = (clCreateBuffer(context_, flags, size * sizeof(T), ptr, &error_code));
        OCL_SAFE_CALL(error_code);
        memory_buffers.insert(result);

        return result;
    }
};

class ProgramWrapper {
public:
    ProgramWrapper(const OpenCLWrapper& openCLWrapper, const char* path);
    ~ProgramWrapper();

    cl_program program() const;
    const OpenCLWrapper& openCLWrapper() const;

private:
    cl_program program_;
    const OpenCLWrapper& opencl_wrapper;
};

class KernelWrapper {
public:
    KernelWrapper(const ProgramWrapper& programWrapper, const char* name);
    ~KernelWrapper();

    cl_kernel kernel() const;

    template <class... Args>
    void runKernel(cl_uint workDim, const size_t* workSize, const size_t* groupSize, Args... args) {
        runKernel(0, workDim, workSize, groupSize, args...);
    }

private:
    cl_kernel kernel_;
    const ProgramWrapper& program_wrapper;

    void runKernel(size_t argNumber, cl_uint workDim, const size_t* workSize, const size_t* groupSize) {
        cl_event event;
        OCL_SAFE_CALL(clEnqueueNDRangeKernel(program_wrapper.openCLWrapper().commandQueue(), kernel_, workDim, nullptr,
                                             workSize, groupSize, 0, nullptr, &event));
        OCL_SAFE_CALL(clWaitForEvents(1, &event));
    }

    template <class T, class... Args>
    void runKernel(size_t argNumber, cl_uint workDim, const size_t* workSize, const size_t* groupSize,
                                  T& firstArgument, Args... args) {
        OCL_SAFE_CALL(clSetKernelArg(kernel_, argNumber++, sizeof(firstArgument), &firstArgument));
        runKernel(argNumber, workDim, workSize, groupSize, args...);
    }
};
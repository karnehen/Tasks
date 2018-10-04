#include <fstream>

#include "commons.h"

OpenCLWrapper::OpenCLWrapper(int argc, char **argv) {
    device = gpu::chooseGPUDevice(argc, argv);

    cl_platform_id platform_id;
    cl_device_id device_id = device.device_id_opencl;
    OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_PLATFORM, sizeof(cl_platform_id),
                                  &platform_id, nullptr));

    cl_context_properties context_properties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform_id, 0};
    cl_int error_code;
    context_ = clCreateContext(context_properties, 1, &device_id, nullptr, nullptr, &error_code);
    OCL_SAFE_CALL(error_code);

    command_queue = clCreateCommandQueue(context_, device_id, 0, &error_code);
    OCL_SAFE_CALL(error_code);
}

OpenCLWrapper::~OpenCLWrapper() {
    for (cl_mem mem : memory_buffers) {
        OCL_SAFE_CALL(clReleaseMemObject(mem));
    }

    OCL_SAFE_CALL(clReleaseCommandQueue(command_queue));
    OCL_SAFE_CALL(clReleaseContext(context_));
}

cl_command_queue OpenCLWrapper::commandQueue() const {
    return command_queue;
}

void OpenCLWrapper::releaseMemoryBuffer(cl_mem buffer) {
    if (memory_buffers.count(buffer)) {
        memory_buffers.erase(buffer);
        OCL_SAFE_CALL(clReleaseMemObject(buffer));
    }
}

cl_context OpenCLWrapper::context() const {
    return context_;
}

cl_device_id OpenCLWrapper::deviceId() const {
    return device.device_id_opencl;
}

const cl_device_id* OpenCLWrapper::deviceIdPtr() const {
    return &device.device_id_opencl;
}

ProgramWrapper::ProgramWrapper(const OpenCLWrapper &openCLWrapper, const char *path)
        : opencl_wrapper(openCLWrapper) {
    std::string program_sources;
    std::ifstream file(path);
    program_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    if (program_sources.size() == 0) {
        throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
    }
    auto string_content = program_sources.c_str();

    cl_int error_code;
    program_ = clCreateProgramWithSource(openCLWrapper.context(), 1, &string_content, nullptr, &error_code);
    OCL_SAFE_CALL(error_code);

    error_code = clBuildProgram(program_, 1, openCLWrapper.deviceIdPtr(), nullptr, nullptr, nullptr);

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program_, openCLWrapper.deviceId(), CL_PROGRAM_BUILD_LOG,
            0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program_, openCLWrapper.deviceId(), CL_PROGRAM_BUILD_LOG,
            log_size, log.data(), nullptr));

    if (error_code != CL_SUCCESS) {
        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
        }

        throw std::runtime_error("Build failed!");
    }
}

ProgramWrapper::~ProgramWrapper() {
    OCL_SAFE_CALL(clReleaseProgram(program_));
}

cl_program ProgramWrapper::program() const {
    return program_;
}

const OpenCLWrapper& ProgramWrapper::openCLWrapper() const {
    return opencl_wrapper;
}

KernelWrapper::KernelWrapper(const ProgramWrapper &programWrapper, const char *name)
        : program_wrapper(programWrapper) {
    cl_int error_code;
    kernel_ = clCreateKernel(programWrapper.program(), name, &error_code);
    OCL_SAFE_CALL(error_code);
}

KernelWrapper::~KernelWrapper() {
    OCL_SAFE_CALL(clReleaseKernel(kernel_));
}

cl_kernel KernelWrapper::kernel() const {
    return kernel_;
}


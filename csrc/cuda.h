#ifndef CUDA_KERNEL_H
#define CUDA_KERNEL_H

__host__ void cpu_to_cuda(Tensor* tensor, int device_id);
    __host__ void cuda_to_cpu(Tensor* tensor);
    __host__ void free_cuda(float* data);
    
    __global__ void add_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void add_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data);

    __global__ void sub_tensor_cuda_kernel(float* data1, float* data2, float* result_data, int size);
    __host__ void sub_tensor_cuda(Tensor* tensor1, Tensor* tensor2, float* result_data)
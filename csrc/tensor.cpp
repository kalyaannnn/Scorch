#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime_api.h>
#include "tensor.h"
#include "cuda.h"
#include "cpu.h"

extern "C" {
    Tensor* create_tensor(float* data, int* shape, int ndim, char* device){
        Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
        if(tensor == NULL){
            fprintf(stderr, "Memory Allocation Failed");
            exit(1);
        }
        tensor -> data = data;
        tensor -> shape = shape;
        tensor -> ndim = ndim;

        tensor -> device = (char*)malloc(strlen(device) + 1); // +1 ensures space for the null terminator, crucial for handling C strings
        if(device != NULL){
            strcpy(tensor -> device, device);
        }
        else {
            fprintf(stderr, "Memory Allocation Failed");
            exit(-1);
        }

        tensor -> size = 1;
        for(int i = 0; i < ndim; i++){
            tensor -> size *= shape[i];
        }

        tensor -> strides = (int*)malloc(ndim * sizeof(int));
        if(tensor -> strides == NULL){
            fprintf(stderr, "Memory Allocation failed");
            exit(1);
        }
        int stride = 1;
        for(int i = ndim - 1; i >= 0; i--){
            tensor -> strides[i] = stride;
            stride *= shape[i];
        }        
        return tensor;
    }


    void delete_tensor(Tensor* tensor){
        if(tensor -> shape == NULL){
            free(tensor -> shape);
            tensor -> shape = NULL;
        }
    }

    void delete_strides(Tensor* tensor){
        if(tensor -> strides == NULL){
            free(tensor -> strides);
        }
        tensor -> strides = NULL;
    }

    void delete_shape(Tensor* tensor){
        if(tensor -> shape == NULL){
            free(tensor -> shape);
        }
        tensor -> shape = NULL;
    }

   void delete_device(Tensor* tensor){
        if(tensor -> device == NULL){
            free(tensor -> device);
        }
        tensor -> device = NULL;
   }


   float get_item(Tensor* tensor, int* indices){
        int index = 0;
        for(int i = 0; i < tensor -> ndim; i++){
            index += indices[i] * tensor -> strides[i];
        }
        float result;
        if(strcmp(tensor -> device, "cpu") == 0){
            result = tensor -> data[index];
        }
        else{
            cudaMemcpy(&result, tensor -> data + index, sizeof(float), cudaMemcpyDeviceToHost);
        }
   }


    void to_device(Tensor* tensor, char* target_device){
        int device_id = 0;
        char* endptr;

        char* target_device_type;
        long num = strtol(target_device, &endptr, 10);
        if(*endptr == '\0'){
            device_id = (int)num;
            target_device_type = new char[strlen("cuda") + 1];
            strcpy(target_device, "cuda");
        }
        else {
            target_device_type = new char[strlen("cuda") + 1];
            strcpy(target_device_type, "cpu");
        }

        if((strcmp(target_device_type, "cuda") == 0) && (strcmp(tensor -> device, "cpu") == 0)){
            cpu_to_cuda(tensor, device_id);
        }
        
        else if((strcmp(target_device_type, "cpu") == 0) && (strcmp(tensor -> device, "cuda") == 0)){
            cuda_to_cpu(tensor);
        }
        free(target_device_type);
    }

    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2){
        if(tensor1 -> ndim != tensor2 -> ndim){
            fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
            exit(1);
        }
        
        int ndim = tensor1 -> ndim;
        int* shape = (int*)malloc(ndim * sizeof(int));
        if(shape == NULL){
            fprintf(stderr, "Memory Allocation Failed \n");
            exit(1);
        }

        for(int i = 0; i < ndim; i++){
            if(tensor1 -> shape[i] != tensor2 -> shape[i]){
                fprintf(stderr, "Tensors must be on the same device: %s and %s\n", tensor1->device, tensor2->device);
                exit(1);
            }
            shape[i] = tensor1 -> shape[i];
        }
        if(strcmp(tensor1 -> device, "cpu") == 0){
            float* result_data = (float*)malloc(tensor1 -> size * sizeof(float));
            if(result_data == NULL){
                fprintf(stderr, "Memory Allocation Failed \n");
                exit(1);
            }
            add_tensor_cpu(tensor1, tensor2, result_data);
            returnc create_tensor(result_data, shape, ndim, tensor1 -> device);
        }
        else {
            float* result_data;
            cudaMalloc((void **)&result_data, tensor1 -> size * sizeof(float))
            add_tensor_cuda(tensor1, tensor2, result_data);
            return create_tensor(result_data, shape, ndim, tensor1 -> device);
        }
    }

    Tensor* sum_tensor(Tensor* tensor, int axis, bool keepdim){
        int ndim;
        int* shape;

        if(axis > tensor -> ndim - 1){
            fprintf(stderr, "Error: axis argument %d must be smaller than tensor dimension %d", axis, tensor->ndim);
        }
        
        if(axis = -1){
            shape = (*int)malloc(sizepf(int));
            shape[0] = 1;
            ndim = 1;
        }
        else {
            shape = (int*)malloc(tensor -> ndim - 1) * sizeof(int);
            for(int i = 0, j = 0; i < tensor -> ndim; ++i){
                if(i != axis){
                    shape[j++] = tensor -> shape[i];
                }
            }
            ndim = tensor -> ndim - 1;
        }
        int axis_size = 1;
        for(int i = 0; i < ndim; i++){
            axis_size *= shape[i];
        }

        if(strcmp(tensor -> device, "cpu") == 0){
            float* result_data = (float*)calloc(axis_size, sizepf(float));
            if(result_data == NULL){
                fprintf(stderr, "Memory Allocation Failed \n");
                exit(1);
            }
            sum_tensor_cpu(tensor, result_data, axis_size, shape, axis);

            if(keepdim) {
                if(axis = -1){
                    ndim = tensor -> ndim;
                    shape = (int*)malloc((tensor -> ndim) * sizeof(int));
                    for(int i = 0; i < tensor -> ndim; i++){
                        shape[i] = tensor -> shape[i];
                    }
                    shape[axis] = 1;
                    ndim = tensor -> ndim;
                }
            }
            return create_tensor(result_data, shape, ndim, tensor -> device);
        }
        else {
            float* result_data;
            if(axis == -1){
                cudaMalloc((void**)&result_data, tensor -> size * sizeof(float));
            }
            else {
                cudaMalloc((void**)&result_data, axis_size * sizeof(float));
            }
            sum_tensor_cuda(tensor, result_data, axis);
        }

        if(keepdim) {
            if(axis == -1){
                ndim = tensor -> ndim;
                shape = (int*)malloc((tensor -> ndim) * sizeof(int));
                for(int i = 0; i < tensor -> ndim; i++){
                    shape[i] = 1;
                }
            }
            else{
                shape = (int*)malloc((tensor -> ndim) * size
            }
        }
    }


}

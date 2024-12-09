#include <stdio.h>
#include <cuda_runtime.h>

// CUDA Kernel for adding numbers
__global__ void addNumbers(int *input, int *result) {
    int tid = threadIdx.x;
    
    // Each thread adds one number to the result
    atomicAdd(result, input[tid]);
}

int main() {
    // Input array with 5 numbers
    int h_input[5] = {1, 1, 1, 1, 1};
    int h_result = 0;
    
    // Allocate device memory
    int *d_input, *d_result;
    cudaMalloc(&d_input, 5 * sizeof(int));
    cudaMalloc(&d_result, sizeof(int));
    
    // Copy input array to device
    cudaMemcpy(d_input, h_input, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel with 5 threads in one block
    addNumbers<<<1, 5>>>(d_input, d_result);
    
    // Copy result back to host
    cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print result
    printf("Sum of numbers: %d\n", h_result);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_result);
    
    return 0;
} 
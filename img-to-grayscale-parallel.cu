#include <opencv2/opencv.hpp>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

inline void cudaCheckError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error in %s (%d)\n", file, line);
    } else {
        printf("CUDA success in %s (%d)\n", file, line);
    }
}

// CUDA kernel for grayscale conversion
__global__ void grayscale_kernel(const unsigned char* input, unsigned char* output, 
                                int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int idx = (y * width + x) * 3;  // Input index (BGR)
    int out_idx = y * width + x;    // Output index (grayscale)
    
    output[out_idx] = static_cast<unsigned char>(
        0.299f * input[idx + 2] +    // R
        0.587f * input[idx + 1] +    // G
        0.114f * input[idx]);        // B
}

// Function to convert an image to grayscale
void convert_to_grayscale(const char *input_file, const char *output_file) {
    nvtxRangePush("Load image");

    // Load the input image
    cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
    if (image.empty()) {
        fprintf(stderr, "Error: Could not open or find the image '%s'.\n", input_file);
        return;
    }
    if (image.channels() != 3) {
        fprintf(stderr, "Error: Input image must have 3 channels (BGR).\n");
        return;
    }

    printf("Loaded image '%s' with dimensions %dx%d and %d channels.\n", 
           input_file, image.cols, image.rows, image.channels());
    printf("Launching kernel with grid\n");
    
    // Create a matrix for the grayscale image
    cv::Mat grayscale_image(image.rows, image.cols, CV_8UC1);
    nvtxRangePop();

    // Allocate device memory
    unsigned char *d_input, *d_output;
    size_t input_size = image.total() * image.channels();
    size_t output_size = image.total();
    printf("Image dimensions: %dx%d, Channels: %d\n", image.cols, image.rows, image.channels());
    printf("Input size: %zu, Output size: %zu\n", input_size, output_size);
    
    cudaCheckError(cudaMalloc(&d_input, input_size), __FILE__, __LINE__);
    cudaCheckError(cudaMalloc(&d_output, output_size), __FILE__, __LINE__);

    // Copy input image to device memory
    cudaCheckError(cudaMemcpy(d_input, image.data, input_size, cudaMemcpyHostToDevice), __FILE__, __LINE__);

    // Configure grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((image.cols + block.x - 1) / block.x, 
              (image.rows + block.y - 1) / block.y);

    // Launch the kernel
    printf("Launching kernel with grid (%d, %d) and block (%d, %d)...\n", grid.x, grid.y, block.x, block.y);
    nvtxRangePush("Convert to grayscale");
    grayscale_kernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows);
    cudaCheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    nvtxRangePop();

    // Copy result back to host memory
    // cudaCheckError(cudaMemcpy(grayscale_image.data, d_output, output_size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    // Free device memory
    cudaCheckError(cudaFree(d_input), __FILE__, __LINE__);
    cudaCheckError(cudaFree(d_output), __FILE__, __LINE__);

    // Save the grayscale image
    nvtxRangePush("Save grayscale image");
    if (!cv::imwrite(output_file, grayscale_image)) {
        fprintf(stderr, "Error: Unable to save grayscale image to '%s'.\n", output_file);
    } else {
        printf("Grayscale image saved to '%s'.\n", output_file);
    }
    nvtxRangePop();
}

// Add this function before convert_to_grayscale
bool check_gpu() {
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    
    if (error != cudaSuccess) {
        fprintf(stderr, "Error: Failed to get CUDA device count: %s\n", cudaGetErrorString(error));
        return false;
    }
    
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA-capable devices found\n");
        return false;
    }
    
    // Print information about the available GPU(s)
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("GPU Device %d: \"%s\"\n", i, prop.name);
    }
    
    return true;
}

int main(int argc, char *argv[]) {
    // Check for GPU before proceeding
    if (!check_gpu()) {
        fprintf(stderr, "Exiting due to GPU check failure\n");
        return 1;
    }

    const char *input_file = "input/8192 x 5464.jpg";
    const char *output_file = "output/8192x5464_grayscale.jpg";

    convert_to_grayscale(input_file, output_file);
    return 0;
}

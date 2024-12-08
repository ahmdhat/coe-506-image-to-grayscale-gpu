#include <opencv2/opencv.hpp>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define cudaCheckError(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s (%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
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
    printf("Launching kernel with grid");

    // Create a matrix for the grayscale image
    cv::Mat grayscale_image(image.rows, image.cols, CV_8UC1);
    nvtxRangePop();

    // Allocate device memory
    unsigned char *d_input, *d_output;
    size_t input_size = image.total() * image.channels(); // Size in bytes
    size_t output_size = image.total();                  // Size in bytes (grayscale)
    
    cudaCheckError(cudaMalloc(&d_input, input_size));
    cudaCheckError(cudaMalloc(&d_output, output_size));

    // Copy input image to device memory
    cudaCheckError(cudaMemcpy(d_input, image.data, input_size, cudaMemcpyHostToDevice));

    // Configure grid and block dimensions
    dim3 block(16, 16);
    dim3 grid((image.cols + block.x - 1) / block.x, 
              (image.rows + block.y - 1) / block.y);

    // Launch the kernel
    printf("Launching kernel with grid (%d, %d) and block (%d, %d)...\n", grid.x, grid.y, block.x, block.y);
    nvtxRangePush("Convert to grayscale");
    grayscale_kernel<<<grid, block>>>(d_input, d_output, image.cols, image.rows);
    cudaCheckError(cudaDeviceSynchronize());
    nvtxRangePop();

    // Copy result back to host memory
    cudaCheckError(cudaMemcpy(grayscale_image.data, d_output, output_size, cudaMemcpyDeviceToHost));

    // Free device memory
    cudaCheckError(cudaFree(d_input));
    cudaCheckError(cudaFree(d_output));

    // Save the grayscale image
    nvtxRangePush("Save grayscale image");
    if (!cv::imwrite(output_file, grayscale_image)) {
        fprintf(stderr, "Error: Unable to save grayscale image to '%s'.\n", output_file);
    } else {
        printf("Grayscale image saved to '%s'.\n", output_file);
    }
    nvtxRangePop();
}

int main(int argc, char *argv[]) {
    const char *input_file = "input/8192 x 5464.jpg";
    const char *output_file = "output/8192x5464_grayscale.jpg";

    convert_to_grayscale(input_file, output_file);
    return 0;
}

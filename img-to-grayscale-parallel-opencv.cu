#include <opencv2/opencv.hpp>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
inline void cudaCheckError(cudaError_t err, const char* file, int line) {
    if (err != cudaSuccess) {
        printf("CUDA error in %s (%d): %s\n", file, line, cudaGetErrorString(err));
    } else {
        printf("CUDA success in %s (%d)\n", file, line);
    }
}

__global__ void grayscale(const unsigned char* input, unsigned char* output, 
               int width, int height) {

    int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;

    for (int y = 0; y < height; y++) {
        for (int x = thread_idx; x < width; x += stride) {
            int idx = (y * width + x) * 3;      // Input index (BGR)
            int out_idx = y * width + x;        // Output index (grayscale)
            
            output[out_idx] = static_cast<unsigned char>(
                0.299f * input[idx + 2] +    // R
                0.587f * input[idx + 1] +    // G
                0.114f * input[idx]);        // B
        }
    }
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

    size_t input_size = image.total() * image.channels() * sizeof(unsigned char);
    size_t output_size = image.total() * sizeof(unsigned char);
    printf("Image dimensions: %dx%d, Channels: %d\n", image.cols, image.rows, image.channels());
    printf("Input size: %zu, Output size: %zu\n", input_size, output_size);
    
    unsigned char* h_input;
    cudaCheckError(cudaMallocHost(&h_input, input_size), __FILE__, __LINE__);
    memcpy(h_input, image.data, input_size);

    unsigned char* h_output;
    cudaCheckError(cudaMallocHost(&h_output, output_size), __FILE__, __LINE__);

    unsigned char* d_input;
    unsigned char* d_output;
    cudaCheckError(cudaMalloc(&d_input, input_size), __FILE__, __LINE__);
    cudaCheckError(cudaMalloc(&d_output, output_size), __FILE__, __LINE__);
    printf("Starting cudaMemcpy to device...\n");
    cudaCheckError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), __FILE__, __LINE__);
    printf("cudaMemcpy to device completed.\n");

    int device_id = 0; // Assuming using device 0
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, device_id);

    // Print out some properties to guide your decision
    printf("Device %d: %s\n", device_id, props.name);
    printf("  SMs: %d\n", props.multiProcessorCount);
    printf("  Max threads per block: %d\n", props.maxThreadsPerBlock);
    printf("  Max threads per SM: %d\n", props.maxThreadsPerMultiProcessor);

    // Choose threads per block as the device maximum
    int threads_per_block = 512;

    // Choose a number of blocks to comfortably saturate the GPU
    // For example, launch at least several blocks per SM
    int blocks_per_grid = props.multiProcessorCount * 32;   // 32 is a magic number to saturate the GPU
    // Launch the kernel
    nvtxRangePush("Convert to grayscale");
    grayscale<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, image.cols, image.rows);
    cudaCheckError(cudaPeekAtLastError(), __FILE__, __LINE__);
    cudaCheckError(cudaDeviceSynchronize(), __FILE__, __LINE__);
    nvtxRangePop();

    cudaCheckError(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), __FILE__, __LINE__);

    memcpy(grayscale_image.data, h_output, output_size);
    cudaFree(d_input);
    cudaFree(d_output);

    // Free pinned host memory
    cudaCheckError(cudaFreeHost(h_input), __FILE__, __LINE__);
    cudaCheckError(cudaFreeHost(h_output), __FILE__, __LINE__);

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

    // const char *input_file = "input/4000 x 2667.jpg";
    // const char *output_file = "output/4000x2667_grayscale.jpg";

    convert_to_grayscale(input_file, output_file);
    return 0;
}

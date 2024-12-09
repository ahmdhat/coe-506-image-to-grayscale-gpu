#include <opencv2/opencv.hpp>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

void grayscale(const unsigned char* input, unsigned char* output, 
               int width, int height) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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

    // Launch the kernel
    nvtxRangePush("Convert to grayscale");
    grayscale(image.data, grayscale_image.data, image.cols, image.rows);
    nvtxRangePop();

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

#include <opencv2/opencv.hpp>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

// Function to convert an image to grayscale using a manual for loop
void convert_to_grayscale(const char *input_file, const char *output_file) {
    // Start profiling for loading the image
    // std::cout << cv::getBuildInformation() << std::endl;
    nvtxRangePush("Load image");
    
    // Load the image
    cv::Mat image = cv::imread(input_file, cv::IMREAD_COLOR);
    if (image.empty()) {
        printf("Error: Could not open or find the image '%s'.\n", input_file);
        return;
    }
    
    printf("Loaded image '%s' with dimensions %dx%d and %d channels.\n", 
           input_file, image.cols, image.rows, image.channels());

    // Create a new matrix for the grayscale image
    cv::Mat grayscale_image(image.rows, image.cols, CV_8UC1);
    nvtxRangePop();

    // Convert to grayscale manually
    nvtxRangePush("Convert to grayscale");
    for (int y = 0; y < image.rows; y++) {
        for (int x = 0; x < image.cols; x++) {
            // Access the pixel in the original image
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x); // BGR pixel

            // Compute the grayscale value using the formula
            unsigned char gray = static_cast<unsigned char>(
                0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0]); // RGB -> Gray
            
            // Assign the grayscale value to the new image
            grayscale_image.at<uchar>(y, x) = gray;
        }
    }
    nvtxRangePop();
    printf("Converted image to grayscale using manual loop.\n");

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

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#include <nvtx3/nvToolsExt.h>


#include <stdio.h>
#include <stdlib.h>

void convert_to_grayscale(const char *input_file, const char *output_file) {
    int width, height, channels;

    nvtxRangePush("Load image");
    // Load the image
    unsigned char *image = stbi_load(input_file, &width, &height, &channels, 0);
    if (!image) {
        fprintf(stderr, "Error: Unable to load image file '%s'.\n", input_file);
        return;
    }
    nvtxRangePop();
    printf("Loaded image '%s' with dimensions %dx%d and %d channels.\n", input_file, width, height, channels);

    // Allocate memory for the grayscale image
    unsigned char *grayscale_image = (unsigned char *)malloc(width * height);
    if (!grayscale_image) {
        fprintf(stderr, "Error: Unable to allocate memory for grayscale image.\n");
        stbi_image_free(image);
        return;
    }

    nvtxRangePush("Convert to grayscale");
    // Convert to grayscale
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = (y * width + x) * channels;
            int r = image[idx];
            int g = image[idx + 1];
            int b = image[idx + 2];
            grayscale_image[y * width + x] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    nvtxRangePop();

    nvtxRangePush("Save grayscale image");
    // Save the grayscale image
    if (!stbi_write_jpg(output_file, width, height, 1, grayscale_image, width)) {
        fprintf(stderr, "Error: Unable to save grayscale image to '%s'.\n", output_file);
    } else {
        printf("Grayscale image saved to '%s'.\n", output_file);
    }
    nvtxRangePop();

    // Free memory
    stbi_image_free(image);
    free(grayscale_image);
}

int main(int argc, char *argv[]) {
    const char *input_file = "input/4000 x 2667.jpg";
    const char *output_file = "output/4000 x 2667.jpg";

    convert_to_grayscale(input_file, output_file);
    return 0;
}

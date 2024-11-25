#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>

void convert_to_grayscale(const char *input_file, const char *output_file) {
    int width, height, channels;

    // Load the image
    unsigned char *image = stbi_load(input_file, &width, &height, &channels, 0);
    if (!image) {
        fprintf(stderr, "Error: Unable to load image file '%s'.\n", input_file);
        return;
    }
    printf("Loaded image '%s' with dimensions %dx%d and %d channels.\n", input_file, width, height, channels);

    // Allocate memory for the grayscale image
    unsigned char *grayscale_image = (unsigned char *)malloc(width * height);
    if (!grayscale_image) {
        fprintf(stderr, "Error: Unable to allocate memory for grayscale image.\n");
        stbi_image_free(image);
        return;
    }

    // Convert to grayscale
    for (int i = 0; i < width * height; i++) {
        int r = image[i * channels];
        int g = image[i * channels + 1];
        int b = image[i * channels + 2];
        // Grayscale formula: Y' = 0.299R + 0.587G + 0.114B
        grayscale_image[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }

    // Save the grayscale image
    if (!stbi_write_png(output_file, width, height, 1, grayscale_image, width)) {
        fprintf(stderr, "Error: Unable to save grayscale image to '%s'.\n", output_file);
    } else {
        printf("Grayscale image saved to '%s'.\n", output_file);
    }

    // Free memory
    stbi_image_free(image);
    free(grayscale_image);
}

int main(int argc, char *argv[]) {
    const char *input_file = "input/dorina-perry-bjWeTnbb-pg-unsplash.jpg";
    const char *output_file = "output/dorina-perry-bjWeTnbb-pg-unsplash.jpg";

    convert_to_grayscale(input_file, output_file);
    return 0;
}
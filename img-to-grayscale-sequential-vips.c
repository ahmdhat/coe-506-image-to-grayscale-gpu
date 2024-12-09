#include <vips/vips.h>
#include <nvtx3/nvToolsExt.h>
#include <stdio.h>
#include <stdlib.h>

void convert_to_grayscale(const char *input_file, const char *output_file) {
    if (VIPS_INIT(NULL)) {
        fprintf(stderr, "Error: Unable to initialize libvips.\n");
        return;
    }

    VipsImage *input_image = NULL;
    unsigned char *image_data = NULL;
    unsigned char *grayscale_data = NULL;

    nvtxRangePush("Load image");
    input_image = vips_image_new_from_file(input_file, "access", VIPS_ACCESS_SEQUENTIAL, NULL);
    if (!input_image) {
        fprintf(stderr, "Error: Unable to load image file '%s'.\n", input_file);
        vips_error_clear();
        vips_shutdown();
        return;
    }
    nvtxRangePop();

    int width = vips_image_get_width(input_image);
    int height = vips_image_get_height(input_image);
    int channels = vips_image_get_bands(input_image);


    if (channels < 3) {
        fprintf(stderr, "Error: Image must have at least 3 channels (RGB).\n");
        g_object_unref(input_image);
        vips_shutdown();
        return;
    }

    printf("Loaded image '%s' with dimensions %dx%d and %d channels.\n", input_file, width, height, channels);

    // Prepare memory for the input image data
    size_t image_size;
    void *temp_data = vips_image_write_to_memory(input_image, &image_size);
    if (!temp_data) {
        fprintf(stderr, "Error: Unable to extract image data.\n");
        g_object_unref(input_image);
        vips_shutdown();
        return;
    }
    image_data = (unsigned char *)temp_data;

    // Allocate memory for the grayscale image
    grayscale_data = (unsigned char *)malloc(width * height);
    if (!grayscale_data) {
        fprintf(stderr, "Error: Unable to allocate memory for grayscale image.\n");
        g_free(image_data);
        g_object_unref(input_image);
        vips_shutdown();
        return;
    }

    nvtxRangePush("Convert to grayscale");
    // Convert to grayscale (CPU)
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int pixel_offset = (y * width + x) * channels;
            unsigned char r = image_data[pixel_offset];
            unsigned char g = image_data[pixel_offset + 1];
            unsigned char b = image_data[pixel_offset + 2];
            grayscale_data[y * width + x] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
        }
    }
    nvtxRangePop();


    nvtxRangePush("Load grayscale image to VipsImage");
    // Create a VipsImage from the grayscale data
    VipsImage *grayscale_image = vips_image_new_from_memory(grayscale_data, width * height, width, height, 1, VIPS_FORMAT_UCHAR);
    nvtxRangePop();
    nvtxRangePush("Save grayscale image");
    if (vips_image_write_to_file(grayscale_image, output_file, NULL)) {
        fprintf(stderr, "Error: Unable to save grayscale image to '%s'.\n", output_file);
    } else {
        printf("Grayscale image saved to '%s'.\n", output_file);
    }
    nvtxRangePop();

    // Free memory
    g_free(image_data);
    free(grayscale_data);
    g_object_unref(input_image);
    g_object_unref(grayscale_image);
    vips_shutdown();
}

int main(int argc, char *argv[]) {
    const char *input_file = "input/4000 x 2667.jpg";
    const char *output_file = "output/4000 x 2667.jpg";
    convert_to_grayscale(input_file, output_file);
    return EXIT_SUCCESS;
}

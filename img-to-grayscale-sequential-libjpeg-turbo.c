#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <jpeglib.h>
#include <nvtx3/nvToolsExt.h>

void convert_to_grayscale(const char *input_file, const char *output_file) {
    int width, height, channels;

    nvtxRangePush("Load image");

    // Open input file
    int fd = open(input_file, O_RDONLY);
    if (fd < 0) {
        perror("Error: Cannot open input file");
        return;
    }

    // Get file size
    struct stat sb;
    if (fstat(fd, &sb) < 0) {
        perror("Error: Cannot get file stats");
        close(fd);
        return;
    }

    // Memory map the file
    uint8_t *file_data = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE | MAP_POPULATE, fd, 0);
    if (file_data == MAP_FAILED) {
        perror("Error: Memory mapping failed");
        close(fd);
        return;
    }

    posix_madvise(file_data, sb.st_size, POSIX_MADV_SEQUENTIAL);
    close(fd); // File descriptor can be closed after mmap
    nvtxRangePop();

    nvtxRangePush("Decompress image");
    // Initialize JPEG decompression object
    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_decompress(&cinfo);

    // Specify the source of the JPEG data
    jpeg_mem_src(&cinfo, file_data, sb.st_size);

    // Read the header
    jpeg_read_header(&cinfo, TRUE);

    // Start decompression
    jpeg_start_decompress(&cinfo);

    width = cinfo.output_width;
    height = cinfo.output_height;
    channels = cinfo.output_components;

    printf("Loaded image '%s' with dimensions %dx%d and %d channels.\n", input_file, width, height, channels);

    // Allocate memory for the image
    unsigned char *image = (unsigned char *)malloc(width * height * channels);
    if (!image) {
        fprintf(stderr, "Error: Unable to allocate memory for image.\n");
        jpeg_destroy_decompress(&cinfo);
        munmap(file_data, sb.st_size); // Free mapped memory
        return;
    }

    // Read scanlines
    while (cinfo.output_scanline < cinfo.output_height) {
        unsigned char *buffer_array[1];
        buffer_array[0] = image + (cinfo.output_scanline) * width * channels;
        jpeg_read_scanlines(&cinfo, buffer_array, 1);
    }

    // Finish decompression
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    munmap(file_data, sb.st_size); // Free mapped memory

    nvtxRangePop();

    // Allocate memory for grayscale image
    unsigned char *grayscale_image = (unsigned char *)malloc(width * height);
    if (!grayscale_image) {
        fprintf(stderr, "Error: Unable to allocate memory for grayscale image.\n");
        free(image);
        return;
    }

    nvtxRangePush("Convert to grayscale");

    // Convert to grayscale
    for (int i = 0; i < width * height; i++) {
        int r = image[i * channels];
        int g = image[i * channels + 1];
        int b = image[i * channels + 2];
        // Grayscale formula: Y' = 0.299R + 0.587G + 0.114B
        grayscale_image[i] = (unsigned char)(0.299 * r + 0.587 * g + 0.114 * b);
    }

    nvtxRangePop();

    nvtxRangePush("Save grayscale image");

    // Initialize JPEG compression object
    struct jpeg_compress_struct cjpeg;
    struct jpeg_error_mgr jerr_compress;

    cjpeg.err = jpeg_std_error(&jerr_compress);
    jpeg_create_compress(&cjpeg);

    // Open output file
    FILE *outfile = fopen(output_file, "wb");
    if (!outfile) {
        fprintf(stderr, "Error: Cannot open output file '%s'.\n", output_file);
        jpeg_destroy_compress(&cjpeg);
        free(image);
        free(grayscale_image);
        return;
    }

    jpeg_stdio_dest(&cjpeg, outfile);

    // Set parameters for compression
    cjpeg.image_width = width;
    cjpeg.image_height = height;
    cjpeg.input_components = 1; // Grayscale
    cjpeg.in_color_space = JCS_GRAYSCALE;

    jpeg_set_defaults(&cjpeg);
    jpeg_set_quality(&cjpeg, 95, TRUE);

    // Start compressor
    jpeg_start_compress(&cjpeg, TRUE);

    // Write scanlines
    while (cjpeg.next_scanline < cjpeg.image_height) {
        unsigned char *buffer_array[1];
        buffer_array[0] = &grayscale_image[cjpeg.next_scanline * width];
        jpeg_write_scanlines(&cjpeg, buffer_array, 1);
    }

    // Finish compression
    jpeg_finish_compress(&cjpeg);
    jpeg_destroy_compress(&cjpeg);
    fclose(outfile);

    printf("Grayscale image saved to '%s'.\n", output_file);

    nvtxRangePop();

    // Free memory
    free(image);
    free(grayscale_image);
}

int main(int argc, char *argv[]) {
    const char *input_file = "input/8192 x 5464.jpg";
    const char *output_file = "output/8192 x 5464.jpg";

    convert_to_grayscale(input_file, output_file);
    return 0;
}

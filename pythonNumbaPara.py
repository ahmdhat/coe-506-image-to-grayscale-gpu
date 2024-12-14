import cv2
import numpy as np
import time
from numba import cuda
from numba.cuda import nvtx

@cuda.jit
def grayscale_kernel_flat(image, grayscale_image, total_pixels):
    # Calculate global thread ID
    thread_id = cuda.grid(1)

    # Ensure thread is within bounds of total pixels
    if thread_id < total_pixels:
        # Convert flat thread ID to row and column
        row = thread_id // image.shape[1]
        col = thread_id % image.shape[1]

        b = image[row, col, 0]
        g = image[row, col, 1]
        r = image[row, col, 2]

        # Compute grayscale value
        grayscale_image[row, col] = int(0.299 * r + 0.587 * g + 0.114 * b)

def convert_to_grayscale_cuda(image_path, output_path):
    # Load the image
    nvtx.range_push("Load Image")
    start_load = time.time()
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image. Please check the path.")
        return
    load_time = time.time() - start_load
    nvtx.range_pop()  # End "Load Image"

    height, width, channels = image.shape
    print(f"Image dimensions: {width}x{height}")

    # Create an empty grayscale image on the host
    grayscale_image = np.zeros((height, width), dtype=np.uint8)

    # Allocate device memory and copy image to the GPU
    nvtx.range_push("Memory Transfer to GPU")
    start_transfer = time.time()
    image_device = cuda.to_device(image)
    grayscale_image_device = cuda.device_array((height, width), dtype=np.uint8)
    transfer_time = time.time() - start_transfer
    nvtx.range_pop()  # End "Memory Transfer to GPU"

    # Define block and grid dimensions (flat indexing)
    threads_per_block = 256
    blocks_per_grid = (height * width + threads_per_block - 1) // threads_per_block

    # Launch the kernel for grayscale conversion
    nvtx.range_push("Kernel Execution")
    start_convert = time.time()
    grayscale_kernel_flat[blocks_per_grid, threads_per_block](
        image_device, grayscale_image_device, height * width
    )
    cuda.synchronize()  # Ensure all threads are done
    convert_time = time.time() - start_convert
    nvtx.range_pop()  # End "Kernel Execution"

    # Copy the result back to the host
    nvtx.range_push("Memory Transfer to Host")
    start_copy_back = time.time()
    grayscale_image = grayscale_image_device.copy_to_host()
    copy_back_time = time.time() - start_copy_back
    nvtx.range_pop()  # End "Memory Transfer to Host"

    # Save the grayscale image
    nvtx.range_push("Save Image")
    start_save = time.time()
    cv2.imwrite(output_path, grayscale_image)
    save_time = time.time() - start_save
    nvtx.range_pop()  # End "Save Image"

    # Print timing summary
    print(f"\nSummary:")
    print(f"  Load Image Time: {load_time:.4f} seconds")
    print(f"  Transfer to GPU Time: {transfer_time:.4f} seconds")
    print(f"  Grayscale Conversion Time: {convert_time:.4f} seconds")
    print(f"  Transfer to Host Time: {copy_back_time:.4f} seconds")
    print(f"  Save Image Time: {save_time:.4f} seconds")
    print(f"Grayscale image saved to {output_path}.")

# Define input and output paths
input_image = "/content/coe-506-image-to-grayscale-gpu-main/coe-506-image-to-grayscale-gpu-main/input/8192 x 5464.jpg"
output_image = "/content/coe-506-image-to-grayscale-gpu-main/coe-506-image-to-grayscale-gpu-main/output/8192 x 5464_cuda_py_num.jpg"

# Perform CUDA grayscale conversion
convert_to_grayscale_cuda(input_image, output_image)

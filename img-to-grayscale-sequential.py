import cv2
import numpy as np
import time

def convert_to_grayscale(image_path, output_path):
    # Read the image in BGR format
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image. Please check the path.")
        return
    
    # Get the dimensions of the image
    height, width, _ = image.shape
    print(f"Image dimensions: {width}x{height}")
    
    # Create an empty array for the grayscale image
    grayscale_image = np.zeros((height, width), dtype=np.uint8)
    
    # Start the timer
    start_time = time.time()
    
    # Sequentially process each pixel
    for i in range(height):
        for j in range(width):
            # Get the BGR values
            b, g, r = image[i, j]
            # Convert to grayscale using the formula: Y' = 0.299R + 0.587G + 0.114B
            grayscale_value = int(0.299 * r + 0.587 * g + 0.114 * b)
            # Assign the value to the grayscale image
            grayscale_image[i, j] = grayscale_value
    
    # Stop the timer
    elapsed_time = time.time() - start_time
    print(f"Grayscale conversion completed in {elapsed_time:.4f} seconds.")
    
    # Save the grayscale image
    cv2.imwrite(output_path, grayscale_image)
    print(f"Grayscale image saved to {output_path}.")

input_image = "input/dorina-perry-bjWeTnbb-pg-unsplash.jpg"  # Replace with the path to your input image
output_image = "output/dorina-perry-bjWeTnbb-pg-unsplash.jpg"  # Replace with the path for the output image
convert_to_grayscale(input_image, output_image)

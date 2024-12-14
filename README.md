# Image to Grayscale Converter

This project provides implementations of an image-to-grayscale converter in Python, CUDA Python, C, and C with OpenACC for accelerated parallel processing. Each implementation processes the image sequentially or in parallel (OpenACC/CUDA), converting RGB values to grayscale using the formula:  
**Y' = 0.299R + 0.587G + 0.114B**

---

## Python Implementation

### Prerequisites
- Python 3.6 or higher
- OpenCV (cv2)
- NumPy

### Setup Environment

1. Create and activate a virtual environment (recommended):

   ```bash
   # On Windows
   python -m venv .venv
   .venv\Scripts\activate

   # On macOS/Linux
   python -m venv .venv
   . .venv/bin/activate
   ```

2. Install required packages:

   ```bash
   pip install -r requirements.txt
   ```

### Running the Python Version

1. Ensure you have an image in the `input` directory.
2. Create an `output` directory if it doesn't exist.
3. Run the script:

   ```bash
   python img-to-grayscale-sequential.py
   ```

The script will:
- Load the image from `input/dorina-perry-bjWeTnbb-pg-unsplash.jpg`
- Convert it to grayscale
- Save the result to `output/dorina-perry-bjWeTnbb-pg-unsplash.jpg`
- Display processing time and image dimensions

---

## CUDA Python Implementation

### Prerequisites
- Python 3.6 or higher
- OpenCV (cv2)
- NumPy
- Numba with CUDA support

### Setup Environment

1. Install Numba:

   ```bash
   pip install numba
   ```

2. Ensure a CUDA-compatible GPU is installed and the necessary drivers are configured.

### Running the CUDA Python Version

1. Ensure you have an image in the `input` directory.
2. Create an `output` directory if it doesn't exist.
3. Run the script:

   ```bash
   python grayscale_cuda.py
   ```

The script will:
- Load the image from `input/8192x5464.jpg`
- Convert it to grayscale using GPU parallelization
- Save the result to `output/8192x5464_cuda_py_num.jpg`
- Print profiling information for each step (loading, transfer, kernel execution, etc.)

---

## C Implementation

### Prerequisites
- GCC or NVCC compiler
- OpenCV library
- stb_image.h and stb_image_write.h (already included in the project)

### Compilation and Running

#### Original Sequential
1. Compile the C program:

   ```bash
   gcc -o img-grayscale-sequential img-to-grayscale-sequential.c -lm
   ```

2. Run the compiled program:

   ```bash
   ./img-grayscale-sequential
   ```

#### Sequential with OpenCV
1. Compile the C program:

   ```bash
   g++ -O2 -o img-grayscale-opencv img-to-grayscale-opencv.c `pkg-config --cflags --libs opencv4`
   ```

2. Run the compiled program:

   ```bash
   ./img-grayscale-opencv
   ```

---

## OpenACC Implementation

### Prerequisites
- GCC or NVCC compiler with OpenACC support
- OpenCV library
- NVTX library for profiling

### Compilation and Running

1. Compile the OpenACC-enabled program:

   ```bash
   g++ -o img-grayscale-openacc img-to-grayscale-openacc.c -fopenacc -lm `pkg-config --cflags --libs opencv4`
   ```

2. Run the program:

   ```bash
   ./img-grayscale-openacc
   ```

### Features
- Uses OpenACC directives for GPU acceleration.
- Transfers data between CPU and GPU for efficient processing.
- Outputs grayscale image to the `output` directory.

### Example Output
The OpenACC program will:
- Load the input image from `input/8192x5464.jpg`.
- Convert it to grayscale using GPU parallelization.
- Save the result to `output/8192x5464_grayscale.jpg`.
- Print profiling information for kernel execution.

---

## Directory Structure

Ensure the following directory structure:

```plaintext
.
├── input/                      # Place your input images here
├── output/                     # Grayscale images will be saved here
├── grayscale_cuda.py           # CUDA Python implementation
├── img-to-grayscale-sequential.py
├── img-to-grayscale-sequential.c
├── img-to-grayscale-opencv.c
├── img-to-grayscale-openacc.c
├── stb_image.h
└── stb_image_write.h
```

---

## Modifying Input/Output Paths

To use different images:

1. **In Python version** - modify these lines at the bottom of the script:

   ```python
   input_image = "input/your-image.jpg"
   output_image = "output/your-output.jpg"
   ```

2. **In CUDA Python version** - modify these lines at the bottom of the script:

   ```python
   input_image = "input/your-image.jpg"
   output_image = "output/your-output.jpg"
   ```

3. **In C/OpenACC versions** - modify these lines in `main()`:

   ```c
   const char *input_file = "input/your-image.jpg";
   const char *output_file = "output/your-output.jpg";
   ```

---

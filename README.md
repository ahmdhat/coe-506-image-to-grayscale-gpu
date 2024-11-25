# Image to Grayscale Converter

This project provides implementations of an image to grayscale converter in both Python and C. Each implementation processes the image sequentially, converting RGB values to grayscale using the formula: Y' = 0.299R + 0.587G + 0.114B.

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

## C Implementation

### Prerequisites
- GCC compiler
- stb_image.h and stb_image_write.h (already included in the project)

### Compilation and Running

1. First, compile the C program:

```bash
# On Linux/macOS
gcc -o img-grayscale-sequential-C img-to-grayscale-sequential.c -lm

# On Windows with MinGW
gcc -o img-grayscale-sequential-C img-to-grayscale-sequential.c -lm
```

2. Run the compiled program:

```bash
# On Linux/macOS
./img-grayscale-sequential-C

# On Windows
img-grayscale-sequential-C
```

The C program will:
- Read the input image from `input/dorina-perry-bjWeTnbb-pg-unsplash.jpg`
- Convert it to grayscale
- Save the result as PNG to `output/dorina-perry-bjWeTnbb-pg-unsplash.jpg`

### Directory Structure
Ensure you have the following directory structure:

```plaintext
.
├── input/                      # Place your input images here
├── output/                     # Grayscale images will be saved here
├── img-to-grayscale-sequential.py
├── img-to-grayscale-sequential.c
├── stb_image.h
└── stb_image_write.h
```

### Modifying Input/Output Paths

To use different images:

1. **In Python version** - modify these lines at the bottom of the script:

```python
input_image = "input/your-image.jpg"
output_image = "output/your-output.jpg"
```

2. **In C version** - modify these lines in `main()`:

```c
const char *input_file = "input/your-image.jpg";
const char *output_file = "output/your-output.jpg";
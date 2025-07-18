# Circelizer

A Python library for detecting circles in images (like manhole covers), centering them, and cropping to square format.

## Features

- 🔍 **Circle Detection**: Uses OpenCV's Hough Circle Transform to detect circles in images
- 🎯 **Auto-centering**: Automatically centers the detected circle in the image
- ✂️ **Square Cropping**: Crops images to square format with the circle centered
- 📁 **Batch Processing**: Process multiple images in a directory
- 🖥️ **CLI Interface**: Easy-to-use command-line interface

## Installation

This project uses `uv` for package management. Make sure you have `uv` installed, then:

```bash
# Clone the repository
git clone <your-repo-url>
cd circelizer

# Install dependencies
uv sync
```

## Usage

### Command Line Interface

The easiest way to use circelizer is through the command-line interface:

```bash
# Process all JPG images in a directory
circelizer /path/to/input/images /path/to/output/directory

# With verbose logging
circelizer -v /path/to/input/images /path/to/output/directory
```

### Python API

You can also use the library programmatically:

```python
from circelizer import process_images

# Process all images in a directory
stats = process_images("/path/to/input", "/path/to/output")
print(f"Processed {stats['processed']} out of {stats['total']} images")
```

### Example Script

```python
#!/usr/bin/env python3
import sys
from circelizer import process_images

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py <input_path> <output_path>")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    stats = process_images(input_path, output_path)
    print(f"Processing complete: {stats}")

if __name__ == "__main__":
    main()
```

## How It Works

1. **Image Scanning**: Recursively finds all JPG images in the input directory
2. **Circle Detection**: Uses OpenCV's Hough Circle Transform to detect circles
3. **Centering**: Calculates the optimal crop area to center the detected circle
4. **Cropping**: Crops the image to a square format with the circle centered
5. **Saving**: Saves the processed image to the output directory

## Supported Image Formats

- JPG/JPEG (case-insensitive)
- Images are automatically converted to JPG format in the output

## Requirements

- Python 3.13+
- OpenCV (opencv-python)
- NumPy
- Pillow

## Development

To set up the development environment:

```bash
# Install development dependencies
uv sync --extra dev

# Run tests (when implemented)
pytest

# Format code
black .

# Lint code
flake8
```

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]

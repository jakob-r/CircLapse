# Circelizer

A Python library for detecting circles and ellipses in images (like manhole covers), centering them, and cropping to square format.

## Features

- 🔍 **Circle Detection**: Uses openCV circle detection.
- 🔄 **Ellipse Detection**: Uses scikit-image Hough ellipse detection with automatic transformation to circles.
- 🎯 **Auto-scaling**
 * All circles will have the same size.
 * Crops the picture as little as possible.
 * Output resolution unified to ensure as little as possible upscaling.
- 📁 **Batch Processing**: Process multiple images in a directory
- 🖥️ **CLI Interface**: Easy-to-use command-line interface with method selection

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

After installation, you can use circelizer from the command line:

### Basic Usage

```bash
uv run python -m circelizer.cli examples/example_in examples/example_out
```

### Detection Methods

Circelizer supports multiple detection methods:

- `hough` (default): OpenCV Hough circle detection
- `ellipse`: Scikit-image Hough ellipse detection with automatic transformation to circles
- `contour`: Contour-based circle detection
- `ransac`: RANSAC-based circle detection
- `gradient`: Gradient-based circle detection

Example with ellipse detection:
```bash
uv run python -m circelizer.cli examples/example_in examples/example_out --detection-method ellipse
```

For more commands check
```bash
uv run  python -m circelizer.cli
```
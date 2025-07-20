# Circlapse

A Python library for detecting circles and ellipses in images (like manhole covers), centering them, and cropping to square format.

<p align="center">
  <img src="examples/example_in/1.jpg" alt="input 1" width="80">
  <img src="examples/example_in/2.jpg" alt="input 2" width="80">
  <img src="examples/example_in/3.jpg" alt="input 3" width="80">
  <img src="examples/example_in/4.jpg" alt="input 4" width="80">
  <img src="examples/example_in/5.jpg" alt="input 5" width="80">
</p>

![Example Output Animation](examples/example_out/all.gif)

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
cd circlapse

# Install dependencies
uv sync
```

## Usage

After installation, you can use circlapse from the command line:

### Basic Usage

```bash
uv run python -m circlapse.cli examples/example_in examples/example_out
```

For more commands check
```bash
uv run  python -m circlapse.cli
```

### Convert to mp4

I recommend sorting out wrong crops and then using ffmpeg.

Example:

```bash
# random filenames for random order
set i 1; for file in (ls *.jpg | shuf); set new_name (printf "%04d.jpg" $i); mv $file $new_name; set i (math "$i + 1"); end
# ffmpeg with slight sharpening and 12fps
ffmpeg -framerate 12 -i %04d.jpg -vf "scale=1080:1080,unsharp=5:5:1.0:3:3:0.5" -c:v libx264 -crf 18 output.mp4
```


## Disclaimer

This is a hobby project and most of the code is AI generated.

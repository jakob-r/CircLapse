"""
Circelizer - A library for detecting circles in images and centering them.

This library scans for images containing circles (like manhole covers),
detects the circle, centers it, crops to square, and saves the result.
"""

from .processor import process_images

__version__ = "0.1.0"
__all__ = ["process_images"]

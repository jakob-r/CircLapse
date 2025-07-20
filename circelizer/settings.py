"""
Settings for the circelizer package.
"""

import os

# Debug mode - enables saving debug images
DEBUG = True

# Circle detection method to use
# Options: 'hough', 'contour', 'ransac', 'gradient', 'ensemble'
DETECTION_METHOD = os.getenv("CIRCELIZER_DETECTION_METHOD", "circle")

# Available detection methods mapping
DETECTION_METHODS = {
    "hough": "detect_circle",
    "contour": "detect_circle_contour_based",
    "ransac": "detect_circle_ransac",
    "gradient": "detect_circle_gradient_based",
}

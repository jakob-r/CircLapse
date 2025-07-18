import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from circelizer import settings
from circelizer.context import output_dir

logger = logging.getLogger(__name__)


def center_and_crop_image(image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
    """
    Center the image on the detected circle and crop to square.
    
    Args:
        image: Input image
        circle: Tuple of (x, y, radius) of the detected circle
        
    Returns:
        Centered and cropped square image
    """
    x, y, radius = circle
    height, width = image.shape[:2]
    
    # Calculate the size of the square crop (2 * radius + some padding)
    crop_size = min(2 * radius + 50, min(width, height))
    
    # Calculate crop boundaries to center the circle
    crop_x = max(0, min(x - crop_size // 2, width - crop_size))
    crop_y = max(0, min(y - crop_size // 2, height - crop_size))
    
    # Ensure we don't go out of bounds
    crop_x = max(0, min(crop_x, width - crop_size))
    crop_y = max(0, min(crop_y, height - crop_size))
    
    # Crop the image
    cropped = image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    
    return cropped


def center_and_crop_image_consistent(image: np.ndarray, circle: Tuple[int, int, int], target_circle_share: float) -> np.ndarray:
    """
    Center the image on the detected circle and crop to a square with a consistent radius.
    Scales the image so that all circles have the same size in the output.
    
    Args:
        image: Input image
        circle: Tuple of (x, y, radius) of the detected circle
        target_circle_share: How much of the image should be used for the circle. 1 means all the image is used for the circle.
        
    Returns:
        The image with a circle of the target radius and the target ratio
    """
    x, y, radius = circle

    # if the radius is 100, the circle_share is 0.8, the image has to be 100 * 1/0.8 big
    final_shortest_side = int(radius / target_circle_share) * 2

    crop_x = x - (final_shortest_side // 2)
    crop_y = y - (final_shortest_side // 2)
    
    return image[crop_y:crop_y + final_shortest_side, crop_x:crop_x + final_shortest_side]


def output_width(shortest_sides: list[int]) -> int:
    """
    Calculate the unified output width based on all shortest sides.
    
    Args:
        shortest_sides: List of shortest side lengths from all images
        
    Returns:
        The minimum shortest side length to use as output width
    """
    return min(shortest_sides)


def scale_image(image: np.ndarray, target_width: int) -> np.ndarray:
    """
    Scale an image to a target width while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_width: The desired width for the output image
        
    Returns:
        Scaled image with the target width
    """
    height, width = image.shape[:2]
    
    # Calculate scale factor to achieve target width
    scale_factor = target_width / width
    
    # Calculate new height to maintain aspect ratio
    new_height = int(height * scale_factor)
    
    # Resize the image
    scaled_image = cv2.resize(image, (target_width, new_height))
    
    return scaled_image

def circle_share(image: np.ndarray, circle: tuple[int, int, int]) -> float:
    """
    Calculate the share of the image that is covered by the circle.
    """
    x, y, radius = circle
    height, width = image.shape[:2]
    shortest_side = min(height, width)
    distance_to_left = x - radius
    distance_to_right = width - (x + radius)
    distance_to_top = y - radius
    distance_to_bottom = height - (y + radius)
    shortest_distance = max(0, min(distance_to_left, distance_to_right, distance_to_top, distance_to_bottom))
    return radius / (radius + shortest_distance)


def crop_image(image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    """
    Crop an image with out-of-bounds support, filling with black pixels.
    
    Args:
        image: Input image
        x: Starting x coordinate (can be negative)
        y: Starting y coordinate (can be negative)
        width: Width of the crop region
        height: Height of the crop region
        
    Returns:
        Cropped image with black padding for out-of-bounds areas
    """
    img_height, img_width = image.shape[:2]
    channels = image.shape[2] if len(image.shape) > 2 else 1
    
    # Create output image filled with black
    output = np.zeros((height, width, channels), dtype=image.dtype)
    
    # Calculate the actual crop region within the image bounds
    crop_x_start = max(0, x)
    crop_y_start = max(0, y)
    crop_x_end = min(img_width, x + width)
    crop_y_end = min(img_height, y + height)
    
    # Calculate the corresponding region in the output image
    output_x_start = max(0, -x)
    output_y_start = max(0, -y)
    output_x_end = output_x_start + (crop_x_end - crop_x_start)
    output_y_end = output_y_start + (crop_y_end - crop_y_start)
    
    # Copy the valid region from input to output
    if crop_x_start < crop_x_end and crop_y_start < crop_y_end:
        output[output_y_start:output_y_end, output_x_start:output_x_end] = \
            image[crop_y_start:crop_y_end, crop_x_start:crop_x_end]
    
    return output
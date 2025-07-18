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


def center_and_crop_image_consistent(image: np.ndarray, circle: Tuple[int, int, int], target_radius: int) -> np.ndarray:
    """
    Center the image on the detected circle and crop to a square with a consistent radius.
    Scales the image so that all circles have the same size in the output.
    
    Args:
        image: Input image
        circle: Tuple of (x, y, radius) of the detected circle
        target_radius: The desired radius for the cropped image
        
    Returns:
        Centered and cropped square image with consistent radius
    """
    x, y, radius = circle
    height, width = image.shape[:2]
    
    # Calculate scale factor to make this circle match the target radius
    scale_factor = target_radius / radius
    
    # Scale the image and circle coordinates
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_x = int(x * scale_factor)
    new_y = int(y * scale_factor)
    
    # Resize the image
    scaled_image = cv2.resize(image, (new_width, new_height))
    
    # Calculate crop size (2 * target_radius + padding)
    crop_size = min(2 * target_radius + 50, min(new_width, new_height))
    
    # Calculate crop boundaries to center the circle
    crop_x = max(0, min(new_x - crop_size // 2, new_width - crop_size))
    crop_y = max(0, min(new_y - crop_size // 2, new_height - crop_size))
    
    # Ensure we don't go out of bounds
    crop_x = max(0, min(crop_x, new_width - crop_size))
    crop_y = max(0, min(crop_y, new_height - crop_size))
    
    # Crop the image
    cropped = scaled_image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    
    return cropped


def center_and_crop_image_relative(image: np.ndarray, circle: Tuple[int, int, int], max_ratio: float, unified_width: int) -> np.ndarray:
    """
    Center the image on the detected circle and crop to show the same relative amount of the circle.
    Uses the maximum radius-to-image-size ratio to determine how much of each circle to show.
    Ensures all output images have the same unified width.
    
    Args:
        image: Input image
        circle: Tuple of (x, y, radius) of the detected circle
        max_ratio: The maximum radius-to-image-size ratio found across all images
        unified_width: The minimum shortest side length to use as output width
        
    Returns:
        Centered and cropped square image showing consistent relative circle size with unified dimensions
    """
    x, y, radius = circle
    height, width = image.shape[:2]
    shortest_side = min(height, width)
    
    # Calculate the target radius for this image based on the max ratio
    target_radius = int(shortest_side * max_ratio)
    
    # Calculate scale factor to make this circle match the target radius
    scale_factor = target_radius / radius
    
    # Scale the image and circle coordinates
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    new_x = int(x * scale_factor)
    new_y = int(y * scale_factor)
    
    # Resize the image
    scaled_image = cv2.resize(image, (new_width, new_height))
    
    # Calculate crop size to show the target radius with some padding, but ensure it doesn't exceed unified_width
    crop_size = min(2 * target_radius + 50, min(new_width, new_height), unified_width)
    
    # Calculate crop boundaries to center the circle
    crop_x = max(0, min(new_x - crop_size // 2, new_width - crop_size))
    crop_y = max(0, min(new_y - crop_size // 2, new_height - crop_size))
    
    # Ensure we don't go out of bounds
    crop_x = max(0, min(crop_x, new_width - crop_size))
    crop_y = max(0, min(crop_y, new_height - crop_size))
    
    # Crop the image
    cropped = scaled_image[crop_y:crop_y + crop_size, crop_x:crop_x + crop_size]
    
    # Resize to unified width if necessary
    if cropped.shape[0] != unified_width or cropped.shape[1] != unified_width:
        cropped = cv2.resize(cropped, (unified_width, unified_width))
    
    return cropped


def output_width(shortest_sides: list[int]) -> int:
    """
    Calculate the unified output width based on all shortest sides.
    
    Args:
        shortest_sides: List of shortest side lengths from all images
        
    Returns:
        The minimum shortest side length to use as output width
    """
    return min(shortest_sides)
import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from circelizer import image_operators, settings
from circelizer.context import output_dir

logger = logging.getLogger(__name__)

def detect_circle(image: np.ndarray, image_name: str = "debug", target_size: int = 800, min_distance_to_border: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """
    Detect the largest circle in the image.
    
    Args:
        image: Input image as numpy array
        image_name: Name for debug image (used when DEBUG=True)
        target_size: Target size for the longest side (default: 800px)
        
    Returns:
        Tuple of (x, y, radius) of the detected circle, or None if no circle found
    """
    # Scale image to consistent size for better circle detection
    height, width = image.shape[:2]
    scale_factor = target_size / max(height, width)
    
    # scale image to target size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    shortest_side = min(new_width, new_height)
    image = cv2.resize(image, (new_width, new_height))
    logger.debug(f"Scaled image from {width}x{height} to {new_width}x{new_height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Circle Transform
    # Parameters optimized for ~800px images
    circles_raw = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,   # Minimum distance between circles
        param1=30,    # Edge detection threshold
        param2=70,    # Accumulator threshold - adjusted for scaled images
        minRadius=shortest_side // 6, # Minimum radius - adjusted for scaled images
        maxRadius=shortest_side // 2 # Maximum radius - adjusted for scaled images
    )

    circles = circles_raw
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Return the largest circle (assuming it's the main object)
        largest_circle = max(circles, key=lambda x: x[2])

                # Save debug image if DEBUG is enabled
        if settings.DEBUG:
            debug_image = image.copy()
            for (x, y, radius) in circles:
                # Draw circle outline
                cv2.circle(debug_image, (x, y), radius, (0, 255, 0), 2)
                # Draw center point
                cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)

            # draw biggest circle
            cv2.circle(debug_image, (largest_circle[0], largest_circle[1]), largest_circle[2], (255, 255, 255), 2)
            
            # Save debug image if output directory is available
            debug_path = output_dir.get() / "debug" / f"{image_name}_circles.jpg"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), debug_image)
            logger.debug(f"Saved debug image: {debug_path}")
        
        # Scale coordinates back to original image size if image was scaled
        if scale_factor < 1.0:
            x, y, radius = largest_circle
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)
            original_radius = int(radius / scale_factor)
            return (original_x, original_y, original_radius)
        
        return tuple(largest_circle)
    
    return None
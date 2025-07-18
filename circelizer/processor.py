"""
Image processor for detecting circles and centering them.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from circelizer import settings
from circelizer.context import output_dir

logger = logging.getLogger(__name__)


def detect_circle(image: np.ndarray, image_name: str = "debug", target_size: int = 800) -> Optional[Tuple[int, int, int]]:
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
    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=80,   # Minimum distance between circles
        param1=25,    # Edge detection threshold
        param2=70,    # Accumulator threshold - adjusted for scaled images
        minRadius=shortest_side // 4, # Minimum radius - adjusted for scaled images
        maxRadius=shortest_side # Maximum radius - adjusted for scaled images
    )
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        
        # Save debug image if DEBUG is enabled
        if settings.DEBUG:
            debug_image = image.copy()
            for (x, y, radius) in circles:
                # Draw circle outline
                cv2.circle(debug_image, (x, y), radius, (0, 255, 0), 2)
                # Draw center point
                cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
            
            # Save debug image if output directory is available
            debug_path = output_dir.get() / "debug" / f"{image_name}_circles.jpg"
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_path), debug_image)
            logger.debug(f"Saved debug image: {debug_path}")
        
        # Return the largest circle (assuming it's the main object)
        largest_circle = max(circles, key=lambda x: x[2])
        
        # Scale coordinates back to original image size if image was scaled
        if scale_factor < 1.0:
            x, y, radius = largest_circle
            original_x = int(x / scale_factor)
            original_y = int(y / scale_factor)
            original_radius = int(radius / scale_factor)
            return (original_x, original_y, original_radius)
        
        return tuple(largest_circle)
    
    return None


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


def process_single_image(image_path: Path) -> bool:
    """
    Process a single image: detect circle, center it, crop to square, and save.
    
    Args:
        image_path: Path to input image
        
    Returns:
        True if processing was successful, False otherwise
    """
    try:
        # Read the image
        image = cv2.imread(str(image_path))
        if image is None:
            logger.error(f"Could not read image: {image_path}")
            return False
        
        # Detect circle
        image_name = image_path.stem
        circle = detect_circle(image, image_name)
        if circle is None:
            logger.warning(f"No circle detected in {image_path}")
            return False
        
        # Center and crop the image
        processed_image = center_and_crop_image(image, circle)
                
        # Save the processed image
        output_path = output_dir.get() / image_path.name
        cv2.imwrite(str(output_path), processed_image)
        logger.info(f"Successfully processed {image_path} -> {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {image_path}: {repr(e)}")
        return False


def process_images(input_path: str, output_path: str) -> dict:
    """
    Process all JPG images in the input path.
    
    Args:
        input_path: Directory containing images to process
        output_path: Directory to save processed images
        
    Returns:
        Dictionary with processing statistics
    """
    input_dir = Path(input_path)
    output_dir.set(Path(output_path))
    
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")
    
    # Create output directory
    output_dir.get().mkdir(parents=True, exist_ok=True)
    
    # Find all JPG images
    image_extensions = {'.jpg', '.jpeg', '.JPG', '.JPEG'}
    image_files = [
        f for f in input_dir.rglob('*') 
        if f.is_file() and f.suffix in image_extensions
    ]
    
    if not image_files:
        logger.warning(f"No JPG images found in {input_path}")
        return {"total": 0, "processed": 0, "failed": 0}
    
    logger.info(f"Found {len(image_files)} images to process")
    
    
    processed_count = 0
    failed_count = 0
    
    for image_file in image_files:
        if process_single_image(image_file):
            processed_count += 1
        else:
            failed_count += 1
    
    stats = {
        "total": len(image_files),
        "processed": processed_count,
        "failed": failed_count
    }
    
    logger.info(f"Processing complete: {stats}")
    return stats 
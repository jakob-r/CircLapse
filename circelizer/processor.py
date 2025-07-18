"""
Image processor for detecting circles and centering them.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from circelizer import settings, detector, image_operators
from circelizer.context import output_dir

logger = logging.getLogger(__name__)

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
        return {"total": 0, "processed": 0, "failed": 0, "no_circles": 0}
    
    logger.info(f"Found {len(image_files)} images to process")
    
    # Step 1: Detect circles for all images
    circle_data = {}
    no_circles_count = 0
    
    for image_file in image_files:
        try:
            # Read the image
            image = cv2.imread(str(image_file))
            if image is None:
                logger.error(f"Could not read image: {image_file}")
                continue
            
            # Detect circle
            image_name = image_file.stem
            circle = detector.detect_circle(image, image_name)
            
            if circle is None:
                logger.warning(f"No circle detected in {image_file}")
                no_circles_count += 1
            else:
                circle_data[image_file] = (image, circle)
                
        except Exception as e:
            logger.error(f"Error detecting circle in {image_file}: {repr(e)}")
    
    if not circle_data:
        logger.warning("No circles detected in any images")
        return {"total": len(image_files), "processed": 0, "failed": 0, "no_circles": len(image_files)}
    
    # Step 2: Find how much space we have to work with
    min_boarders = [image_operators.min_distance_to_border(image, circle) for image, circle in circle_data.values()]
    min_boarder = min(min_boarders)
    logger.info(f"Using minimum relative border: {min_boarder}")

    min_radius = min(circle[2] for _, circle in circle_data.values())
    logger.info(f"Using minimum radius: {min_radius}")

    
    # Step 3: Process images with detected circles
    processed_count = 0
    failed_count = 0
    unified_width = 400
    
    for image_file, (image, circle) in circle_data.items():
        try:
            # Center and crop the image with consistent relative circle size and unified output width
            processed_image = image_operators.center_and_crop_image_consistent(image, circle, 1)

            # scaled_image = image_operators.scale_image(processed_image, unified_width)
            scaled_image = processed_image
            
            # Save the processed image
            output_path = output_dir.get() / image_file.name
            cv2.imwrite(str(output_path), scaled_image)
            logger.info(f"Successfully processed {image_file} -> {output_path}")
            processed_count += 1
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {repr(e)}")
            failed_count += 1
    
    stats = {
        "total": len(image_files),
        "processed": processed_count,
        "failed": failed_count,
        "no_circles": no_circles_count,
        "unified_width": unified_width
    }
    
    logger.info(f"Processing complete: {stats}")
    return stats 
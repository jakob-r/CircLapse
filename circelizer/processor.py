"""
Image processor for detecting circles and centering them.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
from circelizer import settings, detector, image_operators, saving
from circelizer.context import output_dir

logger = logging.getLogger(__name__)

def process_images(input_path: str, output_path: str, output_format: str = 'jpg') -> dict:
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
        f for f in input_dir.iterdir()
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
            
            # Detect circle using configured method
            image_name = image_file.stem
            method_name = settings.DETECTION_METHODS[settings.DETECTION_METHOD]
            method_func = getattr(detector, method_name)
            circle = method_func(image, image_name)
            
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
    circle_shares = [image_operators.circle_share(image, circle) for image, circle in circle_data.values()]
    max_circle_share = max(circle_shares)
    logger.info(f"Using maximum circle share: {max_circle_share}")

    
    # Step 3: Process images with detected circles
    processed_images = []
    processed_filenames = []
    failed_count = 0
    unified_width = 400
    
    for image_file, (image, circle) in circle_data.items():
        try:
            # Center and crop the image with consistent relative circle size and unified output width
            processed_image = image_operators.center_and_crop_image_consistent(image, circle, max_circle_share)

            scaled_image = image_operators.scale_image(processed_image, unified_width)
            
            # Collect processed images and filenames for batch saving
            processed_images.append(scaled_image)
            processed_filenames.append(image_file)
            
        except Exception as e:
            logger.error(f"Error processing {image_file}: {repr(e)}")
            failed_count += 1
    
    # Save all processed images
    if output_format == 'jpg':
        processed_count = saving.save_jpg(processed_images, processed_filenames)
    elif output_format == 'gif':
        processed_count = saving.save_gif(processed_images)
    else:
        raise ValueError(f"Invalid output format: {output_format}")
    
    stats = {
        "total": len(image_files),
        "processed": processed_count,
        "failed": failed_count,
        "no_circles": no_circles_count,
        "unified_width": unified_width,
        "max_circle_share": max_circle_share
    }
    
    logger.info(f"Processing complete: {stats}")
    return stats 
"""
Image processor for detecting circles and centering them.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import logging
import tempfile
import shutil
from circelizer import settings, detector, image_operators, saving
from circelizer.context import output_dir
from circelizer.disk_image import DiskImage

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
    
    # Create temporary directory for intermediate images
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Step 1: Detect circles and store intermediate images on disk
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
                if settings.DETECTION_METHOD == 'ellipse':
                    res_image, circle = detector.detect_ellipse_and_transform(image, image_name)
                else:
                    res_image, circle = detector.detect_circle(image, image_name)
                
                if circle is None:
                    logger.warning(f"No circle detected in {image_file}")
                    no_circles_count += 1
                else:
                    # Store intermediate image on disk using clean abstraction
                    disk_image = DiskImage(res_image, image_name, temp_path)
                    circle_data[image_file] = (disk_image, circle)
                    
            except Exception as e:
                logger.error(f"Error detecting circle in {image_file}: {repr(e)}")
        
        if not circle_data:
            logger.warning("No circles detected in any images")
            return {"total": len(image_files), "processed": 0, "failed": 0, "no_circles": len(image_files)}
        
        # Step 2: Find how much space we have to work with
        circle_shares = []
        for disk_image, circle in circle_data.values():
            # Load image temporarily to calculate circle share
            image = disk_image.load()
            circle_shares.append(image_operators.circle_share(image, circle))
        
        max_circle_share = max(circle_shares)
        logger.info(f"Using maximum circle share: {max_circle_share}")
        
        # Step 3: Process images with detected circles (streaming approach)
        processed_count = 0
        failed_count = 0
        processed_disk_images = []  # Store DiskImage objects instead of numpy arrays
        
        # Calculate unified width from all intermediate images efficiently
        intermediate_images = [disk_image for disk_image, _ in circle_data.values()]
        unified_width = image_operators.output_width(intermediate_images)
        
        for image_file, (disk_image, circle) in circle_data.items():
            try:
                # Load image from disk (clean abstraction)
                image = disk_image.load()
                
                # Center and crop the image with consistent relative circle size and unified output width
                processed_image = image_operators.center_and_crop_image_consistent(image, circle, max_circle_share)
                
                scaled_image = image_operators.scale_image(processed_image, unified_width)
                
                # Apply automatic post-processing enhancements
                enhanced_image = image_operators.automatic_postprocess(scaled_image)
                
                
                processed_disk_image = DiskImage(enhanced_image, f"{image_file.stem}_processed", temp_path)
                processed_disk_images.append(processed_disk_image)
                
            except Exception as e:
                logger.error(f"Error processing {image_file}: {repr(e)}")
                failed_count += 1
        
        # Save GIF if that's the output format
        if output_format == 'gif':
            if processed_disk_images:
                # Load all processed images from disk for GIF creation
                processed_images = [disk_image.load() for disk_image in processed_disk_images]
                success = saving.save_gif(processed_images)
                processed_count = len(processed_images) if success else 0
            else:
                processed_count = 0
        elif output_format == 'jpg':
            for processed_disk_image in processed_disk_images:
                output_file_path = output_dir.get() / f"{processed_disk_image.filename}.jpg"
                processed_disk_image.save(output_file_path)
                processed_count += 1
                logger.info(f"Saved processed image: {output_file_path}")
        elif output_format != 'jpg':
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
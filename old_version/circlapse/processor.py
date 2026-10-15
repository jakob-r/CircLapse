"""
Image processor for detecting circles and centering them.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Union

import cv2
import numpy as np

from circlapse import detector, image_operators, saving, settings
from circlapse.context import output_dir
from circlapse.disk_image import DiskImage

logger = logging.getLogger(__name__)


def process_images(
    input_path: str, output_path: str, output_format: str = "jpg"
) -> dict:
    """
    Process a batch of images to detect circles, center them, and save the results.

    Args:
        input_path: Path to input directory or file
        output_path: Path to output directory
        output_format: Output format ("jpg" or "gif")

    Returns:
        Dictionary with processing statistics
    """
    # Ensure output directory exists
    output_dir.set(Path(output_path))
    output_path_obj = Path(output_path)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    # Get input files
    input_path_obj = Path(input_path)
    if input_path_obj.is_file():
        image_files = [input_path_obj]
    else:
        image_files = (
            list(input_path_obj.glob("*.jpg"))
            + list(input_path_obj.glob("*.jpeg"))
            + list(input_path_obj.glob("*.png"))
        )

    if not image_files:
        logger.warning(f"No image files found in {input_path}")
        return {
            "total": 0,
            "processed": 0,
            "failed": 0,
            "no_circles": 0,
        }

    logger.info(f"Processing {len(image_files)} images")

    # Create temporary directory for intermediate files
    temp_path = output_path_obj / "temp"
    temp_path.mkdir(exist_ok=True)

    # Step 1: Detect circles in all images and store intermediate results
    circle_data: Dict[Path, Tuple[DiskImage, Tuple[int, int, int]]] = {}
    no_circles_count = 0

    for image_file in image_files:
        try:
            # Read the image
            image = cv2.imread(str(image_file))

            # Detect circle using configured method
            image_name = image_file.stem
            if settings.DETECTION_METHOD == "ellipse":
                res_image, circle = detector.detect_ellipse_and_transform(
                    image, image_name
                )
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
        return {
            "total": len(image_files),
            "processed": 0,
            "failed": 0,
            "no_circles": len(image_files),
        }

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
    processed_disk_images: List[
        DiskImage
    ] = []  # Store DiskImage objects instead of numpy arrays

    # Calculate unified width from all intermediate images efficiently
    intermediate_images: List[Union[np.ndarray, DiskImage]] = [
        disk_image for disk_image, _ in circle_data.values()
    ]
    unified_width = image_operators.output_width(intermediate_images)

    for image_file, (disk_image, circle) in circle_data.items():
        try:
            # Load image from disk (clean abstraction)
            image = disk_image.load()

            # Center and crop the image with consistent relative circle size and unified output width
            processed_image = image_operators.center_and_crop_image_consistent(
                image, circle, max_circle_share
            )

            scaled_image = image_operators.scale_image(processed_image, unified_width)

            # Apply automatic post-processing enhancements
            enhanced_image = image_operators.automatic_postprocess(scaled_image)

            processed_disk_image = DiskImage(
                enhanced_image, f"{image_file.stem}_processed", temp_path
            )
            processed_disk_images.append(processed_disk_image)

        except Exception as e:
            logger.error(f"Error processing {image_file}: {repr(e)}")
            failed_count += 1

    # Save GIF if that's the output format
    if output_format == "gif":
        if processed_disk_images:
            # Load all processed images from disk for GIF creation
            processed_images = [
                disk_image.load() for disk_image in processed_disk_images
            ]
            success = saving.save_gif(processed_images)
            processed_count = len(processed_images) if success else 0
        else:
            processed_count = 0
    elif output_format == "jpg":
        for processed_disk_image in processed_disk_images:
            output_file_path = output_dir.get() / f"{processed_disk_image.filename}.jpg"
            processed_disk_image.save(output_file_path)
            processed_count += 1
            logger.info(f"Saved processed image: {output_file_path}")
    # Note: output_format validation should be done earlier in the function
    # to avoid unreachable code issues

    stats = {
        "total": len(image_files),
        "processed": processed_count,
        "failed": failed_count,
        "no_circles": no_circles_count,
        "unified_width": unified_width,
        "max_circle_share": max_circle_share,
    }

    logger.info(f"Processing complete: {stats}")
    return stats

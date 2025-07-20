"""
Image saving utilities.
"""

import logging
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
from PIL import Image

from circelizer.context import output_dir

logger = logging.getLogger(__name__)


def save_jpg(images: List[np.ndarray], filenames: List[Union[str, Path]]) -> int:
    """
    Save multiple images as JPG files.

    Args:
        images: List of images to save (numpy arrays)
        filenames: List of filenames to save as (strings or Path objects)

    Returns:
        Number of successfully saved images
    """
    if len(images) != len(filenames):
        raise ValueError(f"Number of images ({len(images)}) must match number of filenames ({len(filenames)})")

    saved_count = 0
    output_directory = output_dir.get()

    for image, filename in zip(images, filenames):
        try:
            # Convert filename to Path if it's a string
            if isinstance(filename, str):
                filename = Path(filename)

            # Create output path
            output_path = output_directory / filename.name

            # Save the image
            success = cv2.imwrite(str(output_path), image)

            if success:
                logger.info(f"Successfully saved {filename} -> {output_path}")
                saved_count += 1
            else:
                logger.error(f"Failed to save {filename}")

        except Exception as e:
            logger.error(f"Error saving {filename}: {repr(e)}")

    return saved_count


def save_gif(images: List[np.ndarray]) -> bool:
    """
    Save multiple images as a single GIF file.

    Args:
        images: List of images to save as GIF frames (numpy arrays)

    Returns:
        True if GIF was successfully saved, False otherwise
    """
    if not images:
        logger.warning("No images provided for GIF creation")
        return False

    try:
        output_directory = output_dir.get()
        gif_path = output_directory / "all.gif"

        # Convert OpenCV BGR images to PIL RGB images
        pil_images = []
        for img in images:
            # Convert BGR to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(rgb_img)
            pil_images.append(pil_img)

        # Save as GIF
        if pil_images:
            pil_images[0].save(
                str(gif_path),
                save_all=True,
                append_images=pil_images[1:],
                duration=1000 // 6,
                loop=0,  # Loop indefinitely
            )
            logger.info(f"Successfully saved GIF with {len(images)} frames: {gif_path}")
            return True
        else:
            logger.error("No valid images to save as GIF")
            return False

    except Exception as e:
        logger.error(f"Error saving GIF: {repr(e)}")
        return False

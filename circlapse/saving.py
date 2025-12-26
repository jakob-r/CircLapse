"""
Image saving utilities.
"""

import logging
import math
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from circlapse.image_types import GenericImage, RGBImage

from circlapse.context import output_dir

logger = logging.getLogger(__name__)


def save_jpg(images: List[GenericImage], filenames: List[str] | List[Path]) -> int:
    """
    Save multiple images as JPG files.

    Args:
        images: List of images to save (numpy arrays)
        filenames: List of filenames to save as (strings or Path objects)

    Returns:
        Number of successfully saved images
    """
    if len(images) != len(filenames):
        raise ValueError(
            f"Number of images ({len(images)}) must match number of filenames ({len(filenames)})"
        )

    saved_count = 0
    output_directory = Path(output_dir.get())

    for image, filename in zip(images, filenames):
        try:
            # Convert filename to Path if it's a string
            if isinstance(filename, str):
                filename_path = Path(filename)
            elif isinstance(filename, Path):
                filename_path = filename
            else:
                raise ValueError(f"Invalid filename type: {type(filename)}")

            # Create output path
            output_path = output_directory / filename_path.name

            # Save the image
            success = cv2.imwrite(str(output_path), image.imwrite_data)

            if success:
                logger.info(f"Successfully saved {filename_path} -> {output_path}")
                saved_count += 1
            else:
                logger.error(f"Failed to save {filename}")

        except Exception as e:
            logger.error(f"Error saving {filename}: {repr(e)}")

    return saved_count


def save_multi_jpg_with_text(
    images: list[GenericImage],
    texts: list[str],
    filename: str | Path,
    resolution: tuple[int, int] = (1024, 768),
) -> bool:
    """
    Save multiple images with text annotations on a single canvas.

    All images are laid out in a grid with their corresponding text below each image.
    Text supports multi-line using \\n.

    Args:
        images: List of images to display
        texts: List of text annotations (same length as images), supports \\n for multi-line
        filename: Output filename
        resolution: Canvas resolution as (width, height), default (1024, 768)

    Returns:
        True if successfully saved, False otherwise
    """
    if len(images) != len(texts):
        raise ValueError(
            f"Number of images ({len(images)}) must match number of texts ({len(texts)})"
        )

    if not images:
        logger.warning("No images provided for multi-image save")
        return False

    try:
        canvas_width, canvas_height = resolution
        n_images = len(images)

        # Calculate grid layout (try to make it roughly square)
        n_cols = math.ceil(math.sqrt(n_images))
        n_rows = math.ceil(n_images / n_cols)

        # Calculate cell dimensions
        cell_width = canvas_width // n_cols
        cell_height = canvas_height // n_rows

        # Reserve space for text (approximately 20% of cell height)
        text_height = max(40, cell_height // 5)
        image_height = cell_height - text_height
        image_width = cell_width

        # Create canvas (RGB)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        # Fill with dark gray background
        canvas[:] = (40, 40, 40)

        # Try to load a nice font, fall back to default if not available
        font_size = max(10, min(16, text_height // 4))
        try:
            # Try common monospace fonts
            font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", font_size)
        except OSError:
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", font_size
                )
            except OSError:
                font = ImageFont.load_default()

        # Convert to PIL for text rendering
        canvas_pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(canvas_pil)

        for idx, (img, text) in enumerate(zip(images, texts)):
            row = idx // n_cols
            col = idx % n_cols

            # Calculate position for this cell
            x_offset = col * cell_width
            y_offset = row * cell_height

            # Convert image to RGB numpy array for display
            img_rgb = RGBImage.from_numpy(img.data)
            img_data = img_rgb.data

            # Get original image dimensions
            orig_h, orig_w = img_data.shape[:2]

            # Calculate scaling to fit in cell while maintaining aspect ratio
            # Leave some padding (4 pixels on each side)
            padding = 4
            available_width = image_width - 2 * padding
            available_height = image_height - 2 * padding

            scale = min(available_width / orig_w, available_height / orig_h)
            new_w = int(orig_w * scale)
            new_h = int(orig_h * scale)

            # Resize image
            resized = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_AREA)

            # Calculate centered position within cell
            img_x = x_offset + padding + (available_width - new_w) // 2
            img_y = y_offset + padding + (available_height - new_h) // 2

            # Convert canvas back to numpy to paste image
            canvas = np.array(canvas_pil)

            # Paste resized image onto canvas
            canvas[img_y : img_y + new_h, img_x : img_x + new_w] = resized

            # Convert back to PIL for text
            canvas_pil = Image.fromarray(canvas)
            draw = ImageDraw.Draw(canvas_pil)

            # Draw text below image
            text_x = x_offset + padding
            text_y = y_offset + image_height + 2

            # Split text by newlines and draw each line
            lines = text.split("\n")
            line_height = font_size + 2
            for line_idx, line in enumerate(lines):
                line_y = text_y + line_idx * line_height
                if line_y + line_height > y_offset + cell_height:
                    break  # Don't draw beyond cell boundary
                draw.text((text_x, line_y), line, fill=(200, 200, 200), font=font)

        # Convert final canvas to numpy and save
        final_canvas = np.array(canvas_pil)

        # Create output path
        output_directory = Path(output_dir.get())
        if isinstance(filename, str):
            filename_path = Path(filename)
        else:
            filename_path = filename

        output_path = output_directory / filename_path.name

        # Convert RGB to BGR for cv2.imwrite
        final_bgr = cv2.cvtColor(final_canvas, cv2.COLOR_RGB2BGR)
        success = cv2.imwrite(str(output_path), final_bgr)

        if success:
            logger.info(f"Successfully saved multi-image: {output_path}")
            return True
        else:
            logger.error(f"Failed to save multi-image: {output_path}")
            return False

    except Exception as e:
        logger.error(f"Error saving multi-image {filename}: {repr(e)}")
        return False


def save_gif(images: List[GenericImage]) -> bool:
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
            rgb_img = cv2.cvtColor(img.data, cv2.COLOR_BGR2RGB)
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

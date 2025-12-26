import logging
from typing import Tuple, Union

import cv2
import numpy as np

from circlapse.disk_image import DiskImage
from circlapse.image_types import GrayscaleImage, RGBImage, BinaryImage

logger = logging.getLogger(__name__)


def center_and_crop_image(
    rgb_image: RGBImage, circle: Tuple[int, int, int]
) -> RGBImage:
    """
    Center the image on the detected circle and crop to square.

    Args:
        rgb_image: Input RGB image
        circle: Tuple of (x, y, radius) of the detected circle

    Returns:
        Centered and cropped square RGB image
    """
    x, y, radius = circle
    height, width = rgb_image.shape[:2]

    # Calculate the size of the square crop (2 * radius + some padding)
    crop_size = min(2 * radius + 50, min(width, height))

    # Calculate crop boundaries to center the circle
    crop_x = max(0, min(x - crop_size // 2, width - crop_size))
    crop_y = max(0, min(y - crop_size // 2, height - crop_size))

    # Ensure we don't go out of bounds
    crop_x = max(0, min(crop_x, width - crop_size))
    crop_y = max(0, min(crop_y, height - crop_size))

    # Crop the image
    cropped = rgb_image[crop_y : crop_y + crop_size, crop_x : crop_x + crop_size]

    return cropped


def center_and_crop_image_consistent(
    rgb_image: RGBImage, circle: Tuple[int, int, int], target_circle_share: float
) -> RGBImage:
    """
    Center the image on the detected circle and crop to a square with a consistent radius.
    Scales the image so that all circles have the same size in the output.

    Args:
        rgb_image: Input RGB image
        circle: Tuple of (x, y, radius) of the detected circle
        target_circle_share: How much of the image should be used for the circle.
            1 means all the image is used for the circle.

    Returns:
        The RGB image with a circle of the target radius and the target ratio
    """

    x, y, radius = circle

    # if the radius is 100, the circle_share is 0.8, the image has to be 100 * 1/0.8 big
    final_shortest_side = int(radius / target_circle_share) * 2

    crop_x = x - (final_shortest_side // 2)
    crop_y = y - (final_shortest_side // 2)

    return rgb_image[
        crop_y : crop_y + final_shortest_side, crop_x : crop_x + final_shortest_side
    ]


def output_width(images: list[Union[np.ndarray, DiskImage]]) -> int:
    """
    Calculate the unified output width based on all shortest sides.
    Works efficiently with both numpy arrays and DiskImage objects.

    Args:
        images: List of images (numpy arrays or DiskImage objects)

    Returns:
        A output width that downscales most images to a similar size but allows some upscaling.
    """
    all_shortest_sides = []

    for image in images:
        if isinstance(image, DiskImage):  # DiskImage object
            # Load image temporarily to get dimensions
            loaded_image = image.load()
            shortest_side = min(loaded_image.shape[:2])
        else:  # numpy array
            shortest_side = min(image.shape[:2])

        all_shortest_sides.append(shortest_side)

    return int(np.percentile(all_shortest_sides, 10))


def scale_image(rgb_image: RGBImage, target_width: int) -> RGBImage:
    """
    Scale an RGB image to a target width while maintaining aspect ratio.

    Args:
        rgb_image: Input RGB image
        target_width: The desired width for the output image

    Returns:
        Scaled RGB image with the target width
    """

    height, width = rgb_image.shape[:2]

    # Calculate scale factor to achieve target width
    scale_factor = target_width / width

    # Calculate new height to maintain aspect ratio
    new_height = int(height * scale_factor)

    # Resize the image
    scaled_image = cv2.resize(rgb_image, (target_width, new_height))

    return scaled_image


def circle_share(rgb_image: RGBImage, circle: tuple[int, int, int]) -> float:
    """
    Calculate the share of the image that is covered by the circle.

    Args:
        rgb_image: Input RGB image
        circle: Tuple of (x, y, radius) of the detected circle

    Returns:
        Circle share as a float between 0.0 and 1.0
    """

    x, y, radius = circle
    height, width = rgb_image.shape[:2]
    distance_to_left = x - radius
    distance_to_right = width - (x + radius)
    distance_to_top = y - radius
    distance_to_bottom = height - (y + radius)
    shortest_distance = max(
        0, min(distance_to_left, distance_to_right, distance_to_top, distance_to_bottom)
    )
    return radius / (radius + shortest_distance)


def automatic_postprocess(rgb_image: RGBImage) -> RGBImage:
    """
    Apply automatic post-processing enhancements similar to Lightroom auto adjustments.

    This function applies:
    - Automatic brightness and contrast adjustment
    - Saturation enhancement
    - Sharpness improvement
    - Color balance optimization

    Args:
        rgb_image: Input RGB image (BGR format)

    Returns:
        Enhanced RGB image with improved visual quality
    """

    # Convert to LAB color space for better color processing
    lab = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    # 1. Automatic brightness and contrast adjustment using CLAHE
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    l_channel = clahe.apply(l_channel)

    # 2. Enhance contrast using histogram equalization on L channel
    # Apply a subtle contrast enhancement
    l_channel = cv2.convertScaleAbs(l_channel, alpha=1.05, beta=3)

    # 3. Enhance saturation in LAB space (more conservative)
    a_channel = cv2.convertScaleAbs(a_channel, alpha=1.05, beta=0)
    b_channel = cv2.convertScaleAbs(b_channel, alpha=1.05, beta=0)

    # Merge channels back
    enhanced_lab = cv2.merge([l_channel, a_channel, b_channel])
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

    # 4. Apply unsharp masking for sharpness
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.5)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.3, gaussian, -0.3, 0)

    # 5. Final color balance adjustment (more conservative)
    # Convert to HSV for saturation and value adjustments
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Enhance saturation slightly (reduced from 1.1 to 1.05)
    s = cv2.convertScaleAbs(s, alpha=1.05, beta=0)

    # Enhance value (brightness) slightly (reduced from 1.05 to 1.02)
    v = cv2.convertScaleAbs(v, alpha=1.02, beta=0)

    # Merge and convert back to BGR
    enhanced_hsv = cv2.merge([h, s, v])
    final_image = cv2.cvtColor(enhanced_hsv, cv2.COLOR_HSV2BGR)

    # 6. Ensure values are in valid range
    final_image = np.clip(final_image, 0, 255).astype(np.uint8)

    return final_image


def crop_image(
    rgb_image: RGBImage, x: int, y: int, width: int, height: int
) -> RGBImage:
    """
    Crop an RGB image with out-of-bounds support, filling with black pixels.

    Args:
        rgb_image: Input RGB image
        x: Starting x coordinate (can be negative)
        y: Starting y coordinate (can be negative)
        width: Width of the crop region
        height: Height of the crop region

    Returns:
        Cropped RGB image with black padding for out-of-bounds areas
    """

    img_height, img_width = rgb_image.shape[:2]
    channels = rgb_image.shape[2] if len(rgb_image.shape) > 2 else 1

    # Create output image filled with black
    output = np.zeros((height, width, channels), dtype=rgb_image.dtype)

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
        output[output_y_start:output_y_end, output_x_start:output_x_end] = rgb_image[
            crop_y_start:crop_y_end, crop_x_start:crop_x_end
        ]

    return output


def image_median_edges(edges_binary: BinaryImage) -> tuple[int, int]:
    """
    Find the median point of edge pixels from Canny edge detection.

    This function calculates the point where 50% of edge pixels are to the right
    and 50% of edge pixels are above this point.

    Args:
        edges_binary: Binary edge image from Canny edge detection (bool)

    Returns:
        Tuple of (x, y) coordinates representing the median point
    """

    # Ensure edges is boolean
    if edges_binary.dtype != bool:
        edges_binary = edges_binary.astype(bool)

    # Find coordinates of all edge pixels
    edge_coords = np.where(edges_binary)
    y_coords = edge_coords[0]  # Row coordinates
    x_coords = edge_coords[1]  # Column coordinates

    if len(x_coords) == 0:
        # No edges found, return center of image
        height, width = edges_binary.shape
        return width // 2, height // 2

    # Calculate median x and y coordinates
    median_x = int(np.median(x_coords))
    median_y = int(np.median(y_coords))

    return median_x, median_y


def scale_shortest_side(
    rgb_image: RGBImage, target_size: int
) -> tuple[RGBImage, float]:
    """
    Scale an RGB image so that its longest side equals target_size, maintaining aspect ratio.

    Args:
        rgb_image: Input RGB image
        target_size: Target size for the longest side

    Returns:
        Tuple of (scaled_rgb_image, scale_factor)
    """

    processed_image = rgb_image.copy()
    height, width = processed_image.shape[:2]
    scale_factor = target_size / max(height, width)

    # scale image to target size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    processed_image = cv2.resize(processed_image, (new_width, new_height))
    logger.debug(f"Scaled image from {width}x{height} to {new_width}x{new_height}")

    return processed_image, scale_factor


def blur_and_normalize(image: BinaryImage, kernel_size: int = 5) -> GrayscaleImage:
    """
    Blur a binary image to create a glow effect and normalize it to 0-255.
    All white pixels in the original image remain white in the result.
    """
    image_data = image.data.astype(np.uint8) * 255
    blurred = cv2.GaussianBlur(image_data, (kernel_size, kernel_size), 0)
    max_value = np.max(blurred)
    min_value = np.min(blurred)
    if max_value > min_value:
        normalized = (blurred - min_value) / (max_value - min_value) * 255
    else:
        normalized = blurred
    # Ensure original white pixels remain white (glow effect)
    blurred_weight = 0.5
    result = np.maximum(normalized * blurred_weight, image_data)
    return GrayscaleImage.from_numpy(result.astype(np.uint8))

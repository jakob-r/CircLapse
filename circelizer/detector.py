import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from skimage.draw import ellipse_perimeter
from skimage.feature import canny
from skimage.transform import hough_ellipse

from circelizer import image_operators, settings
from circelizer.context import output_dir

logger = logging.getLogger(__name__)


def detect_circle(
    image: np.ndarray,
    image_name: str = "debug",
    target_size: int = 800,
    min_distance_to_border: float = 0.05,
    max_iterations: int = 5,
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """
    Detect the largest circle in the image using Hough Circle Transform.

    Args:
        image: Input image as numpy array
        image_name: Name for debug image (used when DEBUG=True)
        target_size: Target size for the longest side (default: 800px)
        min_distance_to_border: Minimum distance from border as fraction of image size
        max_iterations: Maximum number of iterations to try lowering thresholds (default: 5)

    Returns:
        Tuple of (processed_image, circle_coords) where circle_coords is (x, y, radius) or None
    """
    # Scale image to consistent size for better circle detection
    processed_image = image.copy()
    height, width = processed_image.shape[:2]
    scale_factor = target_size / max(height, width)

    # scale image to target size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    shortest_side = min(new_width, new_height)
    processed_image = cv2.resize(processed_image, (new_width, new_height))
    logger.debug(f"Scaled image from {width}x{height} to {new_width}x{new_height}")

    # Convert to grayscale
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)

    # Initial parameters for Hough Circle Transform
    param1 = 30  # Edge detection threshold
    param2 = 100  # Accumulator threshold

    # Try to detect circles with progressively lower thresholds
    circles_raw = None
    for iteration in range(max_iterations):
        logger.debug(
            f"Circle detection iteration {iteration + 1}/{max_iterations} with param1={param1}, param2={param2}"
        )

        circles_raw = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=100,
            param1=param1,
            param2=param2,
            minRadius=shortest_side // 6,
            maxRadius=shortest_side // 2,
        )

        if circles_raw is not None and len(circles_raw) > 0:
            logger.debug(f"Found circles in iteration {iteration + 1}")
            break

        # Lower thresholds for next iteration
        param1 = max(10, int(param1 * 0.95))  # Don't go below 10
        param2 = max(30, int(param2 * 0.95))  # Don't go below 30

    if circles_raw is None or len(circles_raw) == 0:
        logger.info(f"No circles found after {max_iterations} iterations. Returning None.")
        return image, None

    circles = np.round(circles_raw[0, :]).astype("int")

    # filter out circles that are not centered in the middle 1/3rd of the image
    x_center = processed_image.shape[1] // 2
    y_center = processed_image.shape[0] // 2
    x_range = (x_center - processed_image.shape[1] // 6, x_center + processed_image.shape[1] // 6)
    y_range = (y_center - processed_image.shape[0] // 6, y_center + processed_image.shape[0] // 6)
    circles = [
        circle
        for circle in circles
        if circle[0] > x_range[0] and circle[0] < x_range[1] and circle[1] > y_range[0] and circle[1] < y_range[1]
    ]

    # filter out circles too close to the border
    circle_shares = [image_operators.circle_share(processed_image, circle) for circle in circles]
    too_big = [share > (1 - min_distance_to_border) for share in circle_shares]
    small_circles = [circle for circle, too_big in zip(circles, too_big) if not too_big]
    if len(small_circles) == 0:
        logger.info(f"All {sum(too_big)} circles are too close to the border. Returning None.")
        return image, None

    big_circles = [circle for circle, too_big in zip(circles, too_big) if too_big]

    # Return the largest circle (assuming it's the main object)
    largest_circle = max(small_circles, key=lambda x: x[2])
    max_circle_share = image_operators.circle_share(image, largest_circle)

    # Save debug image if DEBUG is enabled
    if settings.DEBUG:
        debug_image = processed_image.copy()
        for x, y, radius in big_circles:
            # Draw circle outline
            cv2.circle(debug_image, (x, y), radius, (0, 0, 120), 2)
            # Draw center point
            cv2.circle(debug_image, (x, y), 2, (0, 0, 120), 3)
        for x, y, radius in small_circles:
            # Draw circle outline
            cv2.circle(debug_image, (x, y), radius, (0, 255, 0), 2)
            # Draw center point
            cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)

        # draw biggest circle
        cv2.circle(debug_image, (largest_circle[0], largest_circle[1]), largest_circle[2], (255, 255, 255), 2)
        cv2.putText(
            debug_image,
            f"Max circle share: {max_circle_share:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2,
        )

        # draw box centered on the largest circle
        box_center = (largest_circle[0], largest_circle[1])
        box_side = (2 * largest_circle[2]) / max_circle_share
        top_left = (int(box_center[0] - box_side // 2), int(box_center[1] - box_side // 2))
        bottom_right = (int(box_center[0] + box_side // 2), int(box_center[1] + box_side // 2))
        cv2.rectangle(debug_image, top_left, bottom_right, (255, 255, 255), 2)

        # Save debug image if output directory is available
        debug_path = output_dir.get() / "debug" / f"{image_name}_circles.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_image)
        logger.debug(f"Saved debug image: {debug_path}")

    # Scale coordinates back to original image size if image was scaled
    x, y, radius = largest_circle
    original_x = int(x / scale_factor)
    original_y = int(y / scale_factor)
    original_radius = int(radius / scale_factor)
    return (image, (original_x, original_y, original_radius))


def detect_ellipse_and_transform(
    image: np.ndarray, image_name: str = "debug", target_size: int = 256, min_distance_to_border: float = 0.05
) -> Tuple[np.ndarray, Optional[Tuple[int, int, int]]]:
    """
    Detect the largest ellipse in the image using Hough Ellipse Transform and transform it to a circle.

    Args:
        image: Input image as numpy array
        image_name: Name for debug image (used when DEBUG=True)
        target_size: Target size for the longest side (default: 800px)
        min_distance_to_border: Minimum distance from border as fraction of image size

    Returns:
        Tuple of (transformed_image, circle_coords) where circle_coords is (x, y, radius) or None
    """
    # Scale image to consistent size for better ellipse detection
    processed_image = image.copy()
    height, width = processed_image.shape[:2]
    scale_factor = target_size / max(height, width)

    # scale image to target size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    shortest_side = min(new_width, new_height)
    processed_image = cv2.resize(processed_image, (new_width, new_height))
    logger.debug(f"Scaled image from {width}x{height} to {new_width}x{new_height}")

    # Convert to grayscale
    gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = canny(gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

    # Detect ellipses using Hough Ellipse Transform

    logger.info(f"Detecting ellipses with accuracy: {16} on picture of size {edges.shape}")
    ellipses = hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
    if len(ellipses) == 0:
        logger.info("No ellipses found. Returning None.")
        return image, None

    ellipses.sort(order="accumulator")

    best = list(ellipses[-1])
    yc, xc, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]

    # Draw the ellipse on the original image

    if settings.DEBUG:
        debug_image = processed_image.copy()
        for ellipse in ellipses:
            yc, xc, a, b = (int(round(x)) for x in list(ellipse)[1:5])
            orientation = ellipse[5]
            cy, cx = ellipse_perimeter(yc, xc, a, b, orientation)
            # bound the ellipse to the image
            cy = np.clip(cy, 0, debug_image.shape[0] - 1)
            cx = np.clip(cx, 0, debug_image.shape[1] - 1)
            debug_image[cy, cx] = (0, 0, 255)

        debug_path = output_dir.get() / "debug" / f"{image_name}_ellipses.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_image)
        logger.debug(f"Saved debug image: {debug_path}")

        # Use the largest valid ellipse
    largest_ellipse = best
    center_y, center_x, a, b = (int(round(x)) for x in best[1:5])
    orientation = best[5]

    # Transform the image so the ellipse becomes a circle
    transformed_image = transform_ellipse_to_circle(image, tuple(largest_ellipse))

    # Calculate the radius of the resulting circle (average of a and b)
    radius = int((a + b) / 2)

    # Scale coordinates back to original image size
    original_x = int(center_x / scale_factor)
    original_y = int(center_y / scale_factor)
    original_radius = int(radius / scale_factor)

    return transformed_image, (original_x, original_y, original_radius)


def transform_ellipse_to_circle(image: np.ndarray, ellipse: tuple) -> np.ndarray:
    """
    Transform an image so that the detected ellipse becomes a circle.

    Args:
        image: Input image
        ellipse: Ellipse parameters (center_y, center_x, a, b, orientation)

    Returns:
        Transformed image with the ellipse converted to a circle
    """
    center_y, center_x, a, b = ellipse[1:5]
    orientation = ellipse[5]

    # Calculate the aspect ratio of the ellipse
    aspect_ratio = a / b if b > 0 else 1.0

    # Create transformation matrix
    # First, translate to center the ellipse
    translation_matrix = np.array([[1, 0, -center_x], [0, 1, -center_y], [0, 0, 1]])

    # Rotate to align with axes
    rotation_matrix = np.array(
        [[np.cos(-orientation), -np.sin(-orientation), 0], [np.sin(-orientation), np.cos(-orientation), 0], [0, 0, 1]]
    )

    # Scale to make it circular
    scale_matrix = np.array([[1, 0, 0], [0, aspect_ratio, 0], [0, 0, 1]])

    # Combine transformations
    transform_matrix = scale_matrix @ rotation_matrix @ translation_matrix

    # Apply transformation
    height, width = image.shape[:2]
    transformed_image = cv2.warpAffine(
        image,
        transform_matrix[:2, :],
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )

    return transformed_image

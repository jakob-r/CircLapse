import numpy as np
import cv2
from typing import cast


def create_circle_image(size: int = 400, radius: int = 150) -> np.ndarray:
    """
    Creates a square image with a white background and a black filled circle in the center.
    """
    image = np.ones((size, size), dtype=np.uint8) * 255  # White background
    center_x, center_y = size // 2, size // 2
    # Cast to ensure proper typing for cv2.circle
    image_array = cast(np.ndarray, image)
    cv2.circle(
        image_array, (center_x, center_y), radius, (0,), -1
    )  # Black filled circle
    return image


def create_circle_outline_image(
    size: int = 400, radius: int = 150, thickness: int = 1
) -> np.ndarray:
    """
    Creates a square image with a black background and a white outline circle in the center.
    """
    image = np.zeros((size, size), dtype=np.uint8)  # Black background
    center_x, center_y = size // 2, size // 2
    cv2.circle(
        image, (center_x, center_y), radius, (255,), thickness
    )  # White outline circle
    return image


def create_circle_image_with_noise(
    size: int = 400, radius: int = 350, noise_level: float = 0.1
) -> np.ndarray:
    """
    Creates a square image with a white background and a black circle in the center.
    """
    circle_image = create_circle_image(size, radius)
    noise_image = np.random.normal(0, noise_level, size)
    # cap at 255
    noise_image = np.clip(noise_image, 0, 255)
    return circle_image + noise_image

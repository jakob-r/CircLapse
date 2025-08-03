"""
Image type definitions and utilities for better type clarity.

This module provides clear type aliases and utility functions to distinguish between
different image formats (RGB, grayscale, binary) and their data types.
"""

from typing import Type
import numpy as np
from pydantic import BaseModel
from numpydantic import NDArray, Shape


class GenericImage(BaseModel):
    data: NDArray[Shape["*,..."], int]  # noqa: F722

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def to_type(self, type: Type["GenericImage"]) -> "GenericImage":
        return type.from_numpy(self.data)

    def blank_like(self) -> "GenericImage":
        return self.model_copy(data=np.zeros_like(self.data))

    @classmethod
    def blank(cls, resolution: tuple[int, int]) -> "GenericImage":
        raise NotImplementedError(
            "Blank image creation is not implemented for this type"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GenericImage):
            return False
        return np.array_equal(self.data, other.data)

    def __ne__(self, other: object) -> bool:
        if not isinstance(other, GenericImage):
            return True
        return not np.array_equal(self.data, other.data)

    def copy(self) -> "GenericImage":
        return self.model_copy()


class RGBImage(GenericImage):
    data: NDArray[Shape["*,*,3"], int]  # noqa: F722

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> "RGBImage":
        if is_rgb_image(data):
            return cls(data=data)
        elif is_grayscale_image(data):
            # Convert grayscale to RGB by repeating the channel
            rgb_data = np.stack([data, data, data], axis=2)
            return cls(data=rgb_data)
        elif is_binary_image(data):
            # Convert binary to RGB (white for True, black for False)
            rgb_data = np.stack(
                [
                    data.astype(np.uint8) * 255,
                    data.astype(np.uint8) * 255,
                    data.astype(np.uint8) * 255,
                ],
                axis=2,
            )
            return cls(data=rgb_data)
        else:
            raise ValueError(f"Cannot convert {data.shape} to RGB")

    @classmethod
    def blank(cls, resolution: tuple[int, int]) -> "RGBImage":
        return cls(data=np.zeros((resolution[0], resolution[1], 3), dtype=np.uint8))


class GrayscaleImage(GenericImage):
    data: NDArray[Shape["*,*"], int]  # noqa: F722

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> "GrayscaleImage":
        if is_grayscale_image(data):
            return cls(data=data)
        elif is_rgb_image(data):
            # Convert RGB to grayscale using luminance formula
            # Y = 0.299*R + 0.587*G + 0.114*B
            grayscale_data = np.dot(data[..., :3], [0.299, 0.587, 0.114]).astype(
                np.uint8
            )
            return cls(data=grayscale_data)
        elif is_binary_image(data):
            # Convert binary to grayscale (255 for True, 0 for False)
            grayscale_data = data.astype(np.uint8) * 255
            return cls(data=grayscale_data)
        else:
            raise ValueError(f"Cannot convert {data.shape} to Grayscale")

    @classmethod
    def blank(cls, resolution: tuple[int, int]) -> "GrayscaleImage":
        return cls(data=np.zeros((resolution[0], resolution[1]), dtype=np.uint8))


class BinaryImage(GenericImage):
    data: NDArray[Shape["*,*"], bool]  # noqa: F722

    @classmethod
    def from_numpy(cls, data: np.ndarray) -> "BinaryImage":
        if is_binary_image(data):
            return cls(data=data)
        elif is_rgb_image(data):
            # Convert RGB to binary by converting to grayscale first, then thresholding
            grayscale = np.dot(data[..., :3], [0.299, 0.587, 0.114])
            binary_data = grayscale > 127
            return cls(data=binary_data)
        elif is_grayscale_image(data):
            # Convert grayscale to binary using threshold
            binary_data = data > 127
            return cls(data=binary_data)
        else:
            raise ValueError(f"Cannot convert {data.shape} to Binary")

    @classmethod
    def blank(cls, resolution: tuple[int, int]) -> "BinaryImage":
        return cls(data=np.zeros((resolution[0], resolution[1]), dtype=bool))


def is_rgb_image(image: np.ndarray) -> bool:
    """Check if image is RGB (3 channels, uint8)."""
    return len(image.shape) == 3 and image.shape[2] == 3 and image.dtype == np.uint8


def is_grayscale_image(image: np.ndarray) -> bool:
    """Check if image is grayscale (2D, uint8)."""
    return len(image.shape) == 2 and image.dtype == np.uint8


def is_binary_image(image: np.ndarray) -> bool:
    """Check if image is binary (2D, bool)."""
    return len(image.shape) == 2 and image.dtype == bool

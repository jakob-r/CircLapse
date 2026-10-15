"""
Unit tests for image_types module.
"""

import numpy as np
import pytest
from circlapse.image_types import (
    GenericImage,
    RGBImage,
    GrayscaleImage,
    BinaryImage,
    is_rgb_image,
    is_grayscale_image,
    is_binary_image,
)


class TestImage:
    """Test the base Image class."""

    def test_image_creation(self):
        """Test creating a basic Image instance."""
        data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint8)
        image = GenericImage(data=data)
        assert image.data.shape == (2, 3)
        assert image.data.dtype == np.uint8


class TestRGBImage:
    """Test the RGBImage class."""

    def test_rgb_image_creation(self):
        """Test creating an RGBImage instance."""
        data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        rgb_image = RGBImage(data=data)
        assert rgb_image.data.shape == (10, 10, 3)
        assert rgb_image.data.dtype == np.uint8

    def test_from_numpy_rgb(self):
        """Test from_numpy with RGB data."""
        data = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
        rgb_image = RGBImage.from_numpy(data)
        assert isinstance(rgb_image, RGBImage)
        assert rgb_image.data.shape == (5, 5, 3)

    def test_from_numpy_grayscale(self):
        """Test from_numpy with grayscale data."""
        data = np.random.randint(0, 256, (4, 4), dtype=np.uint8)
        rgb_image = RGBImage.from_numpy(data)
        assert isinstance(rgb_image, RGBImage)
        assert rgb_image.data.shape == (4, 4, 3)
        # Check that all channels are identical
        assert np.array_equal(rgb_image.data[:, :, 0], rgb_image.data[:, :, 1])
        assert np.array_equal(rgb_image.data[:, :, 1], rgb_image.data[:, :, 2])

    def test_from_numpy_binary(self):
        """Test from_numpy with binary data."""
        data = np.array([[True, False], [False, True]], dtype=bool)
        rgb_image = RGBImage.from_numpy(data)
        assert isinstance(rgb_image, RGBImage)
        assert rgb_image.data.shape == (2, 2, 3)
        # Check that True becomes 255, False becomes 0
        assert rgb_image.data[0, 0, 0] == 255  # True
        assert rgb_image.data[0, 1, 0] == 0  # False

    def test_from_numpy_invalid(self):
        """Test from_numpy with invalid data."""
        data = np.random.rand(3, 3, 4)  # 4 channels instead of 3
        with pytest.raises(ValueError):
            RGBImage.from_numpy(data)


class TestGrayscaleImage:
    """Test the GrayscaleImage class."""

    def test_grayscale_image_creation(self):
        """Test creating a GrayscaleImage instance."""
        data = np.random.randint(0, 256, (8, 8), dtype=np.uint8)
        gray_image = GrayscaleImage(data=data)
        assert gray_image.data.shape == (8, 8)
        assert gray_image.data.dtype == np.uint8

    def test_from_numpy_grayscale(self):
        """Test from_numpy with grayscale data."""
        data = np.random.randint(0, 256, (6, 6), dtype=np.uint8)
        gray_image = GrayscaleImage.from_numpy(data)
        assert isinstance(gray_image, GrayscaleImage)
        assert gray_image.data.shape == (6, 6)

    def test_from_numpy_rgb(self):
        """Test from_numpy with RGB data."""
        data = np.random.randint(0, 256, (5, 5, 3), dtype=np.uint8)
        gray_image = GrayscaleImage.from_numpy(data)
        assert isinstance(gray_image, GrayscaleImage)
        assert gray_image.data.shape == (5, 5)
        assert gray_image.data.dtype == np.uint8

    def test_from_numpy_binary(self):
        """Test from_numpy with binary data."""
        data = np.array([[True, False], [False, True]], dtype=bool)
        gray_image = GrayscaleImage.from_numpy(data)
        assert isinstance(gray_image, GrayscaleImage)
        assert gray_image.data.shape == (2, 2)
        # Check that True becomes 255, False becomes 0
        assert gray_image.data[0, 0] == 255  # True
        assert gray_image.data[0, 1] == 0  # False

    def test_from_numpy_invalid(self):
        """Test from_numpy with invalid data."""
        data = np.random.rand(3, 3, 3, 3)  # 4D array
        with pytest.raises(ValueError):
            GrayscaleImage.from_numpy(data)


class TestBinaryImage:
    """Test the BinaryImage class."""

    def test_binary_image_creation(self):
        """Test creating a BinaryImage instance."""
        data = np.array([[True, False], [False, True]], dtype=bool)
        binary_image = BinaryImage(data=data)
        assert binary_image.data.shape == (2, 2)
        assert binary_image.data.dtype == bool

    def test_from_numpy_binary(self):
        """Test from_numpy with binary data."""
        data = np.array([[True, False], [False, True]], dtype=bool)
        binary_image = BinaryImage.from_numpy(data)
        assert isinstance(binary_image, BinaryImage)
        assert binary_image.data.shape == (2, 2)
        assert binary_image.data.dtype == bool

    def test_from_numpy_grayscale(self):
        """Test from_numpy with grayscale data."""
        data = np.array([[100, 200], [50, 150]], dtype=np.uint8)
        binary_image = BinaryImage.from_numpy(data)
        assert isinstance(binary_image, BinaryImage)
        assert binary_image.data.shape == (2, 2)
        assert binary_image.data.dtype == bool
        # Check thresholding: values > 127 should be True
        assert binary_image.data[0, 1]  # 200 > 127
        assert not binary_image.data[0, 0]  # 100 <= 127

    def test_from_numpy_rgb(self):
        """Test from_numpy with RGB data."""
        data = np.array([[[100, 150, 200], [50, 100, 150]]], dtype=np.uint8)
        binary_image = BinaryImage.from_numpy(data)
        assert isinstance(binary_image, BinaryImage)
        assert binary_image.data.shape == (1, 2)
        assert binary_image.data.dtype == bool

    def test_from_numpy_invalid(self):
        """Test from_numpy with invalid data."""
        data = np.random.rand(3, 3, 3, 3)  # 4D array
        with pytest.raises(ValueError):
            BinaryImage.from_numpy(data)


class TestUtilityFunctions:
    """Test the utility functions."""

    def test_is_rgb_image(self):
        """Test is_rgb_image function."""
        # Valid RGB image
        rgb_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        assert is_rgb_image(rgb_data)

        # Invalid: wrong shape
        invalid_shape = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        assert not is_rgb_image(invalid_shape)

        # Invalid: wrong dtype
        wrong_dtype = np.random.randint(0, 256, (10, 10, 3), dtype=np.int32)
        assert not is_rgb_image(wrong_dtype)

    def test_is_grayscale_image(self):
        """Test is_grayscale_image function."""
        # Valid grayscale image
        gray_data = np.random.randint(0, 256, (10, 10), dtype=np.uint8)
        assert is_grayscale_image(gray_data)

        # Invalid: wrong shape
        invalid_shape = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        assert not is_grayscale_image(invalid_shape)

        # Invalid: wrong dtype
        wrong_dtype = np.random.randint(0, 256, (10, 10), dtype=np.int32)
        assert not is_grayscale_image(wrong_dtype)

    def test_is_binary_image(self):
        """Test is_binary_image function."""
        # Valid binary image
        binary_data = np.array([[True, False], [False, True]], dtype=bool)
        assert is_binary_image(binary_data)

        # Invalid: wrong shape
        invalid_shape = np.array([[[True, False]]], dtype=bool)
        assert not is_binary_image(invalid_shape)

        # Invalid: wrong dtype
        wrong_dtype = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        assert not is_binary_image(wrong_dtype)

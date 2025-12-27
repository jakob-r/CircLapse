import numpy as np
from circlapse.morph import morph_image
from circlapse.image_types import GrayscaleImage


class TestMorphImage:
    """Test cases for the morph_image function."""

    def test_no_transformation(self):
        """Test that no transformation returns the same image."""
        # Create a simple test grayscale image
        grayscale_image = GrayscaleImage.blank((10, 10))
        grayscale_image.data[3:7, 3:7] = 255  # White square in center

        result = morph_image(grayscale_image, 0, 0, 1.0, 0.0, 0.0, 0.0)

        # Should be identical
        assert result == grayscale_image

    def test_tilt(self):
        """Test tilt transformation."""
        # Create a simple test grayscale image
        grayscale_image = GrayscaleImage.blank((10, 10))
        grayscale_image.data[0, 0] = 255  # Single white pixel at top left

        result = morph_image(grayscale_image, 0, 0, 1.0, 0.1, 0.0, 0.0)
        assert result != grayscale_image

    def test_scaling(self):
        """Test scaling transformation."""
        # Create a simple test grayscale image
        grayscale_image = GrayscaleImage.blank((11, 11))
        grayscale_image.data[4:7, 4:7] = 255  # White square in center

        result = morph_image(grayscale_image, 0, 0, 1.2, 0.0, 0.0, 0.0)

        # With 2x scaling, the image should change
        assert (result.data[3:8, 3:8] > 0).all()

    def test_output_size(self):
        """Test that output has same size as input."""
        grayscale_image = GrayscaleImage.blank((15, 20))

        result = morph_image(grayscale_image, 5, 3, 1.5, 0.1, 0.2, 45.0)

        assert result.shape == grayscale_image.shape

    def test_x_shift_10_pixels(self):
        """Test x_shift of 10 pixels by comparing with manually constructed result."""
        # Create image with white column at x=5
        img = GrayscaleImage.blank((20, 30))
        img.data[:, 5] = 255

        # Expected: white column shifted to x=15
        expected = GrayscaleImage.blank((20, 30))
        expected.data[:, 15] = 255

        result = morph_image(
            img, x_shift=10, y_shift=0, scale=1.0, tilt_x=0.0, tilt_y=0.0, rotation=0.0
        )

        assert np.array_equal(result.data, expected.data)

    def test_rotate_centered_square_90_180_270(self):
        """Rotating a centered square by 90, 180, 270 degrees should produce the same image."""
        # Centered white square - interior should be perfectly preserved
        # Edge pixels may differ slightly due to interpolation
        img = GrayscaleImage.blank((21, 21))
        img.data[5:16, 5:16] = 255  # 11x11 white square

        for rotation in [90.0, 180.0, 270.0]:
            result = morph_image(
                img,
                x_shift=0,
                y_shift=0,
                scale=1.0,
                tilt_x=0.0,
                tilt_y=0.0,
                rotation=rotation,
            )
            # Interior (excluding edge pixels) should be identical
            assert np.array_equal(result.data[6:15, 6:15], img.data[6:15, 6:15]), (
                f"Failed for rotation={rotation}"
            )
            # Total pixel sum should be preserved
            assert np.sum(result.data) == np.sum(img.data), (
                f"Sum changed for rotation={rotation}"
            )

    def test_rotate_left_top_square_90_180_270(self):
        # if the top left quarter is white, after rotation, the top right quarter should be white
        top_left_square = GrayscaleImage.blank((10, 10))
        top_left_square.data[0:5, 0:5] = 255

        top_right_square = GrayscaleImage.blank((10, 10))
        top_right_square.data[0:5, 5:10] = 255

        bottom_left_square = GrayscaleImage.blank((10, 10))
        bottom_left_square.data[5:10, 0:5] = 255

        bottom_right_square = GrayscaleImage.blank((10, 10))
        bottom_right_square.data[5:10, 5:10] = 255

        rotations = {
            90: bottom_left_square,
            180: bottom_right_square,
            270: top_right_square,
        }

        for rotation, expected in rotations.items():
            result = morph_image(top_left_square, 0, 0, 1.0, 0.0, 0.0, rotation)
            assert (
                np.mean(result.data == expected.data) > 0.9
            )  # not pixel perfect, but close

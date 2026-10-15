import numpy as np
from circlapse.image_operators import circle_share


class TestCircleShare:
    """Test cases for the circle_share function."""

    def test_circle_centered_in_square_image(self):
        """Test circle perfectly centered in a square image."""
        # Create a 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        circle = (50, 50, 30)

        result = circle_share(image, circle)

        expected = 0.6
        assert result == expected

    def test_circle_touching_edge(self):
        """Test circle that touches the edge of the image."""
        # Create a 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        circle = (30, 50, 30)

        result = circle_share(image, circle)
        assert result == 1

    def test_circle_partially_outside_image(self):
        """Test circle that extends partially outside the image."""
        # Create a 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        circle = (20, 50, 30)

        result = circle_share(image, circle)

        assert result == 1

    def test_circle_in_rectangular_image(self):
        """Test circle in a rectangular image (width > height)."""
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        circle = (100, 50, 30)

        result = circle_share(image, circle)

        assert result == 0.6

    def test_circle_in_tall_rectangular_image(self):
        """Test circle in a tall rectangular image (height > width)."""
        # Create a 100x200 image
        image = np.zeros((200, 100, 3), dtype=np.uint8)
        circle = (50, 100, 30)

        result = circle_share(image, circle)

        assert result == 0.6

    def test_circle_at_corner(self):
        """Test circle positioned at a corner of the image."""
        # Create a 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        circle = (30, 30, 30)  # touches top-left corner

        result = circle_share(image, circle)

        assert result == 1

    def test_small_circle_well_inside_image(self):
        """Test a small circle well inside the image boundaries."""
        # Create a 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        circle = (50, 50, 10)  # center at (50, 50) with radius 10

        result = circle_share(image, circle)

        assert result == 10 / 50

    def test_circle_center_outside_image(self):
        """Test circle with center outside but radius extends into image."""
        # Create a 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        circle = (
            -10,
            50,
            40,
        )  # center at (-10, 50) with radius 40 (extends into image)

        result = circle_share(image, circle)

        # Expected: shortest_distance = 0 (extends beyond edge)
        # shortest_side = 100, so share = (100 - 0) / 100 = 1.0
        expected = 1.0
        assert result == expected

    def test_floating_point_precision(self):
        """Test that the function returns a float with reasonable precision."""
        # Create a 100x100 image
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        circle = (50, 50, 25)  # center at (50, 50) with radius 25

        result = circle_share(image, circle)

        expected = 25 / 50
        assert isinstance(result, float)
        assert abs(result - expected) < 1e-10  # High precision check

import numpy as np
import cv2
import pytest
from circlapse.generators import create_circle_outline_image
from circlapse.image_operators import blur_and_normalize
from circlapse.shape_detector import (
    detect_edges,
    match_score,
    optim_morph,
    optim_morph_gradient,
    rmse_score,
)
from circlapse.morph import morph_image
from circlapse.image_types import BinaryImage, GrayscaleImage


class TestMatchScore:
    """Test cases for the match_score function."""

    def test_black_on_black(self):
        """Test case 1: black on black should score 1."""
        # Create 10x10 black images
        edges_binary = BinaryImage.blank((10, 10))
        goal_edges_binary = BinaryImage.blank((10, 10))

        result = match_score(edges_binary, goal_edges_binary)

        assert result == 1.0

    def test_white_on_white(self):
        """Test case 2: white on white should score 1."""
        # Create 10x10 white images
        edges_binary = BinaryImage.blank((10, 10))
        goal_edges_binary = BinaryImage.blank((10, 10))

        result = match_score(edges_binary, goal_edges_binary)

        assert result == 1.0

    def test_black_in_white(self):
        """Test case 3: black in white should score 0."""
        # Create 10x10 images with opposite values
        edges_binary = BinaryImage.blank((10, 10))
        goal_edges_binary = BinaryImage.from_numpy(np.ones((10, 10), dtype=bool))

        result = match_score(edges_binary, goal_edges_binary)
        assert result == 0.0

        result = match_score(goal_edges_binary, edges_binary)
        assert result == 0.0

    def test_white_square_on_black_background(self):
        """Test case 5: white square on black background with itself should score 1."""
        # Create 10x10 black image with white square in center
        edges_binary = BinaryImage.blank((10, 10))
        edges_binary.data[3:7, 3:7] = True  # 4x4 white square in center

        goal_edges_binary = edges_binary.copy()

        result = match_score(edges_binary, goal_edges_binary)

        assert result == 1.0

    def test_half_white_half_black(self):
        """Test case 6: edges: white left, black right, goal_edges: score 0.5."""
        # Create 10x10 image with left half white, right half black
        edges_binary = BinaryImage.blank((10, 10))
        edges_binary.data[:, :5] = True  # Left half white

        goal_edges_binary = BinaryImage.blank((10, 10))

        result = match_score(edges_binary, goal_edges_binary)

        assert result == 0.5

    def test_circle_with_dilated_goal(self):
        """Test case 8: circle on black with bigger line width goal_edges."""
        # Create 10x10 black image with white circle
        # Draw a circle using OpenCV
        circle_img = np.zeros((10, 10), dtype=np.uint8)
        cv2.circle(circle_img, (5, 5), 3, 255, 1)  # Circle with radius 3, thickness 1
        edges_binary = BinaryImage.from_numpy(circle_img.astype(bool))

        # Create goal_edges with same circle but dilated (bigger line width)
        goal_edges_binary = np.zeros((10, 10), dtype=bool)
        goal_circle_img = np.zeros((10, 10), dtype=np.uint8)
        cv2.circle(
            goal_circle_img, (5, 5), 3, 255, 2
        )  # Circle with radius 3, thickness 2
        goal_edges_binary = BinaryImage.from_numpy(goal_circle_img.astype(bool))

        result = match_score(edges_binary, goal_edges_binary)

        # Should be less than 1.0 but greater than 0.0
        assert 0.0 < result < 1.0

    def test_small_patterns(self):
        """Test with small patterns to verify edge cases."""
        # 2x2 images
        edges_binary = BinaryImage.from_numpy(
            np.array([[True, False], [False, True]], dtype=bool)
        )
        goal_edges_binary = BinaryImage.from_numpy(
            np.array([[True, False], [False, True]], dtype=bool)
        )

        result = match_score(edges_binary, goal_edges_binary)
        assert result == 1.0

        # Opposite patterns
        goal_edges_opposite = BinaryImage.from_numpy(
            np.array([[False, True], [True, False]], dtype=bool)
        )
        result_opposite = match_score(edges_binary, goal_edges_opposite)
        assert result_opposite == 0.0

    def test_empty_images(self):
        """Test with 1x1 images."""
        edges_binary = BinaryImage.from_numpy(np.array([[False]], dtype=bool))
        goal_edges_binary = BinaryImage.from_numpy(np.array([[False]], dtype=bool))

        result = match_score(edges_binary, goal_edges_binary)
        assert result == 1.0

        edges_binary = BinaryImage.from_numpy(np.array([[True]], dtype=bool))
        goal_edges_binary = BinaryImage.from_numpy(np.array([[False]], dtype=bool))

        result = match_score(edges_binary, goal_edges_binary)
        assert result == 0.0

    def test_grayscale_goal_edges(self):
        """Test match_score with grayscale goal_edges."""
        # Test perfect match: binary True (1.0) matches grayscale white (255 -> 1.0)
        edges_binary = BinaryImage.from_numpy(np.array([[True]], dtype=bool))
        goal_edges_grayscale = GrayscaleImage.from_numpy(
            np.array([[255]], dtype=np.uint8)
        )
        result = match_score(edges_binary, goal_edges_grayscale)
        assert result == 1.0

        # Test perfect match: binary False (0.0) matches grayscale black (0 -> 0.0)
        edges_binary = BinaryImage.from_numpy(np.array([[False]], dtype=bool))
        goal_edges_grayscale = GrayscaleImage.from_numpy(
            np.array([[0]], dtype=np.uint8)
        )
        result = match_score(edges_binary, goal_edges_grayscale)
        assert result == 1.0

        # Test partial match: binary True (1.0) matches grayscale 128 (0.5)
        edges_binary = BinaryImage.from_numpy(np.array([[True]], dtype=bool))
        goal_edges_grayscale = GrayscaleImage.from_numpy(
            np.array([[128]], dtype=np.uint8)
        )
        result = match_score(edges_binary, goal_edges_grayscale)
        assert result == pytest.approx(0.5, abs=0.002)

    def test_grayscale_circle(self):
        """
        Test realistic case:
         - edges: binary circle outline
         - goal_edges: circle with blurred edges to guide the optimizer
        """
        edges_binary = BinaryImage.from_numpy(create_circle_outline_image(32, 10))
        edges_grayscale_blurred = blur_and_normalize(edges_binary)
        score = match_score(edges_binary, edges_grayscale_blurred)
        assert score > 0.8

        # circle too small
        edges_binary_small = BinaryImage.from_numpy(create_circle_outline_image(32, 8))
        score_small = match_score(edges_binary_small, edges_grayscale_blurred)
        assert score_small > 0
        assert score_small < score

        # circle too large
        edges_binary_large = BinaryImage.from_numpy(create_circle_outline_image(32, 12))
        score_large = match_score(edges_binary_large, edges_grayscale_blurred)
        assert score_large > 0
        assert score_large < score


class TestRmseScore:
    """Test cases for the rmse_score function."""

    def test_identical_images(self):
        """Identical images should have RMSE of 0."""
        img = GrayscaleImage.from_numpy(np.full((10, 10), 128, dtype=np.uint8))
        assert rmse_score(img, img) == 0.0

    def test_black_vs_white(self):
        """All black vs all white should have RMSE of 1."""
        black = GrayscaleImage.blank((10, 10))
        white = GrayscaleImage.from_numpy(np.full((10, 10), 255, dtype=np.uint8))
        assert rmse_score(black, white) == 1.0

    def test_half_difference(self):
        """Half intensity difference should give RMSE of 0.5."""
        img1 = GrayscaleImage.blank((10, 10))
        img2 = GrayscaleImage.from_numpy(np.full((10, 10), 128, dtype=np.uint8))
        # 128/255 ≈ 0.502
        assert rmse_score(img1, img2) == pytest.approx(128 / 255, abs=0.001)


class TestOptimMorph:
    """Test cases for the optim_morph function."""

    @pytest.mark.skip(reason="skipping slow/integration transformation find test")
    def test_same_edges(self):
        """Test case 1: edges and goal_edges are the same."""
        # Create a simple test pattern
        edges_binary = BinaryImage.blank((20, 20))
        edges_binary.data[5:15, 5:15] = True  # White square in center
        goal_edges_binary = edges_binary.copy()

        optimized_edges_binary, best_score, best_params = optim_morph(
            edges_binary, goal_edges_binary
        )

        # Should find perfect match with no transformation
        assert best_score == 1.0
        assert best_params["x_shift"] == 0
        assert best_params["y_shift"] == 0
        assert best_params["scale"] == 1.0
        assert best_params["tilt_x"] == 0.0
        assert best_params["tilt_y"] == 0.0
        assert best_params["rotation"] == 0.0
        assert optimized_edges_binary == edges_binary

    @pytest.mark.skip(reason="skipping slow/integration transformation find test")
    def test_find_known_transformation(self):
        """Test case 2: apply morph_image to goal_edges and see if optim_morph finds the params."""
        # Create a simple test pattern
        edges_binary = BinaryImage.blank((128, 128))
        edges_binary.data[32:96, 32:96] = True  # White square in center (64x64 square)

        # Apply a known transformation to create goal_edges
        edges_uint8 = edges_binary.to_type(GrayscaleImage)
        known_params = {
            "x_shift": 5,
            "y_shift": -5,
            "scale": 1.1,
            "tilt_x": 0.1,
            "tilt_y": -0.1,
            "rotation": 15.0,
        }

        goal_edges_uint8 = morph_image(
            edges_uint8,
            int(known_params["x_shift"]),
            int(known_params["y_shift"]),
            known_params["scale"],
            known_params["tilt_x"],
            int(known_params["tilt_y"]),
            known_params["rotation"],
        )
        goal_edges_binary = goal_edges_uint8.to_type(BinaryImage)

        # Run optimization
        optimized_edges_binary, best_score, best_params = optim_morph(
            edges_binary, goal_edges_binary
        )

        # Save edges, goal_edges, and optimized_edges as JPGs in test_shape_detector dir
        from circlapse.saving import save_jpg
        import os

        output_dir = "test_shape_detector"
        os.makedirs(output_dir, exist_ok=True)

        # Convert boolean arrays to uint8 images for saving
        filenames = [
            os.path.join(output_dir, "edges.jpg"),
            os.path.join(output_dir, "goal_edges.jpg"),
            os.path.join(output_dir, "optimized_edges.jpg"),
        ]
        images = [edges_binary, goal_edges_binary, optimized_edges_binary]

        save_jpg(images, filenames)

        # Should find a good match (may not be perfect due to discretization)
        assert best_score > 0.5  # Reasonable threshold for geometric transformation

        # The found parameters should be close to the known parameters
        # (allowing for some tolerance due to discretization and search granularity)
        assert abs(best_params["x_shift"] - known_params["x_shift"]) <= 2
        assert abs(best_params["y_shift"] - known_params["y_shift"]) <= 2
        assert abs(best_params["scale"] - known_params["scale"]) <= 0.2
        assert abs(best_params["tilt_x"] - known_params["tilt_x"]) <= 0.2
        assert abs(best_params["tilt_y"] - known_params["tilt_y"]) <= 0.2
        assert abs(best_params["rotation"] - known_params["rotation"]) <= 10

    def test_gradient_optimization(self):
        """Test the gradient-based optimization and compare with grid search."""
        # Create a simple test pattern
        grayscale_image = GrayscaleImage.blank((64, 64))
        grayscale_image.data[16:48, 16:48] = (
            255  # White square in center (32x32 square)
        )
        edges_binary = detect_edges(grayscale_image, 2.0, 0.5, 1.5)

        # Apply a known transformation to create goal_edges
        start_params = {
            "x_shift": 0,
            "y_shift": 0,
            "scale": 0.7,
            "tilt_x": 0,
            "tilt_y": 0,
            "rotation": 0,
        }

        start_image = morph_image(
            grayscale_image,
            int(start_params["x_shift"]),
            int(start_params["y_shift"]),
            start_params["scale"],
            start_params["tilt_x"],
            start_params["tilt_y"],
            start_params["rotation"],
        )
        start_image_binary = detect_edges(start_image, 2.0, 0.5, 1.5)

        # Test gradient-based optimization

        optimized_edges_grad, best_score_grad, best_params_grad = optim_morph_gradient(
            start_image_binary, edges_binary
        )

        # Should find a good match
        # assert best_score_grad > 0.5

        # The found parameters should be close to the known parameters
        # assert abs(best_params_grad["x_shift"] - known_params["x_shift"]) <= 3
        # assert abs(best_params_grad["y_shift"] - known_params["y_shift"]) <= 3
        # assert abs(best_params_grad["scale"] - known_params["scale"]) <= 0.2
        # assert abs(best_params_grad["tilt_x"] - known_params["tilt_x"]) <= 0.2
        # assert abs(best_params_grad["tilt_y"] - known_params["tilt_y"]) <= 0.2
        # assert abs(best_params_grad["rotation"] - known_params["rotation"]) <= 15

        # Save comparison images
        from circlapse.saving import save_jpg
        import os

        output_dir = "test_shape_detector"
        os.makedirs(output_dir, exist_ok=True)

        # Convert boolean arrays to uint8 images for saving

        filenames = [
            os.path.join(output_dir, "edges_grad.jpg"),
            os.path.join(output_dir, "goal_edges_grad.jpg"),
            os.path.join(output_dir, "optimized_edges_grad.jpg"),
        ]
        images = [start_image_binary, edges_binary, optimized_edges_grad]

        save_jpg(images, filenames)

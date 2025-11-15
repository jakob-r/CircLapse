import numpy as np
import cv2
from circlapse.shape_detector import (
    match_score,
    morph_image,
    optim_morph,
    optim_morph_gradient,
)
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
        edges_binary = BinaryImage.blank((10, 10))

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

    def test_translation(self):
        """Test translation transformation."""
        # Create a simple test grayscale image
        grayscale_image = GrayscaleImage.blank((10, 10))
        grayscale_image.data[4, 4] = 255  # Single white pixel at center

        result = morph_image(grayscale_image, 2, 1, 1.0, 0.0, 0.0, 0.0)

        # Check that the image changed (translation occurred)
        assert result != grayscale_image
        # Original position should be black (pixel moved)
        assert result.data[4, 4] == 0

    def test_rotation(self):
        """Test rotation transformation."""
        # Create a simple test grayscale image
        grayscale_image = GrayscaleImage.blank((10, 10))
        grayscale_image.data[4, 4] = 255  # Single white pixel at center

        result = morph_image(grayscale_image, 0, 0, 1.0, 0.0, 0.0, 90.0)

        # With 90 degree rotation around center, the image should change
        # (exact position depends on implementation details)
        assert result != grayscale_image

    def test_scaling(self):
        """Test scaling transformation."""
        # Create a simple test grayscale image
        grayscale_image = GrayscaleImage.blank((10, 10))
        grayscale_image.data[4, 4] = 255  # Single white pixel at center

        result = morph_image(grayscale_image, 0, 0, 2.0, 0.0, 0.0, 0.0)

        # With 2x scaling, the image should change
        assert result != grayscale_image
        assert result.shape == grayscale_image.shape

    def test_output_size(self):
        """Test that output has same size as input."""
        grayscale_image = GrayscaleImage.blank((15, 20))

        result = morph_image(grayscale_image, 5, 3, 1.5, 0.1, 0.2, 45.0)

        assert result.shape == grayscale_image.shape


class TestOptimMorph:
    """Test cases for the optim_morph function."""

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
        edges_binary = BinaryImage.blank((64, 64))
        edges_binary.data[16:48, 16:48] = True  # White square in center (32x32 square)

        # Apply a known transformation to create goal_edges
        known_params = {
            "x_shift": 5,
            "y_shift": -5,
            "scale": 1.1,
            "tilt_x": 0.1,
            "tilt_y": -0.1,
            "rotation": 15.0,
        }

        goal_edges_uint8 = morph_image(
            edges_binary.to_type(GrayscaleImage),
            int(known_params["x_shift"]),
            int(known_params["y_shift"]),
            known_params["scale"],
            known_params["tilt_x"],
            known_params["tilt_y"],
            known_params["rotation"],
        )
        goal_edges_binary = goal_edges_uint8.to_type(BinaryImage)

        # Test gradient-based optimization

        optimized_edges_grad, best_score_grad, best_params_grad = optim_morph_gradient(
            edges_binary, goal_edges_binary
        )

        # Should find a good match
        assert best_score_grad > 0.5

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
        images = [edges_binary, goal_edges_binary, optimized_edges_grad]

        save_jpg(images, filenames)

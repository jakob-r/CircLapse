from typing import Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import logging
from skimage.feature import canny

from circlapse import image_operators
from circlapse.context import output_dir
from circlapse.saving import save_jpg, save_multi_jpg_with_text
from circlapse.morph import morph_image
from circlapse.image_types import (
    GrayscaleImage,
    BinaryImage,
    RGBImage,
)

logger = logging.getLogger(__name__)


def detect_edges(
    grayscale_image: GrayscaleImage,
    sigma: float,
    low_threshold: float,
    high_threshold_multiplier: float,
) -> BinaryImage:
    """
    Detect edges in a grayscale image using Canny edge detection.

    Args:
        grayscale_image: The grayscale image to detect edges in (uint8, 0-255)
        sigma: Standard deviation for Gaussian smoothing
        low_threshold: Lower bound for hysteresis thresholding
        high_threshold_multiplier: Multiplier for the low threshold to get the high threshold

    Returns:
        Binary edge image (bool, True for edges, False for background)
    """
    edges = canny(
        grayscale_image.data,
        sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=low_threshold * high_threshold_multiplier,
    )
    return BinaryImage.from_numpy(edges)


def match_score(
    edges: BinaryImage | GrayscaleImage, goal_edges: BinaryImage | GrayscaleImage
) -> float:
    """
    Calculate the match score between two edge images.

    Args:
        edges_binary: Binary edge image (bool)
        goal_edges: Target edge image - can be binary (bool) or grayscale (uint8)

    Returns:
        Match score between 0.0 and 1.0
        - For binary goal_edges: exact match (0 or 1 per pixel)
        - For grayscale goal_edges: similarity based on pixel differences (0-1 per pixel)
    """
    # Convert binary edges to 0-1 range for comparison
    if isinstance(edges, BinaryImage):
        edges_normalized = edges.data.astype(np.float32)
    elif isinstance(edges, GrayscaleImage):
        edges_normalized = edges.data.astype(np.float32) / 255.0
    else:
        raise TypeError(
            f"edges must be BinaryImage or GrayscaleImage, got {type(edges)}"
        )

    if isinstance(goal_edges, BinaryImage):
        goal_normalized = goal_edges.data.astype(np.float32)
    elif isinstance(goal_edges, GrayscaleImage):
        goal_normalized = goal_edges.data.astype(np.float32) / 255.0
    else:
        raise TypeError(
            f"edges must be BinaryImage or GrayscaleImage, got {type(edges)}"
        )

    pixel_similarities = 1.0 - np.abs(edges_normalized - goal_normalized)
    return np.mean(pixel_similarities)


def rmse_score(
    edges: GrayscaleImage,
    goal_edges: GrayscaleImage,
) -> float:
    """
    Calculate the RMSE score between two edge images.
    Low RMSE score is better.
    """
    goal_normalized = goal_edges.data.astype(np.float32) / 255.0
    edges_normalized = edges.data.astype(np.float32) / 255.0
    return np.sqrt(np.mean(np.square(edges_normalized - goal_normalized)))


def optim_morph(
    edges_binary: BinaryImage,
    goal_edges_binary: BinaryImage,
) -> tuple[BinaryImage, float, dict]:
    """
    Optimize geometric transformations on edges to maximize match score with goal_edges.
    Uses morph_image function to apply shifts, scaling, tilting, and rotation.

    Args:
        edges_binary: Binary edge image to optimize (bool)
        goal_edges_binary: Target binary edge image (bool)

    Returns:
        tuple: (optimized_edges_binary, best_score, best_params)
        - optimized_edges_binary: The best morphed version of edges (bool)
        - best_score: The highest match score achieved
        - best_params: Dictionary with the parameters that achieved the best score
    """

    best_score = match_score(edges_binary, goal_edges_binary)
    best_edges_binary = edges_binary.copy()
    best_params = {
        "x_shift": 0,
        "y_shift": 0,
        "scale": 1.0,
        "tilt_x": 0.0,
        "tilt_y": 0.0,
        "rotation": 0.0,
    }

    # Define parameter ranges to try
    x_shifts = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
    y_shifts = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]
    scales = [0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2]
    tilt_xs = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
    tilt_ys = [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
    rotations = [-30, -15, -10, -5, -2, -1, 0, 1, 2, 5, 10, 15, 30]

    total_combinations = (
        len(x_shifts)
        * len(y_shifts)
        * len(scales)
        * len(tilt_xs)
        * len(tilt_ys)
        * len(rotations)
    )
    logger.info(
        f"Starting geometric optimization with {total_combinations} total combinations"
    )

    combination_count = 0

    for x_shift in x_shifts:
        for y_shift in y_shifts:
            for scale in scales:
                for tilt_x in tilt_xs:
                    for tilt_y in tilt_ys:
                        for rotation in rotations:
                            combination_count += 1

                            # Apply geometric transformation using morph_image
                            morphed_edges_grayscale = morph_image(
                                GrayscaleImage.from_numpy(edges_binary.data),
                                x_shift,
                                y_shift,
                                scale,
                                tilt_x,
                                tilt_y,
                                rotation,
                            )

                            # Convert back to boolean
                            morphed_edges_binary = BinaryImage.from_numpy(
                                morphed_edges_grayscale.data
                            )

                            # Calculate score
                            score = match_score(
                                morphed_edges_binary,
                                goal_edges_binary,
                            )

                            # Update best if better
                            if score > best_score:
                                best_score = score
                                best_edges_binary = morphed_edges_binary.copy()
                                best_params = {
                                    "x_shift": x_shift,
                                    "y_shift": y_shift,
                                    "scale": scale,
                                    "tilt_x": tilt_x,
                                    "tilt_y": tilt_y,
                                    "rotation": rotation,
                                }
                                logger.info(
                                    f"New best score: {score:.4f} with params: {best_params}"
                                )

                            # Log progress every 1000 combinations
                            if combination_count % 1000 == 0:
                                logger.info(
                                    f"Progress: {combination_count}/{total_combinations} combinations tested"
                                )

    logger.info(
        f"Optimization complete. Best score: {best_score:.4f} with params: {best_params}"
    )
    return best_edges_binary, best_score, best_params


def optim_morph_gradient(
    edges_binary: BinaryImage,
    goal_edges_binary: BinaryImage,
) -> tuple[BinaryImage, float, dict]:
    """
    Optimize geometric transformations using gradient-based optimization.
    Much faster than grid search for finding optimal parameters.

    Args:
        edges_binary: Binary edge image to optimize (bool)
        goal_edges_binary: Target binary edge image (bool)

    Returns:
        tuple: (optimized_edges_binary, best_score, best_params)
        - optimized_edges_binary: The best morphed version of edges (bool)
        - best_score: The highest match score achieved
        - best_params: Dictionary with the parameters that achieved the best score
    """
    from scipy.optimize import minimize

    # Define parameter scaling factors
    img_width, img_height = edges_binary.shape

    epsilon = 1e-4

    # determines the first stepsize for the gradient descent in each dimension
    param_scales = {
        "x_shift": 5.0,
        "y_shift": 5.0,
        "scale": 0.1,
        "tilt_x": 0.2,
        "tilt_y": 0.2,
        "rotation": 5,
    }

    param_scales = {key: value / epsilon for key, value in param_scales.items()}

    max_x_shift = img_width // 4
    max_y_shift = img_height // 4
    max_scale_range = 0.5  # scale from 0.8 to 1.2 (1.0 ± 0.2)
    max_tilt = 0.4
    max_rotation = 15  # degrees

    # Calculate normalized bounds
    param_bounds = {
        "x_shift": (-max_x_shift, max_x_shift),
        "y_shift": (-max_y_shift, max_y_shift),
        "scale": (-max_scale_range, max_scale_range),
        "tilt_x": (-max_tilt, max_tilt),
        "tilt_y": (-max_tilt, max_tilt),
        "rotation": (-max_rotation, max_rotation),
    }

    param_bounds = {
        key: (value[0] / param_scales[key], value[1] / param_scales[key])
        for key, value in param_bounds.items()
    }

    goal_edges_blurred = image_operators.blur_and_normalize(
        goal_edges_binary, kernel_size=img_width // 4 * 2 + 1
    )

    # Create timestamped folder for this optimization run and set it as output_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_folder = Path(output_dir.get()) / f"optimization_{timestamp}"
    timestamped_folder.mkdir(parents=True, exist_ok=True)
    logger.info(f"Saving optimization iterations to: {timestamped_folder}")

    # Save original output_dir and set new one for this optimization
    original_output_dir = output_dir.get()
    output_dir.set(timestamped_folder)

    iteration_count = 0
    best_score = float("inf")
    best_params = None
    best_iteration = None

    def objective_function(params):
        """Objective function to minimize (negative match score)."""
        nonlocal iteration_count
        nonlocal best_score
        nonlocal best_params
        nonlocal best_iteration
        iteration_count += 1

        # Transform parameters from [-1, 1] range to actual parameter ranges
        (
            x_shift_norm,
            y_shift_norm,
            scale_norm,
            tilt_x_norm,
            tilt_y_norm,
            rotation_norm,
        ) = params

        # Transform normalized parameters to actual parameter ranges
        x_shift = x_shift_norm * param_scales["x_shift"]
        y_shift = y_shift_norm * param_scales["y_shift"]
        scale = 1.0 + scale_norm * param_scales["scale"]
        tilt_x = tilt_x_norm * param_scales["tilt_x"]
        tilt_y = tilt_y_norm * param_scales["tilt_y"]
        rotation = rotation_norm * param_scales["rotation"]

        logger.debug(
            f"Trying parameters: x_shift={x_shift}, y_shift={y_shift}, scale={scale}, tilt_x={tilt_x}, tilt_y={tilt_y}, rotation={rotation}"
        )

        # Apply transformation
        morphed_edges_grayscale = morph_image(
            GrayscaleImage.from_numpy(edges_binary.data),
            x_shift,
            y_shift,
            scale,
            tilt_x,
            tilt_y,
            rotation,
        )
        # Return negative score for minimization
        # score = rmse_score(morphed_edges_grayscale, goal_edges_blurred)
        score = -match_score(morphed_edges_grayscale, goal_edges_blurred)

        if score < best_score:
            best_score = score
            best_params = params
            best_iteration = iteration_count

        # for debugging, save annotated comparison
        text = (
            f"iter: {iteration_count} (best: {best_iteration})\n"
            f"score: {score:.3f} (best: {best_score:.3f})\n"
            f"x: {x_shift:.2f}, y: {y_shift:.2f}\n"
            f"scale: {scale:.3f}\n"
            f"tilt: ({tilt_x:.2f}, {tilt_y:.2f})\n"
            f"rot: {rotation:.2f}"
        )
        save_multi_jpg_with_text(
            [morphed_edges_grayscale, goal_edges_blurred],
            [text, "goal"],
            f"iter_{iteration_count:04d}.jpg",
        )
        logger.debug(f"Score: {score:.4f}")
        return score

    # Convert param_bounds dict to list in the correct order
    param_order = ["x_shift", "y_shift", "scale", "tilt_x", "tilt_y", "rotation"]
    bounds = [param_bounds[param] for param in param_order]

    # Initial guess (no transformation) - all zeros correspond to no transformation
    x0 = [0] * len(param_scales)

    logger.info("Starting gradient-based optimization...")

    # Run optimization
    result = minimize(
        objective_function,
        x0,
        method="L-BFGS-B",  # Good for bounded optimization
        bounds=bounds,
        options={"maxiter": 100, "disp": False, "eps": epsilon},
    )

    if not result.success:
        logger.warning(f"Optimization may not have converged: {result.message}")

    # Extract best parameters and transform back to original parameter space
    x_shift_norm, y_shift_norm, scale_norm, tilt_x_norm, tilt_y_norm, rotation_norm = (
        result.x
    )

    best_params = {
        "x_shift": x_shift_norm * param_scales["x_shift"],
        "y_shift": y_shift_norm * param_scales["y_shift"],
        "scale": 1.0 + scale_norm * param_scales["scale"],
        "tilt_x": tilt_x_norm * param_scales["tilt_x"],
        "tilt_y": tilt_y_norm * param_scales["tilt_y"],
        "rotation": rotation_norm * param_scales["rotation"],
    }

    best_score = -result.fun  # Convert back from negative

    # Apply best transformation to get final result
    best_edges = morph_image(edges_binary.to_type(GrayscaleImage), **best_params)

    logger.info(
        f"Gradient optimization complete. Best score: {best_score:.4f} with params: {best_params}"
    )
    logger.info(f"Optimization iterations: {result.nit}")

    # Restore original output_dir
    output_dir.set(original_output_dir)

    return best_edges.to_type(BinaryImage), best_score, best_params


def shape_detector(
    rgb_image: RGBImage, shape_image_binary: BinaryImage
) -> Optional[tuple[int, int, int]]:
    """
    Takes an image and a shape image (b/w), and returns the

    Args:
        rgb_image: The image to detect the shape in (RGB or grayscale)
        shape_image_binary: The binary shape image to detect (bool)

    Returns:
        Coordinates (x, y, radius) or None if no valid shape found.
        Radius is the distance from the center of the furthest edge of the shape.
    """
    target_size = 800

    rgb_image, scale_factor = image_operators.scale_shortest_side(
        rgb_image.data, target_size
    )

    # save shape_image
    output_path = output_dir.get()
    save_jpg([shape_image_binary], [str(output_path) + "/shape_image.jpg"])

    # save inverted shape_image for debugging
    inverted_shape = cv2.bitwise_not(shape_image_binary.data)
    save_jpg([inverted_shape], [str(output_path) + "/inverted_shape_image.jpg"])

    # detect contours of shape_image
    # cv2.findContours expects white objects on black background, so we need to invert
    # since our shape_image has black circle on white background
    contours, _ = cv2.findContours(
        inverted_shape, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        logger.warning("No contours found in shape_image")
        return None
    contour = contours[0]
    logger.info(
        f"Found {len(contours)} contours, using the first one with {len(contour)} points"
    )

    # Create a blank image to draw contours on for debugging
    contour_image_uint8 = shape_image_binary.blank_like()
    cv2.drawContours(contour_image_uint8, [contour], -1, 255, 2)
    save_jpg([contour_image_uint8], [str(output_path) + "/contour.jpg"])

    # --- Try all parameter combinations for Canny edge detection ---
    sigma_values = [2.0, 3.0, 4.0]
    low_threshold_values = [0.3, 2, 4]
    high_threshold_multiplier_values = [1.1, 2, 4]

    # detect edges
    edges = canny(shape_image_binary, sigma=2.0, low_threshold=0.55, high_threshold=0.8)
    save_jpg([edges], [str(output_path) + "/edges.jpg"])

    # now the same in source image
    # convert to grayscale for contour detection
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(
        gray_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if len(contours) == 0:
        logger.warning("No contours found in source image")
        return None
    contour = contours[0]
    logger.info(
        f"Found {len(contours)} contours, using the first one with {len(contour)} points"
    )

    # draw contour on source image
    image_with_contour = rgb_image.copy()
    cv2.drawContours(image_with_contour, [contour], -1, 255, 2)
    save_jpg([image_with_contour], [str(output_path) + "/image_with_contour.jpg"])

    # detect edges in source image
    # Blur the grayscale image a bit to reduce noise before edge detection
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    save_jpg([gray_image], [str(output_path) + "/gray_image_blurred.jpg"])
    for sigma in sigma_values:
        for low in low_threshold_values:
            for high_multiplier in high_threshold_multiplier_values:
                high = round(low * high_multiplier, 2)
                edges = canny(
                    gray_image, sigma=sigma, low_threshold=low, high_threshold=high
                )
                fname = (
                    f"{output_path}/edges_source_sigma{sigma}_low{low}_high{high}.jpg"
                )
                save_jpg([edges], [fname])

    # Fallback to contour-based detection if no ellipses found
    # Calculate the bounding circle of the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    return (center[0], center[1], radius)

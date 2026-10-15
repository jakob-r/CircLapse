from typing import Optional
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np
import logging
from scipy.ndimage import distance_transform_edt
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

def detect_edges_optimized(
    grayscale_image: GrayscaleImage,
) -> BinaryImage:
    """
    Detect edges in a grayscale image using Canny edge detection.
    Uses grid search to find the parameters that give the 
    """
    v = np.median(grayscale_image.data)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    blurred = cv2.GaussianBlur(grayscale_image.data, (5, 5), 0)
    edges = cv2.Canny(blurred, lower, upper)
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
    target_edges: GrayscaleImage,
) -> float:
    """
    Calculate the RMSE score between two edge images.
    Low RMSE score is better.
    """
    target_normalized = target_edges.data.astype(np.float32) / 255.0
    edges_normalized = edges.data.astype(np.float32) / 255.0
    return np.sqrt(np.mean(np.square(edges_normalized - target_normalized)))


def distance_transform_score(
    edges: BinaryImage,
    target_edges: BinaryImage,
    mask: BinaryImage | None = None,
) -> float:
    """
    Calculate the distance transform score between two edge images.
    Low distance transform score is better.
    Args:
        edges: Binary edge image to score (bool)
        target_edges: Target binary edge image (bool)
        mask: If given, only pixels where mask is True are included in scoring.
              Pixels where mask is False receive a penalty.
    """
    if mask is None:
        mask = BinaryImage.from_numpy(np.ones(edges.shape, dtype=bool))
    inds_excluded = np.where(~mask.data)
    dist_input = distance_transform_edt(np.logical_not(edges.data))
    dist_target = distance_transform_edt(np.logical_not(target_edges.data))
    dist_diff = dist_input - dist_target
    dist_diff[inds_excluded] = np.min(edges.shape)
    loss = np.mean(np.square(dist_diff))
    return loss


def local_distance_transform_score(
    edges: BinaryImage,
    target_edges: BinaryImage,
    mask: BinaryImage | None = None,
    local_radius: int | None = None,
) -> float:
    """
    Calculate the distance transform score between two edge images.
    But only evaluate the distance transform around the target edges.
    Low local distance transform score is better.

    Args:
        edges: Binary edge image to score (bool)
        target_edges: Target binary edge image (bool)
        mask: If given, only pixels where mask is True are included in scoring.
              Pixels where mask is False receive a penalty.
        local_radius: Only evaluate within this distance of target edges.
    """
    if mask is None:
        mask = BinaryImage.from_numpy(np.ones(edges.shape, dtype=bool))
    if local_radius is None:
        local_radius = max(int(np.min(edges.shape) // 6), 4)
    dist_input = distance_transform_edt(np.logical_not(edges.data))
    dist_target = distance_transform_edt(np.logical_not(target_edges.data))
    ## // DEBUG
    # Scale dist_target to 0-255 range for saving as 8-bit grayscale image
    dist_target_scaled = dist_target.astype(np.float32)
    dist_input_scaled = dist_input.astype(np.float32)
    # Normalize to 0-255 (avoid division by zero)
    if dist_target_scaled.max() > 0:
        dist_target_scaled = 255 * dist_target_scaled / dist_target_scaled.max()
    dist_target_uint8 = dist_target_scaled.astype(np.uint8)
    if dist_input_scaled.max() > 0:
        dist_input_scaled = 255 * dist_input_scaled / dist_input_scaled.max()
    dist_input_uint8 = dist_input_scaled.astype(np.uint8)
    # Save the grayscale distance map
    try:
        from circlapse.saving import save_jpg
        from circlapse.image_types import GrayscaleImage
        # Compose a name indicating this is the local distance transform map.
        import os
        fname_target = "dist_target.jpg"
        fname_input = "dist_input.jpg"
        # Optionally try to save in a debug or output folder
        save_jpg([GrayscaleImage.from_numpy(dist_target_uint8)], [fname_target])
        save_jpg([GrayscaleImage.from_numpy(dist_input_uint8)], [fname_input])
    except Exception as e:
        # Do not crash if save fails
        pass
    ## // DEBUG
    target_mask = dist_target < local_radius
    dist_diff = dist_input - dist_target
    inds_excluded = np.where(~mask.data)
    dist_diff[inds_excluded] = np.min(edges.shape)
    inds_target_mask = np.where(target_mask)
    loss = np.mean(np.square(dist_diff[inds_target_mask]))
    return loss


def asymmetric_distance_loss(
    edges: BinaryImage,
    target_edges: BinaryImage,
    mask: BinaryImage | None = None,
    lambda_weight: float = 0.5,
) -> float:
    """
    Calculate an asymmetric distance loss between two edge images.
    Low loss is better.

    This loss has two components:
    1. Precision Loss: Punishes input edge pixels that are far from target edges.
       Noise pixels get a huge penalty here.
    2. Coverage Loss: Ensures we actually overlap with the target (using Dice).

    Args:
        edges: Binary edge image to score (bool)
        target_edges: Target binary edge image (bool)
        mask: If given, only pixels where mask is True are included in scoring.
              Pixels where mask is False receive a penalty.
        lambda_weight: Weight between precision (1.0) and coverage (0.0) loss. Default 0.5.

    Returns:
        Combined loss value (lower is better)
    """
    if mask is None:
        mask = BinaryImage.from_numpy(np.ones(edges.shape, dtype=bool))

    mask_data = mask.data
    inds_excluded = np.where(~mask_data)

    # 1. Compute distance map of the TARGET only
    # Invert target because distance_transform_edt computes distance to False pixels
    inv_target = np.logical_not(target_edges.data)
    dist_target = distance_transform_edt(inv_target)

    # 2. Precision Loss: Punish input edge pixels that are far from target
    edges_float = edges.data.astype(np.float32)
    precision_map = edges_float * dist_target

    # Apply penalty to excluded regions (mask=False)
    # The bigger the excluded area, the bigger the penalty contribution (through the mean)
    mask_penalty_value = np.min(edges.shape)
    precision_map[inds_excluded] = mask_penalty_value

    precision_loss = np.mean(precision_map)

    # 3. Coverage Loss: Use Dice to ensure we actually overlap with target
    # For excluded regions, treat them as non-edges for dice calculation
    edges_for_dice = edges.data.copy()
    target_for_dice = target_edges.data.copy()
    edges_for_dice[inds_excluded] = False
    target_for_dice[inds_excluded] = False

    intersection = np.sum(edges_for_dice & target_for_dice)
    dice_score = (2.0 * intersection) / (
        np.sum(edges_for_dice) + np.sum(target_for_dice) + 1e-6
    )
    coverage_loss = 1.0 - dice_score

    # Combine losses
    return (lambda_weight * precision_loss) + ((1 - lambda_weight) * coverage_loss)


def score(
    edges: BinaryImage,
    target_edges: BinaryImage,
    mask: BinaryImage | None = None,
) -> float:
    return local_distance_transform_score(edges, target_edges, mask=mask)
    # return distance_transform_score(edges, target_edges, mask=mask)
    # return asymmetric_distance_loss(edges, target_edges, mask)


def optim_morph_grid(
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
    method: str = "differential_evolution",
    maxiter: int = 100,
    popsize: int = 15,
    save_iterations: bool = True,
) -> tuple[BinaryImage, float, dict]:
    """
    Optimize geometric transformations using global optimization methods.
    Uses population-based methods to avoid local minima.

    Args:
        edges_binary: Binary edge image to optimize (bool)
        goal_edges_binary: Target binary edge image (bool)
        method: Optimization method - one of:
            - "differential_evolution" (default): Population-based global optimizer
            - "basinhopping": Local optimizer with random jumps
            - "dual_annealing": Simulated annealing approach
            - "cma": CMA-ES (requires cma package)
        maxiter: Maximum number of iterations/generations
        popsize: Population size (for differential_evolution and cma)
        save_iterations: Whether to save iteration images for debugging

    Returns:
        tuple: (optimized_edges_binary, best_score, best_params)
        - optimized_edges_binary: The best morphed version of edges (bool)
        - best_score: The highest match score achieved (lower is better for RMSE)
        - best_params: Dictionary with the parameters that achieved the best score
    """
    from scipy.optimize import differential_evolution, basinhopping, dual_annealing

    img_width, img_height = edges_binary.shape

    max_x_shift = img_width // 4
    max_y_shift = img_height // 4
    max_scale_range = 0.5  # scale from 0.5 to 1.5 (1.0 ± 0.5)
    max_tilt = 0.4
    max_rotation = 15  # degrees

    # Bounds in actual parameter space (not normalized)
    param_bounds = [
        (-max_x_shift, max_x_shift),  # x_shift
        (-max_y_shift, max_y_shift),  # y_shift
        (1.0 - max_scale_range, 1.0 + max_scale_range),  # scale
        (-max_tilt, max_tilt),  # tilt_x
        (-max_tilt, max_tilt),  # tilt_y
        (-max_rotation, max_rotation),  # rotation
    ]

    # Create timestamped folder for this optimization run
    timestamped_folder = None
    original_output_dir = None
    if save_iterations:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        timestamped_folder = Path(output_dir.get()) / f"optimization_{timestamp}"
        timestamped_folder.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving optimization iterations to: {timestamped_folder}")
        original_output_dir = output_dir.get()
        output_dir.set(timestamped_folder)

    iteration_count = 0
    best_score = float("inf")
    best_params_array = None
    best_iteration = None

    def objective_function(params):
        """Objective function to minimize (RMSE score)."""
        nonlocal iteration_count, best_score, best_params_array, best_iteration
        iteration_count += 1

        x_shift, y_shift, scale, tilt_x, tilt_y, rotation = params

        logger.debug(
            f"Trying parameters: x_shift={x_shift:.2f}, y_shift={y_shift:.2f}, "
            f"scale={scale:.3f}, tilt_x={tilt_x:.3f}, tilt_y={tilt_y:.3f}, rotation={rotation:.2f}"
        )

        # Apply transformation
        morph_bg_color = 66
        morphed_edges_grayscale = morph_image(
            GrayscaleImage.from_numpy(edges_binary.data),
            x_shift,
            y_shift,
            scale,
            tilt_x,
            tilt_y,
            rotation,
            background_color=morph_bg_color,
        )
        # mask=True means include in scoring (valid pixels, not background)
        mask = BinaryImage.from_numpy(morphed_edges_grayscale.data != morph_bg_color)
        new_score = score(
            morphed_edges_grayscale.to_type(BinaryImage), goal_edges_binary, mask=mask
        )

        if new_score < best_score:
            best_score = new_score
            best_params_array = params.copy()
            best_iteration = iteration_count
            logger.info(
                f"New best score: {new_score:.4f} at iteration {iteration_count}"
            )

        # Save iteration image for debugging
        if save_iterations:
            text = (
                f"iter: {iteration_count} (best: {best_iteration})\n"
                f"score: {new_score:.3f} (best: {best_score:.3f})\n"
                f"x: {x_shift:.2f}, y: {y_shift:.2f}\n"
                f"scale: {scale:.3f}\n"
                f"tilt: ({tilt_x:.2f}, {tilt_y:.2f})\n"
                f"rot: {rotation:.2f}"
            )
            save_multi_jpg_with_text(
                [morphed_edges_grayscale, goal_edges_binary],
                [text, "goal"],
                f"iter_{iteration_count:04d}.jpg",
            )

        return new_score

    logger.info(f"Starting optimization with method: {method}")

    # Run the selected optimization method
    if method == "differential_evolution":
        result = differential_evolution(
            objective_function,
            bounds=param_bounds,
            maxiter=maxiter,
            popsize=popsize,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            disp=False,
            polish=True,  # Use local optimizer to polish final result
            updating="deferred",  # Better for parallel evaluation
            workers=1,  # Set to -1 for parallel (but loses iteration saving order)
        )
        final_params = result.x
        final_score = result.fun
        success = result.success
        message = result.message

    elif method == "basinhopping":
        # Start from center of bounds
        x0 = [(b[0] + b[1]) / 2 for b in param_bounds]

        # Custom step function to respect bounds
        class BoundedStep:
            def __init__(self, bounds, stepsize=0.5):
                self.bounds = bounds
                self.stepsize = stepsize

            def __call__(self, x):
                ranges = [b[1] - b[0] for b in self.bounds]
                new_x = (
                    x
                    + np.random.uniform(-self.stepsize, self.stepsize, len(x)) * ranges
                )
                # Clip to bounds
                for i, (low, high) in enumerate(self.bounds):
                    new_x[i] = np.clip(new_x[i], low, high)
                return new_x

        result = basinhopping(
            objective_function,
            x0,
            niter=maxiter,
            T=1.0,
            stepsize=0.3,
            minimizer_kwargs={
                "method": "L-BFGS-B",
                "bounds": param_bounds,
            },
            take_step=BoundedStep(param_bounds),
            seed=42,
        )
        final_params = result.x
        final_score = result.fun
        success = True  # basinhopping doesn't have success attribute
        message = f"Completed {maxiter} basin-hopping iterations"

    elif method == "dual_annealing":
        result = dual_annealing(
            objective_function,
            bounds=param_bounds,
            maxiter=maxiter,
            seed=42,
            no_local_search=False,
        )
        final_params = result.x
        final_score = result.fun
        success = result.success
        message = result.message

    elif method == "cma":
        try:
            import cma
        except ImportError:
            raise ImportError(
                "CMA-ES requires the 'cma' package. Install with: pip install cma"
            )

        # Start from center of bounds
        x0 = [(b[0] + b[1]) / 2 for b in param_bounds]
        # Initial standard deviation (roughly 1/4 of each range)
        sigma0 = 0.25

        # Scale parameters to [0, 1] for CMA-ES
        def scale_to_unit(params):
            return [(p - b[0]) / (b[1] - b[0]) for p, b in zip(params, param_bounds)]

        def scale_from_unit(params):
            return [p * (b[1] - b[0]) + b[0] for p, b in zip(params, param_bounds)]

        def objective_scaled(params_unit):
            params = scale_from_unit(params_unit)
            return objective_function(np.array(params))

        x0_unit = scale_to_unit(x0)

        es = cma.CMAEvolutionStrategy(
            x0_unit,
            sigma0,
            {
                "bounds": [0, 1],
                "maxiter": maxiter,
                "popsize": popsize,
                "seed": 42,
                "verbose": -9,  # Suppress output
            },
        )
        es.optimize(objective_scaled)

        final_params = np.array(scale_from_unit(es.result.xbest))
        final_score = es.result.fbest
        success = not es.stop()
        message = f"CMA-ES terminated: {es.stop()}"

    else:
        raise ValueError(
            f"Unknown optimization method: {method}. "
            f"Choose from: differential_evolution, basinhopping, dual_annealing, cma"
        )

    if not success:
        logger.warning(f"Optimization may not have converged: {message}")

    # Extract best parameters
    x_shift, y_shift, scale, tilt_x, tilt_y, rotation = final_params

    best_params = {
        "x_shift": float(x_shift),
        "y_shift": float(y_shift),
        "scale": float(scale),
        "tilt_x": float(tilt_x),
        "tilt_y": float(tilt_y),
        "rotation": float(rotation),
    }

    # Apply best transformation to get final result
    best_edges = morph_image(
        edges_binary.to_type(GrayscaleImage), **best_params, background_color=0
    )

    logger.info(
        f"Optimization complete ({method}). Best score: {final_score:.4f} with params: {best_params}"
    )
    logger.info(f"Total function evaluations: {iteration_count}")

    # Restore original output_dir
    if original_output_dir is not None:
        output_dir.set(original_output_dir)

    return best_edges.to_type(BinaryImage), final_score, best_params


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
        rgb_image, target_size
    )

    # save shape_image
    output_path = output_dir.get()
    save_jpg([shape_image_binary], [str(output_path) + "/shape_image.jpg"])

    # save inverted shape_image for debugging
    inverted_shape = cv2.bitwise_not(shape_image_binary.data)
    save_jpg(
        [BinaryImage.from_numpy(inverted_shape)],
        [str(output_path) + "/inverted_shape_image.jpg"],
    )

    # detect contours of shape_image
    # cv2.findContours expects white objects on black background, so we need to invert
    # since our shape_image has black circle on white background
    contours, _ = cv2.findContours(
        image=inverted_shape, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE
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
    cv2.drawContours(contour_image_uint8.data, [contour], -1, 255, 2)
    save_jpg([contour_image_uint8], [str(output_path) + "/contour.jpg"])

    # --- Try all parameter combinations for Canny edge detection ---
    sigma_values = [2.0, 3.0, 4.0]
    low_threshold_values = [0.3, 2, 4]
    high_threshold_multiplier_values = [1.1, 2, 4]

    # detect edges
    edges = canny(
        shape_image_binary.data, sigma=2.0, low_threshold=0.55, high_threshold=0.8
    )
    save_jpg([BinaryImage.from_numpy(edges)], [str(output_path) + "/edges.jpg"])

    # now the same in source image
    # convert to grayscale for contour detection
    gray_image = cv2.cvtColor(rgb_image.data, cv2.COLOR_BGR2GRAY)
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
    cv2.drawContours(image_with_contour.data, [contour], -1, 255, 2)
    save_jpg([image_with_contour], [str(output_path) + "/image_with_contour.jpg"])

    # detect edges in source image
    # Blur the grayscale image a bit to reduce noise before edge detection
    gray_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    save_jpg(
        [GrayscaleImage.from_numpy(gray_image)],
        [str(output_path) + "/gray_image_blurred.jpg"],
    )
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
                save_jpg([BinaryImage.from_numpy(edges)], [fname])

    # Fallback to contour-based detection if no ellipses found
    # Calculate the bounding circle of the contour
    (x, y), radius = cv2.minEnclosingCircle(contour)
    center = (int(x), int(y))
    radius = int(radius)

    return (center[0], center[1], radius)


if __name__ == "__main__":
    dimension = 128
    input_image = cv2.imread("examples/example_in/later/squareandcircle.jpeg")
    input_image_rgb = RGBImage.from_numpy(input_image)
    input_image_cropped = image_operators.crop_scale(input_image_rgb, dimension, dimension)
    
    target_image = GrayscaleImage.blank((dimension, dimension))
    # square that covers 3/4 of the image
    square_size = int((3 / 4) * dimension)
    offset = (dimension - square_size) // 2
    target_image.data[offset:offset + square_size, offset:offset + square_size] = 255

    target_edges = detect_edges(target_image, 2.0, 0.5, 1.5)

    input_image_grayscale = input_image_cropped.to_type(GrayscaleImage)
    input_image_edges = detect_edges_optimized(input_image_grayscale)
    final_image, final_score, final_params = optim_morph_gradient(input_image_edges, target_edges)
    
    # transform input image 
    transformed_image = morph_image(input_image_cropped, **final_params, background_color=0)

    pics = [
        (input_image_cropped, "input_image_cropped.jpg"),
        (target_image, "target_image.jpg"),
        (target_edges, "target_edges.jpg"),
        (input_image_edges, "input_image_edges.jpg"),
        (final_image, "final_image.jpg"),
        (transformed_image, "transformed_image.jpg"),
    ]

    save_jpg(
        [pic[0] for pic in pics],
        [pic[1] for pic in pics],
    )
    
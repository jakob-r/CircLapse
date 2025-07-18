import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging
from circelizer import image_operators, settings
from circelizer.context import output_dir

logger = logging.getLogger(__name__)

def detect_circle(image: np.ndarray, image_name: str = "debug", target_size: int = 800, min_distance_to_border: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """
    Detect the largest circle in the image using Hough Circle Transform.
    
    Args:
        image: Input image as numpy array
        image_name: Name for debug image (used when DEBUG=True)
        target_size: Target size for the longest side (default: 800px)
        
    Returns:
        Tuple of (x, y, radius) of the detected circle, or None if no circle found
    """
    # Scale image to consistent size for better circle detection
    image = image.copy()
    height, width = image.shape[:2]
    scale_factor = target_size / max(height, width)
    
    # scale image to target size
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    shortest_side = min(new_width, new_height)
    image = cv2.resize(image, (new_width, new_height))
    logger.debug(f"Scaled image from {width}x{height} to {new_width}x{new_height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    
    # Detect circles using Hough Circle Transform
    # Parameters optimized for ~800px images
    circles_raw = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=100,   # Minimum distance between circles
        param1=30,    # Edge detection threshold
        param2=100,    # Accumulator threshold - adjusted for scaled images
        minRadius=shortest_side // 6, # Minimum radius - adjusted for scaled images
        maxRadius=shortest_side // 2 # Maximum radius - adjusted for scaled images
    )

    circles = circles_raw
    
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")

        # filter out circles too close to the border
        circle_shares = [image_operators.circle_share(image, circle) for circle in circles]
        too_big = [share > (1 - min_distance_to_border) for share in circle_shares]
        small_circles = [circle for circle, too_big in zip(circles, too_big) if not too_big]
        if len(small_circles) == 0:
            logger.info(f"All {sum(too_big)} circles are too close to the border. Returning None.")
            return None
        
        big_circles = [circle for circle, too_big in zip(circles, too_big) if too_big]
        
        # Return the largest circle (assuming it's the main object)
        largest_circle = max(small_circles, key=lambda x: x[2])
        max_circle_share = image_operators.circle_share(image, largest_circle)

        # Save debug image if DEBUG is enabled
        if settings.DEBUG:
            debug_image = image.copy()
            for (x, y, radius) in big_circles:
                # Draw circle outline
                cv2.circle(debug_image, (x, y), radius, (0, 0, 120), 2)
                # Draw center point
                cv2.circle(debug_image, (x, y), 2, (0, 0, 120), 3)
            for (x, y, radius) in small_circles:
                # Draw circle outline
                cv2.circle(debug_image, (x, y), radius, (0, 255, 0), 2)
                # Draw center point
                cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)

            # draw biggest circle
            cv2.circle(debug_image, (largest_circle[0], largest_circle[1]), largest_circle[2], (255, 255, 255), 2)
            cv2.putText(debug_image, f"Max circle share: {max_circle_share:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

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
        return (original_x, original_y, original_radius)
    
    return None


def detect_circle_contour_based(image: np.ndarray, image_name: str = "debug", target_size: int = 800, min_distance_to_border: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """
    Detect circles using contour detection and ellipse fitting - more robust than Hough transform.
    
    Args:
        image: Input image as numpy array
        image_name: Name for debug image (used when DEBUG=True)
        target_size: Target size for the longest side (default: 800px)
        min_distance_to_border: Minimum distance from border as fraction of image size
        
    Returns:
        Tuple of (x, y, radius) of the detected circle, or None if no circle found
    """
    # Scale image to consistent size
    image = image.copy()
    height, width = image.shape[:2]
    scale_factor = target_size / max(height, width)
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    shortest_side = min(new_width, new_height)
    image = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply adaptive thresholding for better edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Morphological operations to clean up the image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    circles = []
    min_radius = shortest_side // 8
    max_radius = shortest_side // 2
    
    for contour in contours:
        # Filter by contour area
        area = cv2.contourArea(contour)
        if area < np.pi * min_radius**2 or area > np.pi * max_radius**2:
            continue
            
        # Fit ellipse to contour
        if len(contour) >= 5:  # Need at least 5 points for ellipse fitting
            try:
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                
                # Check if it's approximately circular (axes are similar)
                major_axis = max(axes)
                minor_axis = min(axes)
                aspect_ratio = minor_axis / major_axis
                
                if aspect_ratio > 0.7:  # Allow some deviation from perfect circle
                    radius = int((major_axis + minor_axis) / 4)  # Average radius
                    x, y = int(center[0]), int(center[1])
                    
                    # Check distance to border
                    circle_share = image_operators.circle_share(image, (x, y, radius))
                    if circle_share <= (1 - min_distance_to_border):
                        circles.append((x, y, radius))
            except:
                continue
    
    if not circles:
        return None
    
    # Return the largest circle
    largest_circle = max(circles, key=lambda x: x[2])
    
    # Debug visualization
    if settings.DEBUG:
        debug_image = image.copy()
        for (x, y, radius) in circles:
            cv2.circle(debug_image, (x, y), radius, (0, 255, 0), 2)
            cv2.circle(debug_image, (x, y), 2, (0, 0, 255), 3)
        
        cv2.circle(debug_image, (largest_circle[0], largest_circle[1]), largest_circle[2], (255, 255, 255), 3)
        cv2.putText(debug_image, f"Contour-based detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        debug_path = output_dir.get() / "debug" / f"{image_name}_contour_circles.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_image)
    
    # Scale back to original size
    x, y, radius = largest_circle
    original_x = int(x / scale_factor)
    original_y = int(y / scale_factor)
    original_radius = int(radius / scale_factor)
    return (original_x, original_y, original_radius)


def detect_circle_ransac(image: np.ndarray, image_name: str = "debug", target_size: int = 800, min_distance_to_border: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """
    Detect circles using RANSAC-based circle fitting - robust to outliers.
    
    Args:
        image: Input image as numpy array
        image_name: Name for debug image (used when DEBUG=True)
        target_size: Target size for the longest side (default: 800px)
        min_distance_to_border: Minimum distance from border as fraction of image size
        
    Returns:
        Tuple of (x, y, radius) of the detected circle, or None if no circle found
    """
    # Scale image to consistent size
    image = image.copy()
    height, width = image.shape[:2]
    scale_factor = target_size / max(height, width)
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    shortest_side = min(new_width, new_height)
    image = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale and detect edges
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Find edge points
    edge_points = np.column_stack(np.where(edges > 0))
    if len(edge_points) < 10:
        return None
    
    # RANSAC parameters
    max_iterations = 100
    min_inliers = 20
    threshold = 3.0
    min_radius = shortest_side // 8
    max_radius = shortest_side // 2
    
    best_circle = None
    best_inliers = 0
    
    for _ in range(max_iterations):
        # Randomly sample 3 points
        if len(edge_points) < 3:
            break
            
        indices = np.random.choice(len(edge_points), 3, replace=False)
        p1, p2, p3 = edge_points[indices]
        
        # Calculate circle from 3 points
        try:
            # Convert to float for better precision
            p1, p2, p3 = p1.astype(float), p2.astype(float), p3.astype(float)
            
            # Calculate perpendicular bisectors
            mid1 = (p1 + p2) / 2
            mid2 = (p2 + p3) / 2
            
            # Direction vectors
            dir1 = p2 - p1
            dir2 = p3 - p2
            
            # Perpendicular vectors
            perp1 = np.array([-dir1[1], dir1[0]])
            perp2 = np.array([-dir2[1], dir2[0]])
            
            # Normalize
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = perp2 / np.linalg.norm(perp2)
            
            # Solve for center (intersection of perpendicular bisectors)
            A = np.vstack([perp1, -perp2])
            b = np.array([np.dot(perp1, mid1), np.dot(perp2, mid2)])
            
            center = np.linalg.solve(A, b)
            radius = np.linalg.norm(center - p1)
            
            # Check radius constraints
            if min_radius <= radius <= max_radius:
                # Count inliers
                distances = np.abs(np.linalg.norm(edge_points - center, axis=1) - radius)
                inliers = np.sum(distances < threshold)
                
                if inliers > best_inliers and inliers >= min_inliers:
                    best_inliers = inliers
                    best_circle = (int(center[1]), int(center[0]), int(radius))
                    
        except (np.linalg.LinAlgError, ValueError):
            continue
    
    if best_circle is None:
        return None
    
    # Check distance to border
    circle_share = image_operators.circle_share(image, best_circle)
    if circle_share > (1 - min_distance_to_border):
        return None
    
    # Debug visualization
    if settings.DEBUG:
        debug_image = image.copy()
        cv2.circle(debug_image, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 2)
        cv2.circle(debug_image, (best_circle[0], best_circle[1]), 2, (0, 0, 255), 3)
        cv2.putText(debug_image, f"RANSAC detection (inliers: {best_inliers})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        debug_path = output_dir.get() / "debug" / f"{image_name}_ransac_circles.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_image)
    
    # Scale back to original size
    x, y, radius = best_circle
    original_x = int(x / scale_factor)
    original_y = int(y / scale_factor)
    original_radius = int(radius / scale_factor)
    return (original_x, original_y, original_radius)


def detect_circle_gradient_based(image: np.ndarray, image_name: str = "debug", target_size: int = 800, min_distance_to_border: float = 0.05) -> Optional[Tuple[int, int, int]]:
    """
    Detect circles using gradient-based approach - more robust than Hough transform.
    
    Args:
        image: Input image as numpy array
        image_name: Name for debug image (used when DEBUG=True)
        target_size: Target size for the longest side (default: 800px)
        min_distance_to_border: Minimum distance from border as fraction of image size
        
    Returns:
        Tuple of (x, y, radius) of the detected circle, or None if no circle found
    """
    # Scale image to consistent size
    image = image.copy()
    height, width = image.shape[:2]
    scale_factor = target_size / max(height, width)
    
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    shortest_side = min(new_width, new_height)
    image = cv2.resize(image, (new_width, new_height))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate gradients
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    direction = np.arctan2(grad_y, grad_x)
    
    # Threshold gradient magnitude
    threshold = np.percentile(magnitude, 80)
    edge_mask = magnitude > threshold
    
    # Parameters for circle detection
    min_radius = shortest_side // 8
    max_radius = shortest_side // 2
    radius_step = target_size // 100
    
    best_circle = None
    best_score = 0
    
    # Search for circles using gradient information
    for radius in range(min_radius, max_radius, radius_step):
        # Create circle template
        y_coords, x_coords = np.ogrid[:new_height, :new_width]
        
        for center_y in range(radius, new_height - radius, 5):
            for center_x in range(radius, new_width - radius, 5):
                # Calculate circle points
                circle_mask = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= radius**2
                circle_edge = (x_coords - center_x)**2 + (y_coords - center_y)**2 <= (radius + 2)**2
                circle_edge = circle_edge & ~((x_coords - center_x)**2 + (y_coords - center_y)**2 <= (radius - 2)**2)
                
                # Calculate score based on gradient alignment
                if np.sum(circle_edge) > 0:
                    # Get gradient directions at circle edge
                    edge_points = np.where(circle_edge & edge_mask)
                    if len(edge_points[0]) > 10:
                        scores = []
                        for i in range(len(edge_points[0])):
                            y, x = edge_points[0][i], edge_points[1][i]
                            # Expected gradient direction (radial)
                            expected_dir = np.arctan2(y - center_y, x - center_x)
                            actual_dir = direction[y, x]
                            
                            # Calculate alignment score
                            angle_diff = abs(expected_dir - actual_dir)
                            angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                            score = np.cos(angle_diff)
                            scores.append(score)
                        
                        avg_score = np.mean(scores)
                        if avg_score > best_score:
                            best_score = avg_score
                            best_circle = (center_x, center_y, radius)
    
    if best_circle is None or best_score < 0.3:  # Minimum score threshold
        return None
    
    # Check distance to border
    circle_share = image_operators.circle_share(image, best_circle)
    if circle_share > (1 - min_distance_to_border):
        return None
    
    # Debug visualization
    if settings.DEBUG:
        debug_image = image.copy()
        cv2.circle(debug_image, (best_circle[0], best_circle[1]), best_circle[2], (0, 255, 0), 2)
        cv2.circle(debug_image, (best_circle[0], best_circle[1]), 2, (0, 0, 255), 3)
        cv2.putText(debug_image, f"Gradient-based detection (score: {best_score:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        debug_path = output_dir.get() / "debug" / f"{image_name}_gradient_circles.jpg"
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path), debug_image)
    
    # Scale back to original size
    x, y, radius = best_circle
    original_x = int(x / scale_factor)
    original_y = int(y / scale_factor)
    original_radius = int(radius / scale_factor)
    return (original_x, original_y, original_radius)

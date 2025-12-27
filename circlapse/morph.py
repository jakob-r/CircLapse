import cv2
import numpy as np

from circlapse.image_types import GrayscaleImage, RGBImage


def morph_image(
    image: GrayscaleImage | RGBImage,
    x_shift: int,
    y_shift: int,
    scale: float,
    tilt_x: float,
    tilt_y: float,
    rotation: float,
) -> GrayscaleImage | RGBImage:
    """
    Morph the image by shifting, scaling, tilting, and rotating.
    Fills empty space with black pixels and crops it to the same size as the input image.
    The output image is the same size as the input image.

    Args:
        image: Input grayscale or RGB image (uint8)
        x_shift: Horizontal shift in pixels
        y_shift: Vertical shift in pixels
        scale: Scale factor (1.0 = no scaling)
        tilt_x: Horizontal tilt/shear factor
        tilt_y: Vertical tilt/shear factor
        rotation: Rotation angle in degrees

    Returns:
        Morphed image (same type as input, uint8)
    """

    morphed_image = apply_advanced_transform(
        image.data, rotation, scale, tilt_x, tilt_y, x_shift, y_shift
    )

    result_type = type(image)
    return result_type.from_numpy(morphed_image)


def apply_advanced_transform(
    img, angle=0, scale=1.0, tilt_x=0.0, tilt_y=0.0, shift_x=0, shift_y=0
):
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # --- 1. Generate Affine Matrix (Rotate + Scale) ---
    # getRotationMatrix2D creates a 2x3 matrix
    M_affine = cv2.getRotationMatrix2D(center, angle, scale)

    # We need to convert it to a 3x3 matrix to combine it with perspective
    # Add a row [0, 0, 1] at the bottom
    M_affine_3x3 = np.vstack([M_affine, [0, 0, 1]])

    # --- 2. Generate Perspective Matrix (Tilt) ---
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.copy(src_points)

    # Apply Shift
    dst_points[:, 0] += shift_x
    dst_points[:, 1] += shift_y

    # Apply Tilt (Squeeze corners)
    if tilt_x != 0:
        squeeze = w * abs(tilt_x) * 0.5
        if tilt_x > 0:  # Top back
            dst_points[0, 0] += squeeze
            dst_points[1, 0] -= squeeze
        else:  # Bottom back
            dst_points[2, 0] += squeeze
            dst_points[3, 0] -= squeeze

    if tilt_y != 0:
        squeeze = h * abs(tilt_y) * 0.5
        if tilt_y > 0:  # Right back
            dst_points[1, 1] += squeeze
            dst_points[3, 1] -= squeeze
        else:  # Left back
            dst_points[0, 1] += squeeze
            dst_points[2, 1] -= squeeze

    # Calculate Perspective Matrix
    M_perspective = cv2.getPerspectiveTransform(src_points, dst_points)

    # --- 3. Combine Matrices ---
    # Matrix Multiplication: Perspective @ Affine
    # This applies Affine (Scale/Rot) first, THEN Perspective (Tilt/Shift)
    M_final = np.dot(M_perspective, M_affine_3x3)

    # --- 4. Apply Final Transform ---
    # Note: We use warpPerspective even though we have rotation,
    # because the final matrix is 3x3
    return cv2.warpPerspective(
        img, M_final, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )

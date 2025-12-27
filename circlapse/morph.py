import cv2
import numpy as np

from circlapse.image_types import GrayscaleImage, RGBImage


def morph_image(
    image: GrayscaleImage | RGBImage,
    x_shift: float,
    y_shift: float,
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

    h, w = image.data.shape[:2]
    M = get_transform_matrix(
        image.data.shape, rotation, scale, tilt_x, tilt_y, x_shift, y_shift
    )
    morphed = cv2.warpPerspective(
        image.data, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )

    result_type = type(image)
    return result_type.from_numpy(morphed)


def revert_morph_image(
    image: GrayscaleImage | RGBImage,
    x_shift: float,
    y_shift: float,
    scale: float,
    tilt_x: float,
    tilt_y: float,
    rotation: float,
) -> GrayscaleImage | RGBImage:
    """
    Reverse a morph transformation.
    Pass the SAME parameters that were used in the original morph_image call.
    """
    h, w = image.data.shape[:2]
    M = get_transform_matrix(
        image.data.shape, rotation, scale, tilt_x, tilt_y, x_shift, y_shift
    )
    M_inv = np.linalg.inv(M)
    reverted = cv2.warpPerspective(
        image.data, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderValue=0
    )

    result_type = type(image)
    return result_type.from_numpy(reverted)


def get_transform_matrix(
    img_shape: tuple[int, ...],
    angle: float = 0,
    scale: float = 1.0,
    tilt_x: float = 0.0,
    tilt_y: float = 0.0,
    shift_x: float = 0,
    shift_y: float = 0,
) -> np.ndarray:
    """Calculate the transformation matrix without applying it."""
    h, w = img_shape[:2]
    center = (w / 2, h / 2)

    # Affine Matrix (Rotate + Scale)
    M_affine = cv2.getRotationMatrix2D(center, angle, scale)
    M_affine_3x3 = np.vstack([M_affine, [0, 0, 1]])

    # Perspective Matrix (Tilt + Shift)
    src_points = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    dst_points = np.copy(src_points)
    dst_points[:, 0] += shift_x
    dst_points[:, 1] += shift_y

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

    M_perspective = cv2.getPerspectiveTransform(src_points, dst_points)
    return np.dot(M_perspective, M_affine_3x3)

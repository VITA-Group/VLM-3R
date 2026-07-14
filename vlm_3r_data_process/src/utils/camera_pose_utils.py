"""Lightweight utilities for working with camera poses."""

import numpy as np


def get_camera_position_from_c2w(pose_matrix):
    """Return the camera center in world coordinates from a 4x4 c2w pose.

    A camera-to-world transform maps a camera-space point ``x_c`` to
    ``x_w = R_c2w @ x_c + t_c2w``. The camera center is the camera-space
    origin, so its world-space position is the translation column ``t_c2w``.
    The expression ``-R.T @ t`` instead belongs to the inverse w2c transform.

    Returns ``None`` when the input is not a finite 4x4 matrix.
    """
    try:
        pose = np.asarray(pose_matrix, dtype=np.float64)
    except (TypeError, ValueError):
        return None

    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        return None

    return pose[:3, 3].copy()

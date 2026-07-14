import numpy as np

from src.utils.camera_pose_utils import get_camera_position_from_c2w


def test_camera_position_from_rotated_c2w_is_translation_column():
    pose_c2w = np.array(
        [
            [0.0, -1.0, 0.0, 1.25],
            [1.0, 0.0, 0.0, -2.5],
            [0.0, 0.0, 1.0, 3.75],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    expected_camera_center = pose_c2w[:3, 3]
    old_w2c_formula = -pose_c2w[:3, :3].T @ pose_c2w[:3, 3]

    # Guard against a fixture for which the old, incorrect formula coincides.
    assert not np.allclose(expected_camera_center, old_w2c_formula)
    np.testing.assert_allclose(
        get_camera_position_from_c2w(pose_c2w), expected_camera_center
    )


def test_camera_position_rejects_invalid_poses():
    assert get_camera_position_from_c2w(np.eye(3)) is None

    non_finite_pose = np.eye(4)
    non_finite_pose[0, 3] = np.inf
    assert get_camera_position_from_c2w(non_finite_pose) is None

    assert get_camera_position_from_c2w("not a pose") is None


def test_camera_displacement_is_invariant_to_world_frame_alignment():
    start_c2w = np.array(
        [
            [0.0, -1.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    end_c2w = start_c2w.copy()
    end_c2w[:3, 3] = [2.5, -1.0, 4.0]

    world_alignment = np.array(
        [
            [0.0, -1.0, 0.0, 10.0],
            [1.0, 0.0, 0.0, -7.0],
            [0.0, 0.0, 1.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    displacement = np.linalg.norm(
        get_camera_position_from_c2w(end_c2w)
        - get_camera_position_from_c2w(start_c2w)
    )
    aligned_displacement = np.linalg.norm(
        get_camera_position_from_c2w(world_alignment @ end_c2w)
        - get_camera_position_from_c2w(world_alignment @ start_c2w)
    )

    np.testing.assert_allclose(aligned_displacement, displacement)

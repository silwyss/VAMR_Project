from scipy.spatial.transform import Slerp, Rotation
from src.constants import CALIBRATION_FILE, DATASET_DIR
from src.models.Camera import *
from yaml import safe_load
import numpy as np
import matplotlib.pyplot as plt

def load_frames() -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamps_s [N], filenames [N]) parsed from times.txt."""
    times = np.loadtxt(DATASET_DIR / 'times.txt', comments='#', dtype=str)
    timestamps, filenames = times[:, 1].astype(np.float64), times[:, -1]
    return timestamps, filenames

def load_groundtruth() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (t_s [M], positions [M,3], quats_xyzw [M,4])."""
    gt = np.loadtxt(DATASET_DIR / 'stamped_groundtruth.txt', comments='#', dtype=np.float64)
    timestamps, positions, quats = gt[:, 0], gt[:, 1:4], gt[:, 4:]
    return timestamps, positions, quats

def interpolate_poses(t_gt, pos, quat, t_query) -> tuple[Rotation, np.ndarray, np.ndarray]:
    """GT pose at t_query. Return (R_wc, p_wc, valid_mask)."""
    valid_mask = (t_query >= t_gt[0]) & (t_query <= t_gt[-1])

    t_clipped = np.clip(t_query, t_gt[0], t_gt[-1])
    R_wc = Slerp(t_gt, Rotation.from_quat(quat))(t_clipped)
    
    p_wc = np.column_stack([np.interp(t_clipped, t_gt, pos[:, i]) for i in range(3)])

    return R_wc, p_wc, valid_mask

def correct_poses_to_cam0(R_wc1: Rotation, p_wc1: np.ndarray) -> tuple[Rotation, np.ndarray]:
    """The ground truth is aligned to the event camera (cam1) which is slightly offset to the RGB camera (cam0)
    So we correct the Rotation and position accordingly."""
    T_cn_cnm1 = load_cam0_to_cam1_transformation()
    T_wc1 = np.zeros((p_wc1.shape[0], 4, 4))
    T_wc1[:, :3, :] = np.c_[R_wc1.as_matrix(), p_wc1.reshape(p_wc1.shape[0], 3, 1)]
    T_wc1[:, -1, -1] = 1
    T_wc0 = T_wc1 @ T_cn_cnm1
    R_wc0, p_wc0 = T_wc0[:, :3, :3], T_wc0[:, :3, 3]
    return Rotation.from_matrix(R_wc0), p_wc0

def load_cam0_to_cam1_transformation() -> np.ndarray:
    with open(CALIBRATION_FILE, 'r') as yaml_file:
        calibration_yaml = safe_load(yaml_file)
        cam1_yaml = calibration_yaml['cam1']
        T_cn_cnm1 = cam1_yaml['T_cn_cnm1']
        return np.array(T_cn_cnm1)

def load_calibration() -> list[CameraCalibration]:
    """First = cam0 (RGB camera), Second = cam1 (event camera)"""
    with open(CALIBRATION_FILE, 'r') as yaml_file:
        calibration_yaml = safe_load(yaml_file)
        cameras = parse_cameras(calibration_yaml)
        return cameras

def parse_cameras(calibration_yaml: dict) -> list[CameraCalibration]:
    try:
        cam0_yaml = calibration_yaml['cam0']
        cam0 = parse_camera(cam0_yaml)
        cam1_yaml = calibration_yaml['cam1']
        cam1 = parse_camera(cam1_yaml)
        return [cam0, cam1]
    except Exception as e:
        print(f'Error while parsing camera from calibration yaml: {e}')
        exit(6)

def parse_camera(camera_yaml: dict) -> CameraCalibration:
    intr_ = camera_yaml['intrinsics']
    if len(intr_) != 4:
        print('Invalid Camera intrinsics provided')
        exit(3)
    intrinsics = CameraIntrinsics(intr_[0], intr_[1], intr_[2], intr_[3])
    dist_ = camera_yaml['distortion_coeffs']
    if len(dist_) != 4:
        print('Invalid Camera Distortion provided')
        exit(4)
    distortion = CameraDistortion(dist_[0], dist_[1], dist_[2], dist_[3], camera_yaml['distortion_model'])
    res_ = camera_yaml['resolution']
    if len(res_) != 2:
        print('Invalid Camera Resolution provided')
        exit(5)
    resolution = CameraResolution(res_[0], res_[1])
    camera_model = camera_yaml['camera_model']
    flip = camera_yaml['flip'] if 'flip' in camera_yaml.keys() else False
    camera = CameraCalibration(intrinsics, distortion, resolution, camera_model, flip)
    return camera

def _plot_poses(t_query: np.ndarray,
               interpolated: tuple[Rotation, np.ndarray, np.ndarray],
               groundtruth: tuple[np.ndarray, np.ndarray, np.ndarray],
               show: bool = True) -> plt.Figure:
    """Plot interpolated poses on top of the ground truth trajectory.

    t_query:      [N] query timestamps (s) that were passed to interpolate_poses.
    interpolated: (R_wc, p_wc, valid_mask) as returned by interpolate_poses, with one
                  entry per t_query. Entries where valid_mask is False are skipped.
    groundtruth:  (t_s, positions, quats_xyzw) as returned by load_groundtruth.
    """
    R_wc, p_wc, valid = interpolated
    t_gt, pos_gt, quat_gt = groundtruth
    valid = np.asarray(valid, dtype=bool)

    t0 = t_gt[0]  # plot time relative to the GT start
    t_int = np.asarray(t_query)[valid] - t0
    t_gt = t_gt - t0
    p_int = np.asarray(p_wc)[valid]

    # Orientation as quaternion components (no gimbal lock / angle wrapping like Euler).
    # q and -q are the same rotation, so make the GT sequence sign-continuous and
    # align each interpolated quaternion to the sign of the closest GT quaternion.
    q_gt = Rotation.from_quat(quat_gt).as_quat()
    flips = np.cumsum(np.r_[0, np.einsum('ij,ij->i', q_gt[1:], q_gt[:-1]) < 0]) % 2
    q_gt[flips == 1] *= -1
    q_int = R_wc[valid].as_quat()
    nearest = np.clip(np.searchsorted(t_gt, t_int), 0, len(t_gt) - 1)
    q_int[np.einsum('ij,ij->i', q_int, q_gt[nearest]) < 0] *= -1

    fig = plt.figure(figsize=(14, 7))
    grid = fig.add_gridspec(2, 2, width_ratios=(1, 1.3))
    ax3d = fig.add_subplot(grid[:, 0], projection='3d')
    ax_pos = fig.add_subplot(grid[0, 1])
    ax_rot = fig.add_subplot(grid[1, 1], sharex=ax_pos)

    # 3D trajectory
    ax3d.scatter(*p_int.T, color='tab:orange', s=3, alpha=0.5, label='interpolated')
    ax3d.plot(*pos_gt.T, color='k', lw=0.8, label='ground truth')
    ax3d.scatter(*pos_gt[0], color='tab:green', s=40, marker='o', label='start')
    ax3d.set_xlabel('x [m]')
    ax3d.set_ylabel('y [m]')
    ax3d.set_zlabel('z [m]')
    ax3d.set_title('Trajectory (world frame)')
    ax3d.legend(loc='upper left')
    # equal aspect so the trajectory is not distorted
    span = np.ptp(pos_gt, axis=0).max() / 2
    centre = pos_gt.min(axis=0) + np.ptp(pos_gt, axis=0) / 2
    ax3d.set_xlim(centre[0] - span, centre[0] + span)
    ax3d.set_ylim(centre[1] - span, centre[1] + span)
    ax3d.set_zlim(centre[2] - span, centre[2] + span)

    # Per-component curves over time: GT as lines, interpolation as markers
    for i, (c, n) in enumerate(zip(('tab:red', 'tab:green', 'tab:blue'), 'xyz')):
        ax_pos.plot(t_gt, pos_gt[:, i], color=c, lw=1, alpha=0.6, label=f'{n} GT')
        ax_pos.plot(t_int, p_int[:, i], 'x', color=c, ms=3, label=f'{n} interp')
    for i, (c, n) in enumerate(zip(('tab:red', 'tab:green', 'tab:blue', 'tab:purple'), 'xyzw')):
        ax_rot.plot(t_gt, q_gt[:, i], color=c, lw=1, alpha=0.6, label=f'q{n} GT')
        ax_rot.plot(t_int, q_int[:, i], 'x', color=c, ms=3, label=f'q{n} interp')

    ax_pos.set_ylabel('position [m]')
    ax_pos.set_title('Position')
    ax_pos.legend(ncol=3, fontsize='small')
    ax_pos.grid(alpha=0.3)
    ax_rot.set_ylabel('quaternion (xyzw)')
    ax_rot.set_xlabel(f'time since GT start [s]  (t0 = {t0:.3f})')
    ax_rot.set_title('Orientation')
    ax_rot.legend(ncol=4, fontsize='small')
    ax_rot.grid(alpha=0.3)

    fig.suptitle(f'Interpolated poses vs ground truth ({valid.sum()}/{valid.size} valid)')
    fig.tight_layout()
    if show:
        plt.show()
    return fig

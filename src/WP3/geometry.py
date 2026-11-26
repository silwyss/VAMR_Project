"""Shared camera geometry for WP3: a virtual pinhole camera both modalities are undistorted into, and GT helpers."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from src.constants import DATASET_DIR, IMAGES_DIR
from src.SfM.build_map import load_frame_split
from src.SfM.data import (correct_poses_to_cam0, interpolate_poses, load_calibration, load_cam0_to_cam1_transformation,
                          load_frames, load_groundtruth)
from src.WP3.events import EventWindow, undistort_events


class VirtualCamera:
    """Undistorts cam0 (RGB) images and cam1 (event) coordinates into one distortion-free pinhole camera with cam1's K.
    This also removes the focal-length difference (cam0 767 px vs cam1 561 px), so both inputs have the same scale."""

    def __init__(self):
        cam0, cam1 = load_calibration()
        self.K0, self.D0 = cam0.intrinsics.asArray(), cam0.distortion.asArray()
        self.K1, self.D1 = cam1.intrinsics.asArray(), cam1.distortion.asArray()
        self.size = (cam1.resolution.width, cam1.resolution.heigth)
        self.K = self.K1
        self.map0 = cv2.initUndistortRectifyMap(self.K0, self.D0, None, self.K, self.size, cv2.CV_32FC1)
        valid = cv2.remap(np.full(self.size[::-1], 255, np.uint8), *self.map0, cv2.INTER_NEAREST) > 0
        self.image_valid = cv2.erode(valid.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0   # no kpts on the black border

    def image(self, name: str, flip: bool = False, images_dir=IMAGES_DIR) -> np.ndarray:
        img = cv2.imread(str(images_dir / name), cv2.IMREAD_GRAYSCALE)
        if flip:
            img = cv2.flip(img, 1)
        return cv2.remap(img, *self.map0, cv2.INTER_LINEAR)

    def events(self, ev: EventWindow) -> EventWindow:
        return undistort_events(ev, self.K1, self.D1, self.K, self.size)

    def to_cam0_pixels(self, xy: np.ndarray) -> np.ndarray:
        """Virtual-camera pixels of the RGB image -> original (distorted) cam0 pixels."""
        rays = np.c_[xy, np.ones(len(xy))] @ np.linalg.inv(self.K).T
        px, _ = cv2.projectPoints(rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), self.K0, self.D0)
        return px.reshape(-1, 2)

    def same_time_homography(self) -> np.ndarray:
        """Maps RGB virtual pixels to event virtual pixels at the same instant (pure rotation R_c1c0; the 4.5 mm
        baseline adds only ~f*t/Z = 2.5 px parallax at 1 m)."""
        R_c1c0 = load_cam0_to_cam1_transformation()[:3, :3]
        return self.K @ R_c1c0 @ np.linalg.inv(self.K)


@dataclass
class FrameData:
    """All RGB frames with valid GT: timestamps, names, event-camera (cam1) and RGB-camera (cam0) poses, speed."""
    t: np.ndarray
    names: np.ndarray
    R1: Rotation
    p1: np.ndarray
    R0: Rotation
    p0: np.ndarray
    speed_deg_s: np.ndarray       # angular speed of the camera at each frame
    key_idx: np.ndarray           # indices of map keyframes
    query_idx: np.ndarray         # indices of held-out query frames
    dataset_dir: Path = DATASET_DIR

    @staticmethod
    def load() -> 'FrameData':
        """The map sequence (Dataset/, peanuts_light) with the WP1 keyframe / query split."""
        t, names = load_frames()
        return FrameData._build(t, names, load_groundtruth(), load_frame_split(), DATASET_DIR)

    @staticmethod
    def load_sequence(dataset_dir: Path) -> 'FrameData':
        """Another EDS sequence (e.g. Dataset_Dark/) used only for queries: no keyframes, every frame is a query.
        Same file layout as Dataset/ (times.txt, stamped_groundtruth.txt, images/, events.h5)."""
        times = np.loadtxt(dataset_dir / 'times.txt', comments='#', dtype=str)
        gt = np.loadtxt(dataset_dir / 'stamped_groundtruth.txt', comments='#', dtype=np.float64)
        return FrameData._build(times[:, 1].astype(np.float64), times[:, 4], (gt[:, 0], gt[:, 1:4], gt[:, 4:]),
                                ([], None), dataset_dir)

    @staticmethod
    def _build(t, names, groundtruth, split, dataset_dir) -> 'FrameData':
        R1, p1, valid = interpolate_poses(*groundtruth, t)
        t, names, R1, p1 = t[valid], names[valid], R1[valid], p1[valid]
        R0, p0 = correct_poses_to_cam0(R1, p1)
        speed = np.degrees((R1[:-1].inv() * R1[1:]).magnitude()) / np.diff(t)
        speed = np.r_[speed, speed[-1]]
        idx = {n: i for i, n in enumerate(names)}
        keys, queries = split
        key_idx = np.array([idx[n] for n in keys if n in idx], dtype=int)
        query_idx = np.arange(len(names)) if queries is None else np.array([idx[n] for n in queries if n in idx])
        return FrameData(t, names, R1, p1, R0, p0, speed, key_idx, query_idx, dataset_dir)

    def t_us(self, i: int) -> int:
        return int(round(self.t[i] * 1e6))

    @property
    def images_dir(self) -> Path:
        return self.dataset_dir / 'images'

    def pose_gap(self, i, j: np.ndarray, query: 'FrameData | None' = None) -> tuple[np.ndarray, np.ndarray]:
        """(position distance [m], rotation angle [deg]) between query frame i and frames j of this sequence
        (cam0 GT). `query`: the sequence frame i belongs to (default: this one; another EDS sequence of the same
        scene requires both GTs to be in the same mocap world frame)."""
        q = self if query is None else query
        j = np.atleast_1d(j)
        return (np.linalg.norm(self.p0[j] - q.p0[i], axis=1),
                np.degrees((self.R0[j].inv() * q.R0[i]).magnitude()))

    def nearest_keyframes(self, i: int, k: int, candidates: np.ndarray | None = None,
                          query: 'FrameData | None' = None) -> np.ndarray:
        """Oracle retrieval: the k map keyframes (of `candidates`, default all) closest in pose (5 cm ~ 5 deg)."""
        cand = self.key_idx if candidates is None else candidates
        dp, dr = self.pose_gap(i, cand, query)
        return cand[np.argsort(dp / 0.05 + dr / 5.0)[:k]]

    def nearest_keyframe(self, i: int) -> int:
        return int(self.nearest_keyframes(i, 1)[0])


def pose_error(R_cw: np.ndarray, t_cw: np.ndarray, R_wc_gt: Rotation, p_wc_gt: np.ndarray) -> tuple[float, float]:
    """(position error [m], rotation error [deg]) of an estimated cam_from_world pose against a GT world pose."""
    R_wc = Rotation.from_matrix(R_cw.T)
    C = -R_cw.T @ t_cw
    return float(np.linalg.norm(C - p_wc_gt)), float(np.degrees((R_wc_gt.inv() * R_wc).magnitude()))


def homography_precision(H: np.ndarray, pts_image: np.ndarray, pts_events: np.ndarray, tol_px: float) -> np.ndarray:
    """Bool mask of matches whose image point, mapped by H, lands within tol_px of the event point."""
    if len(pts_image) == 0:
        return np.zeros(0, bool)
    proj = cv2.perspectiveTransform(pts_image.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)
    return np.linalg.norm(proj - pts_events, axis=1) < tol_px

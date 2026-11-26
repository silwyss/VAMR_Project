"""2D-3D lifting against the WP1 map and absolute pose estimation (PnP + LO-RANSAC + refinement).

Lifting ("which 3D point does this matched map-image keypoint see?"): a matched map-image keypoint gets the 3D point
of the nearest 3D-backed keypoint of that map image within a radius. Two maps:
  sift  outputs/SfM/known_poses (WP1): SIFT keypoints in distorted cam0 pixels -> EI-Nexus keypoints only rarely
        coincide (radius 2 px, ~6 % of matches lifted)
  sp    outputs/WP3/sp_map/model (src/WP3/sp_map.py): EI-Nexus' own SuperPoint keypoints in the virtual camera ->
        lifting is an exact lookup (radius 0.5 px)
"""
from dataclasses import dataclass

import numpy as np
import pycolmap
from scipy.spatial import cKDTree

from src.constants import SFM_GT_MAP


class MapIndex:
    """Per map image: KD-tree over the SIFT keypoints that observe a 3D point (cam0 pixel coordinates)."""

    def __init__(self, path=SFM_GT_MAP):
        """COLMAP stores keypoints with the origin at the corner of the top-left pixel (+0.5 w.r.t. OpenCV/ours)."""
        self.rec = pycolmap.Reconstruction(str(path))
        self._by_name = {}
        for im in self.rec.images.values():
            obs = [(p.xy, p.point3D_id) for p in im.points2D if p.has_point3D()]
            xy = np.array([o[0] for o in obs]).reshape(-1, 2) - 0.5
            ids = np.array([o[1] for o in obs], dtype=np.int64)
            self._by_name[im.name] = (cKDTree(xy) if len(xy) else None, ids)

    def lift(self, name: str, xy_cam0: np.ndarray, radius_px: float) -> np.ndarray:
        """3D point id for every keypoint (-1 if no 3D-backed SIFT keypoint within radius)."""
        tree, ids = self._by_name[name]
        out = np.full(len(xy_cam0), -1, np.int64)
        if tree is None or len(xy_cam0) == 0:
            return out
        dist, nn = tree.query(xy_cam0, distance_upper_bound=radius_px)
        ok = np.isfinite(dist)
        out[ok] = ids[nn[ok]]
        return out

    def xyz(self, point_ids: np.ndarray) -> np.ndarray:
        return np.array([self.rec.points3D[int(i)].xyz for i in point_ids]).reshape(-1, 3)


@dataclass
class PoseResult:
    success: bool
    R_cw: np.ndarray | None
    t_cw: np.ndarray | None
    num_correspondences: int
    num_inliers: int


class PnP:
    """Absolute pose of a distortion-free pinhole camera K from 2D-3D correspondences (pycolmap LO-RANSAC + refine)."""

    def __init__(self, K: np.ndarray, size: tuple[int, int], max_error_px: float = 4.0, min_inliers: int = 12):
        self.camera = pycolmap.Camera(model='PINHOLE', width=size[0], height=size[1],
                                      params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
        self.opts = pycolmap.AbsolutePoseEstimationOptions()
        self.opts.ransac.max_error = max_error_px
        self.min_inliers = min_inliers

    def __call__(self, pts2d: np.ndarray, pts3d: np.ndarray) -> PoseResult:
        n = len(pts2d)
        if n < 4:
            return PoseResult(False, None, None, n, 0)
        ret = pycolmap.estimate_and_refine_absolute_pose(pts2d.astype(np.float64), pts3d.astype(np.float64),
                                                         self.camera, self.opts)
        if ret is None:
            return PoseResult(False, None, None, n, 0)
        T = ret['cam_from_world']
        inl = int(np.sum(ret['inlier_mask']))
        return PoseResult(inl >= self.min_inliers, T.rotation.matrix(), np.asarray(T.translation), n, inl)


def collect_correspondences(pts2d_query: np.ndarray, point_ids: np.ndarray, scores: np.ndarray, map_index: MapIndex):
    """Keep lifted matches; if several query points hit the same 3D point keep the highest-scoring one.
    Returns (pts2d [M,2], pts3d [M,3])."""
    ok = point_ids >= 0
    pts2d_query, point_ids, scores = pts2d_query[ok], point_ids[ok], scores[ok]
    order = np.argsort(-scores)
    _, first = np.unique(point_ids[order], return_index=True)
    keep = order[first]
    return pts2d_query[keep], map_index.xyz(point_ids[keep])

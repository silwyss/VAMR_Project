"""Dense reconstruction of the scene (just for fun - not needed for the relocalization).

Two backends:
  sgbm    (default, CPU, works out of the box)
          Pairs of nearby keyframes are stereo-rectified with their known poses, OpenCV semi-global
          matching computes a disparity map per pair, the disparities are back-projected to 3D and all
          pairs are fused into one coloured point cloud. Only voxels seen by >= --min-views pairs are kept.
  colmap  (best quality, GPU)
          COLMAP's PatchMatch multi-view stereo + fusion. The pip build of pycolmap has no CUDA, so this
          needs the COLMAP binary with CUDA: download 'colmap-x64-windows-cuda.zip' (a recent release, it
          must read COLMAP >= 3.12 models with rigs/frames) from https://github.com/colmap/colmap/releases
          and pass bin/colmap.exe with --colmap-exe or the COLMAP_EXE environment variable.

Usage (from the project root):
  python -m src.scripts.dense_map                          # sgbm on the aligned incremental model 0
  python -m src.scripts.dense_map --model known_poses      # use the GT poses instead (noisier)
  python -m src.scripts.dense_map --backend colmap --colmap-exe C:/tools/colmap/bin/colmap.exe
  python -m src.scripts.dense_map --show                   # also open the result in an Open3D viewer

Outputs go to outputs/dense/: dense_<backend>_<model>.ply and a preview PNG.
"""
import argparse
import os
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import pycolmap
from scipy.spatial.transform import Rotation

from src.constants import IMAGES_DIR, OUTPUT_DIR, SFM_OUTPUT_DIR

DENSE_DIR = OUTPUT_DIR / 'dense'


# --------------------------------------------------------------------------------------------------
# SGBM backend
# --------------------------------------------------------------------------------------------------

def _intrinsics(cam: pycolmap.Camera) -> tuple[np.ndarray, np.ndarray]:
    """K and OpenCV distortion (k1, k2, p1, p2) of a COLMAP OPENCV / PINHOLE camera."""
    p = np.asarray(cam.params, dtype=float)
    K = np.array([[p[0], 0, p[2]], [0, p[1], p[3]], [0, 0, 1]])
    D = p[4:8] if cam.model == pycolmap.CameraModelId.OPENCV else np.zeros(4)
    return K, D


def _cam_from_world(im: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    T = im.cam_from_world()
    return T.rotation.matrix(), np.asarray(T.translation, dtype=float)


def _relative(im_i: pycolmap.Image, im_j: pycolmap.Image) -> tuple[np.ndarray, np.ndarray]:
    """R, T with x_j = R x_i + T (the convention of cv2.stereoRectify)."""
    R_i, t_i = _cam_from_world(im_i)
    R_j, t_j = _cam_from_world(im_j)
    R = R_j @ R_i.T
    return R, t_j - R @ t_i


def select_pairs(images: list, baseline: tuple[float, float], max_angle_deg: float, step: int,
                 search: int = 40, target_baseline: float = 0.08) -> list[tuple[int, int]]:
    """For every `step`-th keyframe pick a later keyframe with a mostly sideways baseline in range."""
    pairs = []
    for i in range(0, len(images), step):
        best, best_score = None, np.inf
        for j in range(i + 1, min(i + search, len(images))):
            R, T = _relative(images[i], images[j])
            b = np.linalg.norm(T)
            angle = np.degrees(Rotation.from_matrix(R).magnitude())
            sideways = abs(T[0]) / max(b, 1e-9)
            if not (baseline[0] <= b <= baseline[1]) or angle > max_angle_deg or sideways < 0.7:
                continue
            score = abs(b - target_baseline) + 0.01 * angle
            if score < best_score:
                best, best_score = j, score
        if best is not None:
            pairs.append((i, best))
    return pairs


def _rectify(im_i, im_j, K, D, size):
    R, T = _relative(im_i, im_j)
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(K, D, K, D, size, R, T.reshape(3, 1),
                                                flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
    return R1, R2, P1, P2, Q


def depth_from_pair(im_i: pycolmap.Image, im_j: pycolmap.Image, K: np.ndarray, D: np.ndarray,
                    size: tuple[int, int], depth_range: tuple[float, float], pixel_step: int,
                    ) -> tuple[np.ndarray, np.ndarray]:
    """World points [N, 3] and RGB colours [N, 3] (0..1) from one stereo pair."""
    R1, R2, P1, P2, Q = _rectify(im_i, im_j, K, D, size)
    if abs(P2[1, 3]) > abs(P2[0, 3]):          # vertical rectification - SGBM needs horizontal
        return np.empty((0, 3)), np.empty((0, 3))
    if P2[0, 3] > 0:                           # second camera is on the left -> swap roles
        im_i, im_j = im_j, im_i
        R1, R2, P1, P2, Q = _rectify(im_i, im_j, K, D, size)

    w, h = size
    map_l = cv2.initUndistortRectifyMap(K, D, R1, P1, size, cv2.CV_32FC1)
    map_r = cv2.initUndistortRectifyMap(K, D, R2, P2, size, cv2.CV_32FC1)
    color_l = cv2.remap(cv2.imread(str(IMAGES_DIR / im_i.name)), *map_l, cv2.INTER_LINEAR)
    gray_l = cv2.cvtColor(color_l, cv2.COLOR_BGR2GRAY)
    gray_r = cv2.cvtColor(cv2.remap(cv2.imread(str(IMAGES_DIR / im_j.name)), *map_r, cv2.INTER_LINEAR),
                          cv2.COLOR_BGR2GRAY)
    inside = cv2.remap(np.full((h, w), 255, np.uint8), *map_l, cv2.INTER_NEAREST) > 0

    f, baseline = P1[0, 0], abs(P2[0, 3] / P2[0, 0])
    max_disp = int(np.ceil(f * baseline / depth_range[0] / 16.0)) * 16
    block = 5
    sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=max(16, min(max_disp, 256)), blockSize=block,
                                 P1=8 * block ** 2, P2=32 * block ** 2, disp12MaxDiff=1, uniquenessRatio=10,
                                 speckleWindowSize=100, speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
    disp = sgbm.compute(gray_l, gray_r).astype(np.float32) / 16.0

    pts_rect = cv2.reprojectImageTo3D(disp, Q)
    valid = (disp > 1.0) & inside & np.isfinite(pts_rect).all(axis=2)
    valid &= (pts_rect[..., 2] > depth_range[0]) & (pts_rect[..., 2] < depth_range[1])
    sub = np.zeros_like(valid)
    sub[::pixel_step, ::pixel_step] = True
    valid &= sub

    X_rect = pts_rect[valid].astype(np.float64)
    X_cam = X_rect @ R1                         # rectified -> camera i frame (R1^T X)
    R_i, t_i = _cam_from_world(im_i)
    X_world = (X_cam - t_i) @ R_i               # R_i^T (X - t_i)
    colors = color_l[valid][:, ::-1] / 255.0    # BGR -> RGB
    return X_world, colors


def fuse(clouds: list[tuple[np.ndarray, np.ndarray]], voxel: float, min_views: int) -> o3d.geometry.PointCloud:
    """Keep points whose voxel is observed by >= min_views pairs, then downsample and remove outliers."""
    clouds = [(p, c) for p, c in clouds if len(p)]
    pts, cols = np.vstack([p for p, _ in clouds]), np.vstack([c for _, c in clouds])
    pair_id = np.concatenate([np.full(len(p), k) for k, (p, _) in enumerate(clouds)])

    # one int64 id per voxel
    grid = np.floor(pts / voxel).astype(np.int64)
    grid -= grid.min(axis=0)
    dims = grid.max(axis=0) + 1
    voxel_id = (grid[:, 0] * dims[1] + grid[:, 1]) * dims[2] + grid[:, 2]

    # number of distinct pairs that observe each voxel
    unique_obs = np.unique(np.stack([voxel_id, pair_id], axis=1), axis=0)
    ids, views = np.unique(unique_obs[:, 0], return_counts=True)
    keep = np.isin(voxel_id, ids[views >= min_views])

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts[keep]))
    pcd.colors = o3d.utility.Vector3dVector(cols[keep])
    pcd = pcd.voxel_down_sample(voxel)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def run_sgbm(model_dir: Path, args) -> o3d.geometry.PointCloud:
    rec = pycolmap.Reconstruction(model_dir)
    images = sorted(rec.images.values(), key=lambda im: im.name)
    cam = rec.cameras[images[0].camera_id]
    K, D = _intrinsics(cam)
    size = (cam.width, cam.height)

    pairs = select_pairs(images, tuple(args.baseline), args.max_angle, args.step)
    if args.max_pairs:
        pairs = pairs[:: max(1, len(pairs) // args.max_pairs)][: args.max_pairs]
    print(f'{len(images)} images, {len(pairs)} stereo pairs')

    clouds = []
    for k, (i, j) in enumerate(pairs):
        clouds.append(depth_from_pair(images[i], images[j], K, D, size, tuple(args.depth_range), args.pixel_step))
        if (k + 1) % 20 == 0 or k + 1 == len(pairs):
            print(f'  {k + 1}/{len(pairs)} pairs, {sum(len(p) for p, _ in clouds)} raw points')
    pcd = fuse(clouds, args.voxel, args.min_views)
    print(f'fused: {len(pcd.points)} points')
    return pcd


# --------------------------------------------------------------------------------------------------
# COLMAP (CUDA PatchMatch) backend
# --------------------------------------------------------------------------------------------------

def run_colmap(model_dir: Path, args) -> o3d.geometry.PointCloud:
    exe = args.colmap_exe or os.environ.get('COLMAP_EXE')
    if not exe or not Path(exe).is_file():
        raise FileNotFoundError('COLMAP binary not found - pass --colmap-exe or set COLMAP_EXE (see module docstring).')
    workspace = DENSE_DIR / f'colmap_ws_{model_dir.name}'
    workspace.mkdir(parents=True, exist_ok=True)

    print('undistorting images ...')
    pycolmap.undistort_images(workspace, model_dir, IMAGES_DIR)
    print('PatchMatch stereo (GPU) ...')
    subprocess.run([exe, 'patch_match_stereo', '--workspace_path', str(workspace), '--workspace_format', 'COLMAP',
                    '--PatchMatchStereo.geom_consistency', 'true'], check=True)
    print('fusion ...')
    fused = workspace / 'fused.ply'
    subprocess.run([exe, 'stereo_fusion', '--workspace_path', str(workspace), '--workspace_format', 'COLMAP',
                    '--input_type', 'geometric', '--output_path', str(fused)], check=True)
    return o3d.io.read_point_cloud(str(fused))


# --------------------------------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------------------------------

def save_preview(pcd: o3d.geometry.PointCloud, path: Path, crop_radius_m: float = 0.8, max_points: int = 200_000):
    xyz, rgb = np.asarray(pcd.points), np.asarray(pcd.colors)
    centre = np.median(xyz, axis=0)
    keep = np.all(np.abs(xyz - centre) < crop_radius_m, axis=1)
    xyz, rgb = xyz[keep], np.clip(rgb[keep] * 1.4, 0, 1)
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]

    fig = plt.figure(figsize=(16, 7.5))
    for k, (label, elev, azim) in enumerate((('perspective', 35, -60), ('top view', 90, -90))):
        ax = fig.add_subplot(1, 2, k + 1, projection='3d')
        ax.scatter(*xyz.T, c=rgb, s=0.6, depthshade=False, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(np.ptp(xyz, axis=0))
        ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title=label)
        if elev == 90:                          # z points at the viewer in the top view
            ax.set_zticks([])
            ax.set_zlabel('')
    fig.suptitle(f'Dense reconstruction ({len(pcd.points)} points, {keep.sum()} within {crop_radius_m:g} m of the centre)')
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--backend', choices=('sgbm', 'colmap'), default='sgbm')
    parser.add_argument('--model', default='aligned/0', help='reconstruction folder relative to outputs/SfM')
    parser.add_argument('--colmap-exe', default=None)
    parser.add_argument('--step', type=int, default=2, help='sgbm: use every n-th keyframe as left image')
    parser.add_argument('--max-pairs', type=int, default=0, help='sgbm: limit the number of pairs (0 = all)')
    parser.add_argument('--baseline', type=float, nargs=2, default=(0.04, 0.20), help='sgbm: baseline range [m]')
    parser.add_argument('--max-angle', type=float, default=15.0, help='sgbm: max rotation within a pair [deg]')
    parser.add_argument('--depth-range', type=float, nargs=2, default=(0.3, 4.0), help='sgbm: kept depths [m]')
    parser.add_argument('--pixel-step', type=int, default=2, help='sgbm: back-project every n-th pixel')
    parser.add_argument('--voxel', type=float, default=0.005, help='sgbm: fusion voxel size [m]')
    parser.add_argument('--min-views', type=int, default=2, help='sgbm: min. pairs that must see a voxel')
    parser.add_argument('--show', action='store_true', default=True, help='open the result in an Open3D viewer')
    args = parser.parse_args()

    model_dir = SFM_OUTPUT_DIR / args.model
    if not model_dir.is_dir():
        raise FileNotFoundError(f'{model_dir} not found - run build_map first.')
    DENSE_DIR.mkdir(parents=True, exist_ok=True)

    pcd = run_sgbm(model_dir, args) if args.backend == 'sgbm' else run_colmap(model_dir, args)
    name = f"dense_{args.backend}_{args.model.replace('/', '_')}"
    o3d.io.write_point_cloud(str(DENSE_DIR / f'{name}.ply'), pcd)
    save_preview(pcd, DENSE_DIR / f'{name}.png')
    print(f'wrote {DENSE_DIR / name}.ply and .png')
    if args.show:
        o3d.visualization.draw_geometries([pcd], window_name=name)


if __name__ == '__main__':
    main()

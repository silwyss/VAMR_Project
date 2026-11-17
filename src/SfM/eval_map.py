"""Evaluation of the SfM maps built by build_map.py.

Reads the existing outputs (it never rebuilds the map) and writes
    OUTPUT_DIR / metrics.json   all numbers
    OUTPUT_DIR / tables.md      report tables (Markdown)
    FIGURES_DIR / *.png         report figures
    SFM_OUTPUT_DIR / *.ply      point clouds for interactive inspection (Open3D, MeshLab)

The pose-error helpers (pose_errors, pose_error_summary) are written to be reused
for the hloc frame localization and the event relocalization later.
"""
import json
from dataclasses import dataclass

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d
import pycolmap
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from src.constants import (CALIBRATION_FILE, EVAL_FAST_MOTION_DEG_S, EVAL_MAP_CROP_RADIUS_M, EVAL_RECALL_THRESHOLDS, EVAL_RPE_DELTA,
                           FIGURES_DIR, IMAGES_DIR, METRICS_FILE, SFM_GT_MAP, SFM_MAP_IMAGE_COUNT_THRESHOLD,
                           SFM_MAP_POINT_COUNT_THRESHOLD, SFM_OUTPUT_DIR, SFM_ROTATION_THRESHOLD_DEG,
                           SFM_SEQUENTIAL_OVERLAP, SFM_TRANSLATION_THRESHOLD_M, TABLES_FILE)
from src.SfM.build_map import align_to_groundtruth, db_path, incremental_model_dirs, load_frame_split
from src.SfM.data import (correct_poses_to_cam0, interpolate_poses, load_cam0_to_cam1_transformation,
                          load_frames, load_groundtruth)


# --------------------------------------------------------------------------------------------------
# Data containers
# --------------------------------------------------------------------------------------------------

@dataclass
class FrameGroundTruth:
    """Ground truth of the RGB camera (cam0) at every frame with a valid GT pose."""
    names: np.ndarray       # [N] image file names
    t: np.ndarray           # [N] timestamps [s]
    R_wc: Rotation          # [N] camera-to-world rotations
    p_wc: np.ndarray        # [N, 3] camera centres in the world frame [m]
    ang_speed: np.ndarray   # [N] angular speed [deg/s]
    name_to_idx: dict

    def idx(self, names) -> np.ndarray:
        return np.array([self.name_to_idx[n] for n in names], dtype=int)


@dataclass
class ModelResult:
    """One incremental COLMAP model, aligned to the GT if possible."""
    name: str
    rec: pycolmap.Reconstruction
    aligned: bool
    status: str
    metrics: dict
    reproj: dict            # image name -> per-observation reprojection errors [px]
    names: list             # registered image names, sorted by time
    pos_err: np.ndarray     # [m], same order as names (empty if not aligned)
    rot_err: np.ndarray     # [deg]


def load_frame_groundtruth() -> FrameGroundTruth:
    timestamps, filenames = load_frames()
    R_wc, p_wc, valid = interpolate_poses(*load_groundtruth(), timestamps)
    R_wc, p_wc = correct_poses_to_cam0(R_wc[valid], p_wc[valid])
    t, names = timestamps[valid], filenames[valid]
    step = np.degrees((R_wc[:-1].inv() * R_wc[1:]).magnitude()) / np.diff(t)
    ang_speed = np.r_[step, step[-1]]
    return FrameGroundTruth(names, t, R_wc, p_wc, ang_speed, {n: i for i, n in enumerate(names)})


def load_keyframe_keypoints() -> dict:
    """Number of SIFT keypoints of every keyframe in the COLMAP database."""
    db = pycolmap.Database.open(db_path)
    counts = {im.name: db.num_keypoints_for_image(im.image_id) for im in db.read_all_images()}
    db.close()
    return counts


# --------------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------------

def image_poses(rec: pycolmap.Reconstruction, gt: FrameGroundTruth) -> tuple[list, Rotation, np.ndarray]:
    """(names, R_wc, camera centres) of all registered images that have GT, sorted by time."""
    images = [im for im in rec.images.values() if im.name in gt.name_to_idx]
    images.sort(key=lambda im: gt.name_to_idx[im.name])
    names = [im.name for im in images]
    R_wc = Rotation.from_matrix(np.stack([im.cam_from_world().rotation.matrix() for im in images])).inv()
    centers = np.stack([im.projection_center() for im in images])
    return names, R_wc, centers


def per_image_reprojection_errors(rec: pycolmap.Reconstruction) -> dict:
    """Image name -> reprojection error [px] of every observation of a triangulated point."""
    errors = {}
    for im in rec.images.values():
        obs = [(p2d.xy, p2d.point3D_id) for p2d in im.points2D if p2d.has_point3D()]
        if not obs:
            errors[im.name] = np.empty(0)
            continue
        xy = np.array([o[0] for o in obs])
        xyz = np.array([rec.points3D[o[1]].xyz for o in obs])
        uv = rec.cameras[im.camera_id].img_from_cam(im.cam_from_world() * xyz)
        errors[im.name] = np.linalg.norm(uv - xy, axis=1)
    return errors


def reconstruction_stats(rec: pycolmap.Reconstruction, reproj: dict, gt: FrameGroundTruth) -> dict:
    errs = np.concatenate([e for e in reproj.values() if e.size]) if reproj else np.empty(0)
    idx = np.sort(gt.idx([n for n in reproj if n in gt.name_to_idx]))
    return dict(
        num_images=rec.num_reg_images(),
        num_points3D=rec.num_points3D(),
        mean_track_length=rec.compute_mean_track_length(),
        mean_obs_per_image=rec.compute_mean_observations_per_reg_image(),
        mean_reproj_error_px=rec.compute_mean_reprojection_error(),   # COLMAP definition (mean over points)
        median_obs_reproj_error_px=float(np.median(errs)) if errs.size else None,
        p95_obs_reproj_error_px=float(np.percentile(errs, 95)) if errs.size else None,
        first_frame=str(gt.names[idx[0]]) if idx.size else None,
        last_frame=str(gt.names[idx[-1]]) if idx.size else None,
        time_span_s=float(gt.t[idx[-1]] - gt.t[idx[0]]) if idx.size else 0.0,
    )


def pose_errors(names: list, R_wc_est: Rotation, p_est: np.ndarray, gt: FrameGroundTruth) -> tuple[np.ndarray, np.ndarray]:
    """Absolute position error [m] and rotation error [deg] of estimated camera poses against the GT."""
    idx = gt.idx(names)
    pos_err = np.linalg.norm(p_est - gt.p_wc[idx], axis=1)
    rot_err = np.degrees((gt.R_wc[idx].inv() * R_wc_est).magnitude())
    return pos_err, rot_err


def pose_error_summary(pos_err: np.ndarray, rot_err: np.ndarray, thresholds=EVAL_RECALL_THRESHOLDS) -> dict:
    """ATE, mean/median/max errors and recall at (metres, degrees) thresholds."""
    if pos_err.size == 0:
        return dict(n=0)
    return dict(
        n=int(pos_err.size),
        ate_rmse_m=float(np.sqrt(np.mean(pos_err ** 2))),
        pos_mean_m=float(pos_err.mean()), pos_median_m=float(np.median(pos_err)), pos_max_m=float(pos_err.max()),
        rot_mean_deg=float(rot_err.mean()), rot_median_deg=float(np.median(rot_err)), rot_max_deg=float(rot_err.max()),
        recall={_recall_key(m, d): float(np.mean((pos_err < m) & (rot_err < d))) for m, d in thresholds},
    )


def relative_pose_errors(names: list, R_wc_est: Rotation, p_est: np.ndarray, gt: FrameGroundTruth,
                         delta: int = EVAL_RPE_DELTA) -> tuple[np.ndarray, np.ndarray]:
    """Relative pose error between keyframes `delta` apart (drift; independent of the global alignment)."""
    if len(names) <= delta:
        return np.empty(0), np.empty(0)
    idx = gt.idx(names)
    a, b = np.arange(len(names) - delta), np.arange(delta, len(names))
    rel_R_est = R_wc_est[a].inv() * R_wc_est[b]
    rel_R_gt = gt.R_wc[idx[a]].inv() * gt.R_wc[idx[b]]
    rot = np.degrees((rel_R_est.inv() * rel_R_gt).magnitude())
    rel_t_est = R_wc_est[a].inv().apply(p_est[b] - p_est[a])
    rel_t_gt = gt.R_wc[idx[a]].inv().apply(gt.p_wc[idx[b]] - gt.p_wc[idx[a]])
    return rot, np.linalg.norm(rel_t_est - rel_t_gt, axis=1)


def evaluate_incremental_models(gt: FrameGroundTruth, num_keyframes: int) -> list[ModelResult]:
    results = []
    for model_dir in incremental_model_dirs():
        rec = pycolmap.Reconstruction(model_dir)
        reproj = per_image_reprojection_errors(rec)   # invariant to the Sim(3) alignment
        metrics = dict(model=model_dir.name, **reconstruction_stats(rec, reproj, gt))
        metrics['registered_fraction'] = rec.num_reg_images() / num_keyframes

        if rec.num_reg_images() < SFM_MAP_IMAGE_COUNT_THRESHOLD or rec.num_points3D() < SFM_MAP_POINT_COUNT_THRESHOLD:
            status = 'discarded (too small)'
            sim3 = None
        else:
            sim3, reason = align_to_groundtruth(rec, gt.name_to_idx, gt.R_wc, gt.p_wc)
            status = 'ok' if sim3 is not None else f'discarded ({reason})'
        metrics['status'] = status

        names, R_est, p_est = image_poses(rec, gt)
        pos_err = rot_err = np.empty(0)
        if sim3 is not None:
            metrics['sim3_scale'] = float(np.asarray(sim3.scale))
            pos_err, rot_err = pose_errors(names, R_est, p_est, gt)
            metrics['absolute'] = pose_error_summary(pos_err, rot_err)
            rpe_rot, rpe_trans = relative_pose_errors(names, R_est, p_est, gt)
            metrics['rpe'] = dict(delta_keyframes=EVAL_RPE_DELTA,
                                  rot_mean_deg=float(rpe_rot.mean()) if rpe_rot.size else None,
                                  trans_mean_m=float(rpe_trans.mean()) if rpe_trans.size else None)
        results.append(ModelResult(model_dir.name, rec, sim3 is not None, status, metrics, reproj,
                                   names, pos_err, rot_err))
    return results


def evaluate_known_pose_map(gt: FrameGroundTruth) -> tuple[pycolmap.Reconstruction, dict, dict]:
    rec = pycolmap.Reconstruction(SFM_GT_MAP)
    reproj = per_image_reprojection_errors(rec)
    metrics = reconstruction_stats(rec, reproj, gt)

    names = [n for n in reproj if n in gt.name_to_idx]
    fast = gt.ang_speed[gt.idx(names)] >= EVAL_FAST_MOTION_DEG_S
    for label, mask in (('slow', ~fast), ('fast', fast)):
        sel = [n for n, m in zip(names, mask) if m]
        errs = np.concatenate([reproj[n] for n in sel]) if sel else np.empty(0)
        metrics[f'{label}_motion'] = dict(
            num_images=len(sel),
            mean_obs_per_image=float(np.mean([reproj[n].size for n in sel])) if sel else 0.0,
            mean_obs_reproj_error_px=float(errs.mean()) if errs.size else None,
            median_obs_reproj_error_px=float(np.median(errs)) if errs.size else None,
        )
    metrics['fast_motion_threshold_deg_s'] = EVAL_FAST_MOTION_DEG_S
    return rec, reproj, metrics


def compare_point_clouds(rec: pycolmap.Reconstruction, reference: pycolmap.Reconstruction) -> dict:
    """Distance from every point of `rec` to its nearest neighbour in `reference` (both in the GT frame)."""
    ref = np.array([p.xyz for p in reference.points3D.values()])
    query = np.array([p.xyz for p in rec.points3D.values()])
    dist, _ = cKDTree(ref).query(query)
    return dict(num_points=int(query.shape[0]), median_nn_dist_m=float(np.median(dist)),
                mean_nn_dist_m=float(dist.mean()), p90_nn_dist_m=float(np.percentile(dist, 90)))


def groundtruth_consistency(rec: pycolmap.Reconstruction, gaps=(1, 3, 10, 30),
                            offsets_ms=np.arange(-20.0, 20.01, 2.5)) -> dict:
    """Checks the GT against the relative rotations of an incremental SfM model (independent of the GT):
    - is the GT the pose of the RGB camera (uncorrected) or of the event camera (corrected)?
    - constant rotation offset fitted between GT and SfM, compared with R_cam1_cam0 from the calibration
    - time offset between camera and mocap clock."""
    t_gt, _, quat = load_groundtruth()
    slerp = Slerp(t_gt, Rotation.from_quat(quat))
    R_10 = Rotation.from_matrix(load_cam0_to_cam1_transformation()[:3, :3])
    timestamps, filenames = load_frames()
    name_to_t = dict(zip(filenames, timestamps))

    images = sorted((im for im in rec.images.values() if im.name in name_to_t), key=lambda im: name_to_t[im.name])
    t = np.array([name_to_t[im.name] for im in images])
    R_sfm = Rotation.from_matrix(np.stack([im.cam_from_world().rotation.matrix() for im in images])).inv()

    def gt_rot(dt=0.0):
        return slerp(np.clip(t + dt, t_gt[0], t_gt[-1]))

    def rel_err(R_gt, gap, X=Rotation.identity()):
        a, b = np.arange(len(t) - gap), np.arange(gap, len(t))
        rel_sfm = R_sfm[a].inv() * R_sfm[b]
        rel_gt = X.inv() * R_gt[a].inv() * R_gt[b] * X
        return np.degrees((rel_sfm.inv() * rel_gt).magnitude())

    R_cam1 = gt_rot()
    by_gap = {str(g): dict(uncorrected_deg=float(rel_err(R_cam1, g).mean()),
                           corrected_deg=float(rel_err(R_cam1, g, R_10).mean())) for g in gaps}

    fit = least_squares(lambda v: rel_err(R_cam1, 10, Rotation.from_rotvec(v)), np.zeros(3))
    X = Rotation.from_rotvec(fit.x)

    sweep = np.array([rel_err(gt_rot(dt / 1000.0), 3, R_10).mean() for dt in offsets_ms])
    return dict(
        model_images=len(images),
        rel_rot_error_by_gap=by_gap,
        fitted_offset_deg=float(np.degrees(X.magnitude())),
        R_cam1_cam0_deg=float(np.degrees(R_10.magnitude())),
        fitted_vs_R_cam1_cam0_deg=float(np.degrees((X.inv() * R_10).magnitude())),
        fitted_vs_R_cam0_cam1_deg=float(np.degrees((X.inv() * R_10.inv()).magnitude())),
        time_offset_sweep=dict(offsets_ms=offsets_ms.tolist(), rel_rot_error_deg=sweep.tolist()),
        best_time_offset_ms=float(offsets_ms[np.argmin(sweep)]),
    )


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------

def _model_colors(models: list[ModelResult]) -> dict:
    cmap = plt.get_cmap('tab10')
    return {m.name: cmap(i % 10) for i, m in enumerate(models)}


def _points(rec: pycolmap.Reconstruction) -> np.ndarray:
    return np.array([p.xyz for p in rec.points3D.values()])


def plot_overview(gt: FrameGroundTruth, known_rec: pycolmap.Reconstruction, models: list[ModelResult]) -> plt.Figure:
    pts = _points(known_rec)
    lo, hi = np.percentile(pts, [2, 98], axis=0)       # cut far outliers for a readable view
    pts = pts[np.all((pts >= lo) & (pts <= hi), axis=1)]
    rng = np.random.default_rng(0)
    pts = pts[rng.choice(len(pts), size=min(len(pts), 20000), replace=False)]
    colors = _model_colors(models)

    fig = plt.figure(figsize=(16, 7))
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax2d = fig.add_subplot(1, 2, 2)
    for ax, dims in ((ax3d, (0, 1, 2)), (ax2d, (0, 1))):
        ax.scatter(*pts[:, dims].T, s=0.3, c='0.6', alpha=0.4, label='known-pose map points')
        ax.plot(*gt.p_wc[:, dims].T, color='k', lw=1, label='GT trajectory (RGB camera)')
        for m in models:
            if m.aligned:
                c = np.stack([m.rec.images[i].projection_center() for i in _ids_by_name(m.rec, m.names)])
                ax.plot(*c[:, dims].T, '.-', ms=2, lw=0.8, color=colors[m.name], label=f'SfM model {m.name} (aligned)')
    ax3d.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title='Map and trajectories (GT world frame)')
    both = np.vstack([pts, gt.p_wc])
    ax3d.set_box_aspect(np.ptp(both, axis=0))
    ax2d.set(xlabel='x [m]', ylabel='y [m]', title='Top view (x-y)')
    ax2d.set_aspect('equal')
    ax2d.grid(alpha=0.3)
    ax2d.legend(fontsize='small', markerscale=4, loc='best')
    fig.tight_layout()
    return fig


def _points_and_colors(rec: pycolmap.Reconstruction, brighten: float = 1.6) -> tuple[np.ndarray, np.ndarray]:
    """xyz [N, 3] and RGB colours in [0, 1] (brightened, the sequence is dark)."""
    pts = list(rec.points3D.values())
    xyz = np.array([p.xyz for p in pts])
    rgb = np.clip(np.array([p.color for p in pts]) / 255.0 * brighten, 0, 1)
    return xyz, rgb


def _camera_frustums(rec: pycolmap.Reconstruction, every: int = 10, depth: float = 0.15) -> np.ndarray:
    """Line vertices (NaN rows separate the polylines) drawing a small pyramid for every `every`-th camera."""
    lines = []
    images = sorted(rec.images.values(), key=lambda im: im.name)
    for im in images[::every]:
        cam = rec.cameras[im.camera_id]
        fx, fy, cx, cy = cam.params[:4]
        corners = np.array([[0, 0], [cam.width, 0], [cam.width, cam.height], [0, cam.height]], dtype=float)
        pts_cam = np.c_[(corners[:, 0] - cx) / fx, (corners[:, 1] - cy) / fy, np.ones(4)] * depth
        R_wc = im.cam_from_world().rotation.matrix().T
        C = im.projection_center()
        c = pts_cam @ R_wc.T + C
        nan = np.full((1, 3), np.nan)
        lines += [np.vstack([C, c[i], nan]) for i in range(4)]          # apex -> corners
        lines.append(np.vstack([c, c[:1], nan]))                         # image rectangle
    return np.vstack(lines) if lines else np.empty((0, 3))


def plot_map(rec: pycolmap.Reconstruction, gt: FrameGroundTruth, title: str,
             crop_radius_m: float = EVAL_MAP_CROP_RADIUS_M) -> plt.Figure:
    """The reconstructed map in its real colours, zoomed onto the scene centre, from three viewpoints."""
    xyz, rgb = _points_and_colors(rec)
    centre = np.median(xyz, axis=0)
    keep = np.linalg.norm(xyz[:, :2] - centre[:2], axis=1) < crop_radius_m
    keep &= np.abs(xyz[:, 2] - centre[2]) < crop_radius_m
    xyz, rgb = xyz[keep], rgb[keep]

    fig = plt.figure(figsize=(18, 6.5))
    # (title, elevation, azimuth, axis that points at the viewer -> hidden)
    views = (('perspective', 30, -60, None), ('top view', 90, -90, 'z'), ('side view', 0, -90, 'y'))
    for k, (label, elev, azim, hidden) in enumerate(views):
        ax = fig.add_subplot(1, 3, k + 1, projection='3d')
        ax.scatter(*xyz.T, c=rgb, s=3.0, depthshade=False, linewidths=0)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect(np.ptp(xyz, axis=0))
        ax.set(xlabel='x [m]', ylabel='y [m]', zlabel='z [m]', title=label)
        if hidden:
            getattr(ax, f'set_{hidden}ticks')([])
            getattr(ax, f'set_{hidden}label')('')
    fig.suptitle(f'{title}: {keep.sum()} points within {crop_radius_m:g} m of the scene centre')
    fig.tight_layout()
    return fig


def export_interactive_map(rec: pycolmap.Reconstruction, gt: FrameGroundTruth, models: list[ModelResult],
                           path, max_points: int = 80000):
    """Rotatable/zoomable HTML view (plotly) of the map, the cameras and the trajectories."""
    import plotly.graph_objects as go

    xyz, rgb = _points_and_colors(rec)
    lo, hi = np.percentile(xyz, [1, 99], axis=0)
    keep = np.all((xyz >= lo) & (xyz <= hi), axis=1)
    xyz, rgb = xyz[keep], rgb[keep]
    if len(xyz) > max_points:
        sel = np.random.default_rng(0).choice(len(xyz), max_points, replace=False)
        xyz, rgb = xyz[sel], rgb[sel]
    colors = [f'rgb({r},{g},{b})' for r, g, b in (rgb * 255).astype(int)]
    frustums = _camera_frustums(rec)

    traces = [
        go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='markers', name='map points',
                     marker=dict(size=1.5, color=colors)),
        go.Scatter3d(x=gt.p_wc[:, 0], y=gt.p_wc[:, 1], z=gt.p_wc[:, 2], mode='lines', name='GT trajectory',
                     line=dict(color='black', width=3)),
        go.Scatter3d(x=frustums[:, 0], y=frustums[:, 1], z=frustums[:, 2], mode='lines', name='keyframe cameras',
                     line=dict(color='red', width=2)),
    ]
    model_colors = _model_colors(models)
    for m in models:
        if m.aligned:
            c = np.stack([m.rec.images[i].projection_center() for i in _ids_by_name(m.rec, m.names)])
            r, g, b, _ = (np.array(model_colors[m.name]) * 255).astype(int)
            traces.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode='lines', name=f'SfM model {m.name}',
                                       line=dict(color=f'rgb({r},{g},{b})', width=3), visible='legendonly'))
    fig = go.Figure(traces)
    fig.update_layout(scene=dict(aspectmode='data', xaxis_title='x [m]', yaxis_title='y [m]', zaxis_title='z [m]'),
                      title='Known-pose map (click legend entries to toggle)', margin=dict(l=0, r=0, t=40, b=0))
    fig.write_html(path, include_plotlyjs='cdn')


def _ids_by_name(rec: pycolmap.Reconstruction, names: list) -> list:
    name_to_id = {im.name: im_id for im_id, im in rec.images.items()}
    return [name_to_id[n] for n in names]


def plot_timeline(gt: FrameGroundTruth, keypoints: dict, models: list[ModelResult]) -> plt.Figure:
    t0 = gt.t[0]
    colors = _model_colors(models)
    kf_names = sorted((n for n in keypoints if n in gt.name_to_idx), key=gt.name_to_idx.get)
    kf_t = gt.t[gt.idx(kf_names)] - t0

    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(13, 8), gridspec_kw=dict(height_ratios=(2, 2, 1.6)))
    axes[0].plot(gt.t - t0, gt.ang_speed, color='tab:blue', lw=0.8)
    axes[0].axhline(EVAL_FAST_MOTION_DEG_S, color='tab:red', ls='--', lw=0.8, label='fast-motion threshold')
    axes[0].set(ylabel='angular speed [deg/s]', title='Camera motion, image features and SfM registration')
    axes[0].legend(fontsize='small')

    axes[1].scatter(kf_t, [keypoints[n] for n in kf_names], s=5, color='tab:green')
    axes[1].set(ylabel='SIFT keypoints / keyframe')

    registered = set()
    labels = []
    for row, m in enumerate(models):
        registered |= set(m.names)
        t_m = gt.t[gt.idx(m.names)] - t0 if m.names else np.empty(0)
        axes[2].scatter(t_m, np.full(t_m.shape, row), marker='|', s=60, color=colors[m.name])
        labels.append(f'model {m.name}' + ('' if m.aligned else ' (discarded)'))
    missing = [n for n in kf_names if n not in registered]
    axes[2].scatter(gt.t[gt.idx(missing)] - t0, np.full(len(missing), len(models)), marker='|', s=60, color='tab:red')
    labels.append('unregistered')
    axes[2].set_yticks(range(len(labels)), labels)
    axes[2].set_ylim(-0.7, len(labels) - 0.3)
    axes[2].invert_yaxis()
    axes[2].set(xlabel='time [s]')
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_pose_errors_over_time(gt: FrameGroundTruth, models: list[ModelResult]) -> plt.Figure:
    colors = _model_colors(models)
    fig, (ax_p, ax_r) = plt.subplots(2, 1, sharex=True, figsize=(12, 6))
    for m in models:
        if not m.aligned:
            continue
        t = gt.t[gt.idx(m.names)] - gt.t[0]
        ax_p.plot(t, m.pos_err * 100, '.-', ms=2, lw=0.8, color=colors[m.name], label=f'model {m.name}')
        ax_r.plot(t, m.rot_err, '.-', ms=2, lw=0.8, color=colors[m.name])
    ax_p.set(ylabel='position error [cm]', title='Pose error of the aligned SfM models')
    ax_r.set(ylabel='rotation error [deg]', xlabel='time [s]')
    ax_p.legend(fontsize='small')
    for ax in (ax_p, ax_r):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_reprojection_histogram(known_reproj: dict, models: list[ModelResult]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 4, 81)   # COLMAP drops observations above 4 px by default
    series = [('known-pose map (GT poses fixed)', known_reproj, 'tab:orange')]
    aligned = [m for m in models if m.aligned]
    if aligned:
        largest = max(aligned, key=lambda m: m.rec.num_points3D())
        series.append((f'incremental SfM model {largest.name}', largest.reproj, 'tab:blue'))
    for label, reproj, color in series:
        errs = np.concatenate([e for e in reproj.values() if e.size])
        ax.hist(errs, bins=bins, density=True, histtype='step', lw=1.5, color=color,
                label=f'{label}: median {np.median(errs):.2f} px')
        ax.axvline(np.median(errs), color=color, ls='--', lw=0.8)
    ax.set(xlabel='reprojection error per observation [px]', ylabel='density', title='Reprojection error')
    ax.legend(fontsize='small')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_example_frames(gt: FrameGroundTruth, keypoints: dict) -> plt.Figure:
    names = [n for n in keypoints if n in gt.name_to_idx]
    sharp = max(names, key=keypoints.get)
    blurred = min(names, key=keypoints.get)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, name, label in ((axes[0], sharp, 'most keypoints'), (axes[1], blurred, 'fewest keypoints')):
        img = cv2.cvtColor(cv2.imread(str(IMAGES_DIR / name)), cv2.COLOR_BGR2RGB)
        ax.imshow(img)
        ax.set_title(f'{name} ({label})\n{keypoints[name]} SIFT keypoints, '
                     f'{gt.ang_speed[gt.name_to_idx[name]]:.0f} deg/s', fontsize='small')
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_groundtruth_consistency(consistency: dict) -> plt.Figure:
    fig, (ax_g, ax_t) = plt.subplots(1, 2, figsize=(12, 4.5))
    gaps = list(consistency['rel_rot_error_by_gap'])
    x = np.arange(len(gaps))
    unc = [consistency['rel_rot_error_by_gap'][g]['uncorrected_deg'] for g in gaps]
    cor = [consistency['rel_rot_error_by_gap'][g]['corrected_deg'] for g in gaps]
    ax_g.bar(x - 0.2, unc, 0.4, label='GT as RGB-camera pose (uncorrected)')
    ax_g.bar(x + 0.2, cor, 0.4, label='GT as event-camera pose (corrected)')
    ax_g.set_xticks(x, gaps)
    ax_g.set(xlabel='keyframe gap', ylabel='relative rotation error [deg]', title='GT vs SfM relative rotations')
    ax_g.legend(fontsize='small')

    sweep = consistency['time_offset_sweep']
    ax_t.plot(sweep['offsets_ms'], sweep['rel_rot_error_deg'], 'o-', ms=3)
    ax_t.axvline(consistency['best_time_offset_ms'], color='tab:red', ls='--', lw=0.8,
                 label=f"best: {consistency['best_time_offset_ms']:+.1f} ms")
    ax_t.set(xlabel='time offset added to frame timestamps [ms]', ylabel='relative rotation error [deg]',
             title='Camera-mocap time offset')
    ax_t.legend(fontsize='small')
    for ax in (ax_g, ax_t):
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------------

def _recall_key(m: float, d: float) -> str:
    return f'{m * 100:g}cm_{d:g}deg'


def _fmt(v, spec='.2f', scale=1.0) -> str:
    return '-' if v is None else format(v * scale, spec)


def _md_table(headers: list, rows: list) -> str:
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join('---' for _ in headers) + '|']
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def build_tables(metrics: dict) -> str:
    s, known, cons = metrics['setup'], metrics['known_pose_map'], metrics['groundtruth_consistency']
    out = ['# SfM map evaluation\n', '## Setup\n', _md_table(['Item', 'Value'], [
        ['Frames (valid GT)', f"{s['num_frames']} ({s['num_valid_frames']})"],
        ['Duration', f"{s['duration_s']:.1f} s"],
        ['GT rate', f"{s['gt_rate_hz']:.0f} Hz"],
        ['Calibration', s['calibration']],
        ['Keyframe selection', f"{s['keyframe_translation_m'] * 100:g} cm / {s['keyframe_rotation_deg']:g} deg"],
        ['Keyframes / held-out frames', f"{s['num_keyframes']} / {s['num_held_out']}"],
        ['COLMAP', s['colmap']],
    ])]

    recall_keys = [_recall_key(m, d) for m, d in EVAL_RECALL_THRESHOLDS]
    rows = []
    for m in metrics['incremental_models']:
        a = m.get('absolute', {})
        rows.append([m['model'], m['status'], m['num_images'], f"{m['registered_fraction'] * 100:.1f}%",
                     f"{m['first_frame']} - {m['last_frame']}", m['num_points3D'],
                     _fmt(m['mean_track_length']), _fmt(m['mean_reproj_error_px']),
                     _fmt(m.get('sim3_scale'), '.3f'), _fmt(a.get('ate_rmse_m'), '.2f', 100),
                     _fmt(a.get('pos_median_m'), '.2f', 100), _fmt(a.get('rot_median_deg')),
                     _fmt(m.get('rpe', {}).get('rot_mean_deg')), _fmt(m.get('rpe', {}).get('trans_mean_m'), '.2f', 100)]
                    + [_fmt(a.get('recall', {}).get(k), '.1f', 100) for k in recall_keys])
    out += ['\n## Incremental SfM models (aligned with Sim(3) to the GT)\n', _md_table(
        ['Model', 'Status', 'Images', 'Reg. %', 'Frames', 'Points', 'Track len.', 'Reproj. [px]', 'Scale',
         'ATE RMSE [cm]', 'Pos. med. [cm]', 'Rot. med. [deg]', f'RPE rot [deg] (d={EVAL_RPE_DELTA})',
         f'RPE trans [cm] (d={EVAL_RPE_DELTA})'] + [f'Recall {k} [%]' for k in recall_keys], rows)]

    rows = [['all', known['num_images'], known['num_points3D'], _fmt(known['mean_obs_per_image'], '.0f'),
             _fmt(known['mean_track_length']), _fmt(known['mean_reproj_error_px']),
             _fmt(known['median_obs_reproj_error_px']), _fmt(known['p95_obs_reproj_error_px'])]]
    for label in ('slow', 'fast'):
        k = known[f'{label}_motion']
        rows.append([f"{label} (< / >= {known['fast_motion_threshold_deg_s']:g} deg/s)", k['num_images'], '-',
                     _fmt(k['mean_obs_per_image'], '.0f'), '-', _fmt(k['mean_obs_reproj_error_px']),
                     _fmt(k['median_obs_reproj_error_px']), '-'])
    out += ['\n## Known-pose map (triangulated with fixed GT poses)\n', _md_table(
        ['Subset', 'Images', 'Points', 'Obs./image', 'Track len.', 'Mean reproj. [px]', 'Median reproj. [px]',
         'P95 reproj. [px]'], rows)]

    rows = [[f'model {k}', v['num_points'], _fmt(v['median_nn_dist_m'], '.2f', 100), _fmt(v['mean_nn_dist_m'], '.2f', 100),
             _fmt(v['p90_nn_dist_m'], '.2f', 100)] for k, v in metrics['map_comparison'].items()]
    out += ['\n## Map comparison (aligned SfM points -> nearest known-pose map point)\n', _md_table(
        ['Map', 'Points', 'Median [cm]', 'Mean [cm]', 'P90 [cm]'], rows)]

    rows = [[g, _fmt(v['uncorrected_deg'], '.3f'), _fmt(v['corrected_deg'], '.3f')]
            for g, v in cons['rel_rot_error_by_gap'].items()]
    out += ['\n## Ground-truth consistency (relative rotations GT vs SfM)\n', _md_table(
        ['Keyframe gap', 'GT as RGB pose [deg]', 'GT as event-camera pose [deg]'], rows),
        f"\n- Fitted constant rotation offset GT -> SfM: {cons['fitted_offset_deg']:.2f} deg "
        f"(R_cam1_cam0 from calibration: {cons['R_cam1_cam0_deg']:.2f} deg; difference "
        f"{cons['fitted_vs_R_cam1_cam0_deg']:.2f} deg, vs. inverse {cons['fitted_vs_R_cam0_cam1_deg']:.2f} deg)",
        f"- Best camera-mocap time offset: {cons['best_time_offset_ms']:+.1f} ms\n"]
    return '\n'.join(out)


# --------------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------------

def _json_default(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f'not JSON serialisable: {type(o)}')


def _save(fig: plt.Figure, name: str):
    fig.savefig(FIGURES_DIR / name, dpi=150, bbox_inches='tight')
    plt.close(fig)


def evaluate_map():
    """Compute all map metrics, write metrics.json + tables.md and the figures. Requires build_map() to have run."""
    if not db_path.is_file() or not SFM_GT_MAP.is_dir():
        raise FileNotFoundError('No map found - run build_map() first.')
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_frame_groundtruth()
    timestamps, filenames = load_frames()
    t_gt = load_groundtruth()[0]
    keypoints = load_keyframe_keypoints()
    _, held_out = load_frame_split()

    models = evaluate_incremental_models(gt, num_keyframes=len(keypoints))
    known_rec, known_reproj, known_metrics = evaluate_known_pose_map(gt)
    aligned = [m for m in models if m.aligned]
    largest = max(models, key=lambda m: m.rec.num_reg_images())

    metrics = dict(
        setup=dict(
            num_frames=len(filenames), num_valid_frames=len(gt.names),
            duration_s=float(gt.t[-1] - gt.t[0]), gt_rate_hz=float((len(t_gt) - 1) / (t_gt[-1] - t_gt[0])),
            calibration=f'{CALIBRATION_FILE.parent.name} (OPENCV model, intrinsics fixed); GT converted from event to RGB camera',
            keyframe_translation_m=SFM_TRANSLATION_THRESHOLD_M, keyframe_rotation_deg=SFM_ROTATION_THRESHOLD_DEG,
            num_keyframes=len(keypoints), num_held_out=len(held_out),
            colmap=f'SIFT, sequential matching (overlap {SFM_SEQUENTIAL_OVERLAP}), incremental mapping, no intrinsics refinement',
        ),
        incremental_models=[m.metrics for m in models],
        known_pose_map=known_metrics,
        map_comparison={m.name: compare_point_clouds(m.rec, known_rec) for m in aligned},
        groundtruth_consistency=groundtruth_consistency(largest.rec),
    )

    METRICS_FILE.write_text(json.dumps(metrics, indent=2, default=_json_default))
    TABLES_FILE.write_text(build_tables(metrics), encoding='utf-8')

    _save(plot_overview(gt, known_rec, models), 'overview_3d.png')
    _save(plot_timeline(gt, keypoints, models), 'timeline.png')
    _save(plot_pose_errors_over_time(gt, models), 'pose_errors_over_time.png')
    _save(plot_reprojection_histogram(known_reproj, models), 'reprojection_error_hist.png')
    _save(plot_example_frames(gt, keypoints), 'example_frames.png')
    _save(plot_groundtruth_consistency(metrics['groundtruth_consistency']), 'gt_consistency.png')
    _save(plot_map(known_rec, gt, 'Known-pose map'), 'map_known_poses.png')
    export_interactive_map(known_rec, gt, models, FIGURES_DIR / 'map_interactive.html')

    known_rec.export_PLY(SFM_OUTPUT_DIR / 'known_poses.ply')
    for m in aligned:
        m.rec.export_PLY(SFM_OUTPUT_DIR / f'model_{m.name}_aligned.ply')

    print(f'Wrote {METRICS_FILE}, {TABLES_FILE} and figures to {FIGURES_DIR}')
    return metrics


if __name__ == '__main__':
    evaluate_map()

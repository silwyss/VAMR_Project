import json
import os
import shutil
import numpy as np
import pycolmap
from pathlib import Path
from scipy.spatial.transform import Rotation
from src.models.Camera import CameraCalibration
from src.constants import (CALIBRATION_FILE, IMAGES_DIR, SFM_CONFIG_FILE, SFM_GT_MAP, SFM_MAP, SFM_MAP_FRAMES_FILE,
                           SFM_MAP_IMAGE_COUNT_THRESHOLD, SFM_MAP_MAX_ROTATION_ERROR_DEG, SFM_MAP_POINT_COUNT_THRESHOLD,
                           SFM_OUTPUT_DIR, SFM_QUERY_FRAMES_FILE, SFM_ROTATION_THRESHOLD_DEG, SFM_SEQUENTIAL_OVERLAP,
                           SFM_TRANSLATION_THRESHOLD_M)
from src.SfM.data import correct_poses_to_cam0, interpolate_poses, load_calibration, load_frames, load_groundtruth

db_path = SFM_OUTPUT_DIR / 'database.db'


def map_config() -> dict:
    """Parameters the database (keyframes, features, matches) depends on. A change triggers a rebuild."""
    return dict(
        rotation_threshold_deg=SFM_ROTATION_THRESHOLD_DEG,
        translation_threshold_m=SFM_TRANSLATION_THRESHOLD_M,
        sequential_overlap=SFM_SEQUENTIAL_OVERLAP,
        calibration_file=CALIBRATION_FILE.relative_to(CALIBRATION_FILE.parents[1]).as_posix(),
    )


def _database_is_current() -> bool:
    if not db_path.is_file() or not SFM_CONFIG_FILE.is_file():
        return False
    stored = json.loads(SFM_CONFIG_FILE.read_text())
    if stored != map_config():
        print(f'Map parameters changed ({stored} -> {map_config()}), rebuilding the map.')
        return False
    return SFM_MAP_FRAMES_FILE.is_file() and SFM_QUERY_FRAMES_FILE.is_file()


def load_frame_split() -> tuple[list[str], list[str]]:
    """(map keyframes, held-out query frames) exactly as used when the map was built."""
    if not SFM_MAP_FRAMES_FILE.is_file() or not SFM_QUERY_FRAMES_FILE.is_file():
        raise FileNotFoundError('No frame split found - run build_map() first.')
    return SFM_MAP_FRAMES_FILE.read_text().split(), SFM_QUERY_FRAMES_FILE.read_text().split()

def select_frames(filenames: list, R_wc, p_wc: np.ndarray) -> tuple[list, list]:
    frames = [filenames[0]]
    held_out = []
    last = 0
    # Only take points with enough transformation to last selected frame (few frames for slow motion, many frames for fast motion)
    for i in range(1, len(p_wc)):
        dp = np.linalg.norm(p_wc[i] - p_wc[last])
        dr = (R_wc[last].inv() * R_wc[i]).magnitude()
        if dp > SFM_TRANSLATION_THRESHOLD_M or dr > np.radians(SFM_ROTATION_THRESHOLD_DEG):
            last = i
            frames.append(filenames[i])
            continue
        held_out.append(filenames[i])
    return frames, held_out

def rebuild_map_data(calibration: CameraCalibration, filenames: np.ndarray, R_wc, p_wc: np.ndarray, opts):
    frames, held_out = select_frames(filenames, R_wc, p_wc)
    if os.path.isdir(SFM_OUTPUT_DIR):
        shutil.rmtree(SFM_OUTPUT_DIR, True)
    os.mkdir(SFM_OUTPUT_DIR)
    SFM_MAP_FRAMES_FILE.write_text('\n'.join(frames) + '\n')
    SFM_QUERY_FRAMES_FILE.write_text('\n'.join(held_out) + '\n')
    camera_params = [f'{d}' for d in [calibration.intrinsics.fx, calibration.intrinsics.fy, calibration.intrinsics.u, calibration.intrinsics.v, calibration.distortion.k1, calibration.distortion.k2, calibration.distortion.p1, calibration.distortion.p2]]
    reader = pycolmap.ImageReaderOptions(camera_model='OPENCV', camera_params=','.join(camera_params))
    pycolmap.extract_features(db_path, IMAGES_DIR, image_names=frames, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    pycolmap.match_sequential(db_path, pairing_options=pycolmap.SequentialPairingOptions(overlap=SFM_SEQUENTIAL_OVERLAP))

    pycolmap.incremental_mapping(db_path, IMAGES_DIR, SFM_OUTPUT_DIR, options=opts)
    # written last: only a complete build counts as current
    SFM_CONFIG_FILE.write_text(json.dumps(map_config(), indent=2))

def build_map():
    calibration = load_calibration()[0]
    timestamps, filenames = load_frames()
    R_wc, p_wc, valid = interpolate_poses(*load_groundtruth(), timestamps)
    filenames = filenames[valid]
    R_wc, p_wc = correct_poses_to_cam0(R_wc[valid], p_wc[valid])
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = False
    opts.ba_refine_extra_params = False
    opts.ba_refine_principal_point = False

    name_to_idx = {n: i for i, n in enumerate(filenames)}
    
    if not _database_is_current():
        rebuild_map_data(calibration, filenames, R_wc, p_wc, opts)

    _triangulate_known_poses(name_to_idx, R_wc, p_wc, opts)
    _align_incremental_models(name_to_idx, R_wc, p_wc)
    
def _triangulate_known_poses(name_to_idx: dict,  R_wc: Rotation, p_wc: np.ndarray, opts):
    if os.path.isdir(SFM_GT_MAP):
        shutil.rmtree(SFM_GT_MAP, True)

    db = pycolmap.Database.open(db_path)
    rec = pycolmap.Reconstruction()
    cam = db.read_all_cameras()[0]
    rec.add_camera_with_trivial_rig(cam)

    for db_img in db.read_all_images():
        kps = db.read_keypoints(db_img.image_id)[:, :2]
        img = pycolmap.Image(name=db_img.name, keypoints=kps,
                              camera_id=cam.camera_id, image_id=db_img.image_id)

        idx = name_to_idx[db_img.name]
        R_cw = R_wc[idx].inv()
        t_cw = -R_cw.as_matrix() @ p_wc[idx]
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R_cw.as_quat()), t_cw)
        rec.add_image_with_trivial_frame(img, cam_from_world)

    db.close()
    tri = pycolmap.triangulate_points(rec, db_path, IMAGES_DIR, SFM_GT_MAP, options=opts)
    print(tri.summary())

def _align_incremental_models(name_to_idx: dict, R_wc: Rotation, p_wc: np.ndarray):
    if os.path.isdir(SFM_MAP):
        shutil.rmtree(SFM_MAP, True)

    for model_dir in incremental_model_dirs():
        i = model_dir.name
        rec = pycolmap.Reconstruction(model_dir.absolute())
        if len(rec.images) < SFM_MAP_IMAGE_COUNT_THRESHOLD or len(rec.points3D) < SFM_MAP_POINT_COUNT_THRESHOLD:
            print(f'Model {i} failed because only {f'{len(rec.images)} images' if len(rec.images) < SFM_MAP_IMAGE_COUNT_THRESHOLD else f'{len(rec.points3D)} points'} were in the map.')
            continue
        sim3, status = align_to_groundtruth(rec, name_to_idx, R_wc, p_wc)
        if sim3 is None:
            print(f'Model {i}: {status}')
            continue

        print(rec.summary())

        aligned_model_output = Path(SFM_MAP / i)
        aligned_model_output.mkdir(parents=True, exist_ok=True)
        rec.write(aligned_model_output)

def incremental_model_dirs() -> list[Path]:
    """Folders of the incremental COLMAP models (outputs/SfM/0, 1, ...), sorted numerically."""
    return sorted((p for p in SFM_OUTPUT_DIR.iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name))

def align_to_groundtruth(rec: pycolmap.Reconstruction, name_to_idx: dict, R_wc: Rotation,
                         p_wc: np.ndarray) -> tuple[pycolmap.Sim3d | None, str]:
    """Sim(3)-align `rec` IN PLACE so its camera centres match the GT positions (metres, GT world frame).

    The transform is estimated from camera centres only. When the camera mostly rotates in place the
    centres are nearly collinear and the rotation about that line is unconstrained, so the aligned
    orientations are checked against the GT as well before the transform is applied.
    Returns (transform, 'ok'), or (None, reason) if the alignment failed (rec is then unchanged)."""
    images = [im for im in rec.images.values() if im.name in name_to_idx]
    idx = np.array([name_to_idx[im.name] for im in images])
    sim3 = pycolmap.align_reconstruction_to_locations(rec, [im.name for im in images], p_wc[idx], min_common_images=3,
                                                      ransac_options=pycolmap.RANSACOptions(max_error=0.05))
    if sim3 is None:
        return None, 'Sim(3) alignment to the GT positions failed.'

    # orientations after the transform: R_wc_aligned = R_sim3 * R_wc_model
    R_model = Rotation.from_matrix(np.stack([im.cam_from_world().rotation.matrix() for im in images])).inv()
    R_aligned = Rotation.from_matrix(sim3.rotation.matrix()) * R_model
    rot_err = np.degrees((R_wc[idx].inv() * R_aligned).magnitude())
    if np.median(rot_err) > SFM_MAP_MAX_ROTATION_ERROR_DEG:
        return None, (f'aligned orientations disagree with the GT (median {np.median(rot_err):.1f} deg > '
                      f'{SFM_MAP_MAX_ROTATION_ERROR_DEG:g} deg), alignment rejected.')

    rec.transform(sim3)
    return sim3, 'ok'

if __name__ == '__main__':
    build_map()
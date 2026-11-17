import os
import shutil
import numpy as np
import pycolmap
from pathlib import Path
from scipy.spatial.transform import Rotation
from src.models.Camera import CameraCalibration
from src.constants import IMAGES_DIR, OUTPUT_DIR, SFM_MAP_IMAGE_COUNT_THRESHOLD, SFM_MAP_POINT_COUNT_THRESHOLD, SFM_OUTPUT_DIR, SFM_ROTATION_THRESHOLD_DEG, SFM_TRANSLATION_THRESHOLD
from src.SfM.data import correct_poses_to_cam0, interpolate_poses, load_calibration, load_frames, load_groundtruth

db_path = SFM_OUTPUT_DIR / 'database.db'

def rebuild_map_data(calibration: CameraCalibration, filenames: np.ndarray, R_wc, p_wc: np.ndarray, opts):
    frames = [filenames[0]]
    held_out = []
    last = 0
    # Only take points with enough transformation to last selected frame (few frames for slow motion, many frames for fast motion)
    for i in range(1, len(p_wc)):
        dp = np.linalg.norm(p_wc[i] - p_wc[last])
        dr = (R_wc[last].inv() * R_wc[i]).magnitude()
        if dp > SFM_TRANSLATION_THRESHOLD or dr > np.radians(SFM_ROTATION_THRESHOLD_DEG):
            last = i
            frames.append(filenames[i])
            continue
        held_out.append(filenames[i])

    if os.path.isdir(SFM_OUTPUT_DIR):
        shutil.rmtree(SFM_OUTPUT_DIR,  True)
    os.mkdir(SFM_OUTPUT_DIR)
    camera_params = [f'{d}' for d in [calibration.intrinsics.fx, calibration.intrinsics.fy, calibration.intrinsics.u, calibration.intrinsics.v, calibration.distortion.k1, calibration.distortion.k2, calibration.distortion.p1, calibration.distortion.p2]]
    reader = pycolmap.ImageReaderOptions(camera_model='OPENCV', camera_params=','.join(camera_params))
    pycolmap.extract_features(db_path, IMAGES_DIR, image_names=frames, camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader)
    pycolmap.match_sequential(db_path, pairing_options=pycolmap.SequentialPairingOptions(overlap=15))

    pycolmap.incremental_mapping(db_path, IMAGES_DIR, SFM_OUTPUT_DIR, options=opts)

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
    
    if not os.path.isfile(db_path):
        rebuild_map_data(calibration, filenames, R_wc, p_wc, opts)

    _triangulate_known_poses(name_to_idx, R_wc, p_wc, opts)
    _align_incremental_models(name_to_idx, p_wc)
    
def _triangulate_known_poses(name_to_idx: dict,  R_wc: Rotation, p_wc: np.ndarray, opts):
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
    tri = pycolmap.triangulate_points(rec, db_path, IMAGES_DIR, SFM_OUTPUT_DIR / 'known_poses', options=opts)
    print(tri.summary())

def _align_incremental_models(name_to_idx: dict, p_wc: np.ndarray):
    maps = sorted(p for p in SFM_OUTPUT_DIR.iterdir() if p.is_dir() and p.name.isdigit())
    i = 0
    for map in maps:
        rec = pycolmap.Reconstruction(map.absolute())
        if len(rec.images) < SFM_MAP_IMAGE_COUNT_THRESHOLD or len(rec.points3D) < SFM_MAP_POINT_COUNT_THRESHOLD:
            continue
        names = [im.name for im in rec.images.values()]
        gt_centers = np.array([p_wc[name_to_idx[n]] for n in names])

        sim3 = pycolmap.align_reconstruction_to_locations(rec, names, gt_centers, min_common_images=3, ransac_options=pycolmap.RANSACOptions(max_error=0.05))
        rec.transform(sim3)

        print(rec.summary())

        aligned_model_output = Path(SFM_OUTPUT_DIR / 'aligned' / str(i))
        aligned_model_output.mkdir(parents=True, exist_ok=True)
        rec.write(aligned_model_output)
        i += 1

if __name__ == '__main__':
    build_map()
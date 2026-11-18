from tkinter import NO

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from src.SfM.data import interpolate_poses, load_calibration, load_frames, load_groundtruth
from src.constants import IMAGES_DIR

SIFT = cv2.SIFT.create(nfeatures=2000, nOctaveLayers=3, contrastThreshold=0.02,
                       edgeThreshold=10, sigma=1.6)

def check_ground_truth_repeated():
    cam0 = load_calibration()[0]
    K, dist = cam0.intrinsics.asArray(), cam0.distortion.asArray()

    check_ground_truth(K, dist, 10)
    check_ground_truth(K, dist, 150)
    check_ground_truth(K, dist, 300)
    check_ground_truth(K, dist, 450)
    check_ground_truth(K, dist, 800)
    check_ground_truth(K, dist, 1200)

def check_ground_truth(K, distortion, idx1):
    t_frames, filenames = load_frames()
    R_wc, p_wc, _ = interpolate_poses(*load_groundtruth(), t_frames)
    idx2 = idx1 + 15
    if idx2 >= len(filenames):
        return
    img1 = cv2.imread(IMAGES_DIR / filenames[idx1], cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(IMAGES_DIR / filenames[idx2], cv2.IMREAD_GRAYSCALE)

    R1, R2, p1, p2 = R_wc[idx1].as_matrix(), R_wc[idx2].as_matrix(), p_wc[idx1], p_wc[idx2]
    hypotheses = {'T_wc': (R2.T @ R1, R2.T @ (p1 - p2)),
                    'T_cw': (R2 @ R1.T, p2 - R2 @ R1.T @ p1)}
    for flip in (False, True):
        if flip:
            img1 = cv2.flip(img1, 1)
            img2 = cv2.flip(img2, 1)
        R, t = _estimate_relative_pose(img1, img2, K, distortion)
        for name, (R_gt, t_gt) in hypotheses.items():
            e_r, e_t = _rot_angle_deg(R @ R_gt.T), _dir_angle_deg(t, t_gt)
            ok = 'OK' if e_r < 2 and e_t < 10 else '  '
            print(f'{idx1:4d}-{idx2:4d} flip={flip!s:5} {name}: rot {e_r:5.2f}°  t {e_t:5.1f}°  {ok}')

def _estimate_relative_pose(img1, img2, K, distortion):
    """Return R, t with x2 = R @ x1 + t (T_c2c1), t unit length."""
    pts1, d1 = SIFT.detectAndCompute(img1, None)
    pts2, d2 = SIFT.detectAndCompute(img2, None)
    matches = [m for m, n in cv2.BFMatcher(cv2.NORM_L2).knnMatch(d1, d2, k=2) if m.distance < n.distance * 0.75]
    kp1 = np.float32([pts1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    kp2 = np.float32([pts2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(kp1, K, distortion, P=K)
    undistorted2 = cv2.undistortPoints(kp2, K, distortion, P=K)
    E, mask = cv2.findEssentialMat(undistorted, undistorted2, K, cv2.RANSAC, 0.999, 1.0)
    _, R, t, _ = cv2.recoverPose(E, undistorted, undistorted2, K, mask=mask)
    return R, t.ravel()
    
def _rot_angle_deg(R):
    return np.degrees(Rotation.from_matrix(R).magnitude())

def _dir_angle_deg(a, b):
    c = a @ b / (np.linalg.norm(a) * np.linalg.norm(b))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))

if __name__ == '__main__':
    check_ground_truth_repeated()
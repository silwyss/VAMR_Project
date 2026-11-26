"""Re-triangulate the WP1 map with EI-Nexus' own image keypoints (SuperPoint) -> outputs/WP3/sp_map/.

Why: the WP1 map only has 3D points for COLMAP SIFT keypoints, so only ~6 % of EI-Nexus matches could be lifted to 3D
(nearest SIFT keypoint within 2 px). EI-Nexus' image side is a frozen, deterministic SuperPoint. If the map keyframes
are described with exactly that extractor (same virtual camera, same mask, same top-k), every map-image keypoint that
EI-Nexus matches is a keypoint of the map, and lifting becomes an exact lookup.

Steps (hloc "triangulation from reference poses"):
  1. EI-Nexus image features of every keyframe (virtual-camera images in outputs/WP3/sp_map/images) -> features.h5
  2. reference model: PINHOLE virtual camera, GT cam0 poses (the virtual camera has cam0's orientation, only K
     differs and distortion is removed), keypoints attached
  3. pairs: each keyframe with its NUM_PAIRS nearest keyframes by GT pose -> pairs.txt
  4. image-image matching with LightGlue (SuperPoint weights) -> matches.h5
  5. hloc.triangulation: geometric verification with the known poses, triangulation, point-only BA -> model/
"""
import cv2
import h5py
import numpy as np
import pycolmap
import torch
from scipy.spatial.transform import Rotation

from src.constants import OUTPUT_DIR
from src.WP3.ei_nexus import EINexus
from src.WP3.geometry import FrameData, VirtualCamera

SP_MAP_DIR = OUTPUT_DIR / 'WP3' / 'sp_map'
NUM_PAIRS = 10


def build(variant: str = 'MVSEC_SP_LG'):
    from hloc import triangulation
    from hloc.utils.parsers import names_to_pair
    from lightglue import LightGlue

    SP_MAP_DIR.mkdir(parents=True, exist_ok=True)
    img_dir, ref_dir = SP_MAP_DIR / 'images', SP_MAP_DIR / 'reference'
    img_dir.mkdir(exist_ok=True)
    ref_dir.mkdir(exist_ok=True)
    feats_path, matches_path, pairs_path = (SP_MAP_DIR / f for f in ('features.h5', 'matches.h5', 'pairs.txt'))

    cam, fd = VirtualCamera(), FrameData.load()
    model = EINexus(variant)
    names = list(fd.names[fd.key_idx])

    # 1. features (exactly EI-Nexus' image keypoints) + virtual-camera images
    keypoints = {}
    with h5py.File(feats_path, 'w') as f:
        for i in fd.key_idx:
            n = fd.names[i]
            img = cam.image(n)
            cv2.imwrite(str(img_dir / n), img)
            xy, desc, score = model.extract_image(img, cam.image_valid)
            keypoints[n] = xy
            g = f.create_group(n)
            g.create_dataset('keypoints', data=xy.astype(np.float32))
            g.create_dataset('descriptors', data=desc.T.astype(np.float32))
            g.create_dataset('scores', data=score.astype(np.float32))
            g.create_dataset('image_size', data=np.array(cam.size))
    print(f'features: {len(names)} keyframes')

    # 2. reference model: PINHOLE virtual camera, GT cam0 poses, keypoints attached (COLMAP origin: +0.5)
    ref = pycolmap.Reconstruction()
    K = cam.K
    camera = pycolmap.Camera(model='PINHOLE', width=cam.size[0], height=cam.size[1], camera_id=1,
                             params=[K[0, 0], K[1, 1], K[0, 2], K[1, 2]])
    ref.add_camera_with_trivial_rig(camera)
    for image_id, i in enumerate(fd.key_idx, start=1):
        R_cw = fd.R0[i].inv()
        t_cw = -R_cw.as_matrix() @ fd.p0[i]
        im = pycolmap.Image(name=fd.names[i], keypoints=keypoints[fd.names[i]].astype(np.float64) + 0.5,
                            camera_id=1, image_id=image_id)
        ref.add_image_with_trivial_frame(im, pycolmap.Rigid3d(pycolmap.Rotation3d(R_cw.as_quat()), t_cw))
    ref.write(ref_dir)

    # 3. pairs by GT pose
    pairs = set()
    for i in fd.key_idx:
        for k in fd.nearest_keyframes(i, NUM_PAIRS + 1):
            if k != i:
                pairs.add(tuple(sorted((fd.names[i], fd.names[k]))))
    pairs = sorted(pairs)
    pairs_path.write_text('\n'.join(f'{a} {b}' for a, b in pairs) + '\n')

    # 4. LightGlue image-image matching on these features
    lg = LightGlue(features='superpoint').eval().cuda()
    size = torch.tensor(cam.size, dtype=torch.float32)[None].cuda()

    def load(f, n):
        return {'keypoints': torch.from_numpy(f[n]['keypoints'][()])[None].cuda(),
                'descriptors': torch.from_numpy(f[n]['descriptors'][()].T)[None].cuda(), 'image_size': size}

    with h5py.File(feats_path, 'r') as f, h5py.File(matches_path, 'w') as fm, torch.no_grad():
        for a, b in pairs:
            out = lg({'image0': load(f, a), 'image1': load(f, b)})
            g = fm.create_group(names_to_pair(a, b))
            g.create_dataset('matches0', data=out['matches0'][0].cpu().numpy().astype(np.int32))
            g.create_dataset('matching_scores0', data=out['matching_scores0'][0].cpu().numpy().astype(np.float16))
    print(f'matches: {len(pairs)} pairs')

    # 5. triangulation with the known poses (hloc's steps, run individually: hloc writes the verified two-view
    #    geometries with configuration UNDEFINED, which COLMAP >= 4 skips -> mark them CALIBRATED, which they are:
    #    the verification used the known poses and intrinsics)
    model_dir = SP_MAP_DIR / 'model'
    model_dir.mkdir(exist_ok=True)
    database = model_dir / 'database.db'
    image_ids = triangulation.create_db_from_model(ref, database)
    with pycolmap.Database.open(database) as db:
        triangulation.import_features(image_ids, db, feats_path)
        triangulation.import_matches(image_ids, db, pairs_path, matches_path, None, False)
        triangulation.geometric_verification(image_ids, ref, db, feats_path, pairs_path, matches_path)
        pair_ids, geometries = db.read_two_view_geometries()
        for pair_id, tvg in zip(pair_ids, geometries):
            if tvg.config == pycolmap.TwoViewGeometryConfiguration.UNDEFINED and len(tvg.inlier_matches):
                tvg.config = pycolmap.TwoViewGeometryConfiguration.CALIBRATED
                db.update_two_view_geometry(*pycolmap.pair_id_to_image_pair(pair_id), tvg)
    opts = pycolmap.IncrementalPipelineOptions()
    opts.ba_refine_focal_length = opts.ba_refine_principal_point = opts.ba_refine_extra_params = False
    rec = pycolmap.triangulate_points(ref, database, img_dir, model_dir, options=opts)
    print(rec.summary())
    return rec


if __name__ == '__main__':
    build()

"""M3 / M4: relocalize query frames in the WP1 map and evaluate against the GT.

  E0 (--mode rgb)    RGB query frame  <-> map keyframes with SuperPoint + LightGlue (image-image). Validates the
                     lifting + PnP code; an upper bound for the event pipeline. Evaluated against the cam0 GT.
  E1 (--mode events) event window     <-> map keyframes with EI-Nexus. Evaluated against the cam1 (event) GT.

Per query: oracle retrieval of the K nearest keyframes (by GT pose), matching, lifting of the matched keyframe
keypoints to 3D (nearest 3D-backed SIFT keypoint, localize.MapIndex), PnP + LO-RANSAC in the virtual pinhole camera.
A query counts as localized if PnP returns >= MIN_INLIERS inliers; otherwise it is a failure (kept in the stats).

Outputs: outputs/WP3/localization/<name>/results.csv, summary.md
"""
import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.constants import EVAL_RECALL_THRESHOLDS, OUTPUT_DIR, ROOT_DIR, SFM_GT_MAP
from src.WP3.events import EventReader, EventWindow, event_count_image, flicker_phase_gate, voxel_grid
from src.WP3.geometry import FrameData, VirtualCamera, pose_error
from src.WP3.localize import PnP, MapIndex, collect_correspondences


class RGBMatcher:
    """Image-image matching for E0: EI-Nexus' SuperPoint (identical keypoints to the sp map) + LightGlue
    (cvg/LightGlue, SuperPoint weights). Keyframe features are cached."""

    def __init__(self, ei_model, size, device='cuda'):
        from lightglue import LightGlue
        self.ei, self.device = ei_model, device
        self.matcher = LightGlue(features='superpoint').eval().to(device)
        self.size = torch.tensor(size, dtype=torch.float32)[None].to(device)
        self.cache = {}

    def features(self, key, image: np.ndarray, valid: np.ndarray):
        if key not in self.cache:
            xy, desc, _ = self.ei.extract_image(image, valid)
            self.cache[key] = {'keypoints': torch.from_numpy(xy.astype(np.float32))[None].to(self.device),
                               'descriptors': torch.from_numpy(desc)[None].to(self.device), 'image_size': self.size}
        return self.cache[key]

    @torch.no_grad()
    def match(self, f0, f1):
        out = self.matcher({'image0': f0, 'image1': f1})
        m = out['matches'][0].cpu().numpy()
        s = out['scores'][0].cpu().numpy()
        return f0['keypoints'][0].cpu().numpy()[m[:, 0]], f1['keypoints'][0].cpu().numpy()[m[:, 1]], s


def summarise(rows, thresholds=EVAL_RECALL_THRESHOLDS) -> str:
    def stats(sel):
        n = len(sel)
        ok = [r for r in sel if r['success']]
        pe = np.array([r['pos_err_m'] for r in ok])
        re_ = np.array([r['rot_err_deg'] for r in ok])
        rec = {f'{m * 100:g}cm/{d:g}deg': sum(1 for r in ok if r['pos_err_m'] < m and r['rot_err_deg'] < d) / n
               for m, d in thresholds}
        return n, len(ok) / n, (np.median(pe) if len(pe) else np.nan), (np.median(re_) if len(re_) else np.nan), rec, \
            np.median([r['num_2d3d'] for r in sel]), np.median([r['num_inliers'] for r in sel])

    speeds = np.array([r['speed_deg_s'] for r in rows])
    e = np.percentile(speeds, [100 / 3, 200 / 3])
    groups = [('all', -np.inf, np.inf), (f'slow <{e[0]:.0f} deg/s', -np.inf, e[0]),
              (f'medium', e[0], e[1]), (f'fast >={e[1]:.0f} deg/s', e[1], np.inf)]
    keys = [f'{m * 100:g}cm/{d:g}deg' for m, d in thresholds]
    lines = ['| subset | n | success | median pos err | median rot err | ' + ' | '.join(f'recall {k}' for k in keys)
             + ' | 2D-3D (median) | inliers (median) |', '|' + '---|' * (7 + len(keys))]
    for name, lo, hi in groups:
        sel = [r for r in rows if lo <= r['speed_deg_s'] < hi]
        n, sr, pe, re_, rec, nc, ni = stats(sel)
        lines.append(f'| {name} | {n} | {sr:.2f} | {pe * 100:.1f} cm | {re_:.2f} deg | '
                     + ' | '.join(f'{rec[k]:.2f}' for k in keys) + f' | {nc:.0f} | {ni:.0f} |')
    cov = np.median([r['lift_coverage'] for r in rows])
    lines.append(f'\nMedian fraction of matches that could be lifted to 3D: {cov:.2f}')
    if 'map_gap_cm' in rows[0]:
        g = lambda k: np.median([r[k] for r in rows])
        lines.append(f'Median gap query -> nearest available keyframe: {g("map_gap_cm"):.1f} cm / '
                     f'{g("map_gap_deg"):.1f} deg; -> best retrieved keyframe: {g("ret_gap_cm"):.1f} cm / '
                     f'{g("ret_gap_deg"):.1f} deg')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=('rgb', 'events'), required=True)
    ap.add_argument('--window-ms', type=float, default=40.0)
    ap.add_argument('--variant', default='MVSEC_SP_LG')
    ap.add_argument('--num-queries', type=int, default=100)
    ap.add_argument('--num-keyframes', type=int, default=3)
    ap.add_argument('--map', choices=('sp', 'sift'), default='sp')
    ap.add_argument('--lift-radius', type=float, default=None, help='px; default 0.5 (sp) / 2.0 (sift, cam0 px)')
    ap.add_argument('--ransac-px', type=float, default=4.0)
    ap.add_argument('--min-inliers', type=int, default=12)
    ap.add_argument('--map-stride', type=int, default=1,
                    help='A: only every k-th map keyframe may be retrieved (larger viewpoint gap to the queries)')
    ap.add_argument('--retrieval', choices=('oracle', 'netvlad', 'einexus'), default='oracle', help='E4')
    ap.add_argument('--window-offset-ms', type=float, default=0.0,
                    help='D: the event window ends at t_frame + offset')
    ap.add_argument('--window-events', type=int, default=None, help='count window (overrides --window-ms)')
    ap.add_argument('--flicker', choices=('none', 'comb', 'gate'), default='none',
                    help='100 Hz lamp-flicker removal: per-pixel comb filter or polarity phase gate (events.py)')
    ap.add_argument('--rank-time', action='store_true', help='voxel grid binned by event rank, not timestamp')
    ap.add_argument('--query-dataset', default=None,
                    help='E7: take the queries from another EDS sequence of the same scene (e.g. Dataset_Dark); '
                         'all its frames are queries, the map stays peanuts_light')
    ap.add_argument('--name', default=None)
    args = ap.parse_args()
    if args.lift_radius is None:
        args.lift_radius = 0.5 if args.map == 'sp' else 2.0
    name = args.name or (f'E0_rgb_{args.map}map' if args.mode == 'rgb'
                         else f'E1_events_{args.window_ms:g}ms_{args.variant}_{args.map}map')
    if not args.name:
        name += (f'_stride{args.map_stride}' if args.map_stride != 1 else '') \
                + (f'_{args.retrieval}' if args.retrieval != 'oracle' else '') \
                + (f'_offset{args.window_offset_ms:+g}ms' if args.window_offset_ms else '') \
                + (f'_{args.window_events // 1000}kev' if args.window_events else '') \
                + (f'_flicker-{args.flicker}' if args.flicker != 'none' else '') \
                + ('_ranktime' if args.rank_time else '')
        if args.query_dataset:
            name = f'{Path(args.query_dataset).name}_{name}'
    out_dir = OUTPUT_DIR / 'WP3' / 'localization' / name
    out_dir.mkdir(parents=True, exist_ok=True)

    cam = VirtualCamera()
    fd = FrameData.load()                       # map sequence (keyframes)
    qd = FrameData.load_sequence(ROOT_DIR / args.query_dataset) if args.query_dataset else fd   # query sequence
    from src.WP3.sp_map import SP_MAP_DIR
    map_index = MapIndex(SP_MAP_DIR / 'model' if args.map == 'sp' else SFM_GT_MAP)
    to_map_px = (lambda xy: xy) if args.map == 'sp' else cam.to_cam0_pixels
    pnp = PnP(cam.K, cam.size, args.ransac_px, args.min_inliers)
    reader = EventReader(qd.dataset_dir / 'events.h5')
    t_q = qd.t[qd.query_idx] * 1e6
    usable = qd.query_idx[(t_q - 80e3 >= reader.t_start) & (t_q <= reader.t_end)]   # same set for every mode
    queries = usable[np.linspace(0, len(usable) - 1, args.num_queries).astype(int)]

    from src.WP3.ei_nexus import EINexus
    model = EINexus(args.variant)
    if args.mode == 'rgb':
        rgb = RGBMatcher(model, cam.size)
    candidates = fd.key_idx[::args.map_stride]
    if args.retrieval == 'netvlad':
        from src.WP3.retrieval import NetVLADRetrieval
        retriever = NetVLADRetrieval(cam, fd)
    elif args.retrieval == 'einexus':
        from src.WP3.retrieval import EINexusRetrieval
        retriever = EINexusRetrieval(fd, SP_MAP_DIR / 'features.h5')

    rows = []
    for qn, i in enumerate(queries):
        t0 = time.perf_counter()
        q2d, ids, scores, n_matches = [], [], [], 0
        if args.mode == 'events':
            t_end = qd.t_us(i) + int(round(args.window_offset_ms * 1e3))
            if args.flicker == 'gate':   # gate first, then take the count window from the gated stream
                ev = flicker_phase_gate(reader.window_by_time(t_end, args.window_ms if not args.window_events
                                                              else 4 * args.window_ms))
                if args.window_events:
                    ev = EventWindow(*(a[-args.window_events:] for a in (ev.x, ev.y, ev.t, ev.p)))
            else:
                ev = reader.window(t_end, None if args.window_events else args.window_ms, args.window_events,
                                   100.0 if args.flicker == 'comb' else None)
            ev = cam.events(ev)
            count = event_count_image(ev, cam.size)
            vox, mask = voxel_grid(ev, model.bins, cam.size, rank_time=args.rank_time), count > 0
            n_events = len(ev)
        else:
            img_q = cam.image(qd.names[i], images_dir=qd.images_dir)
            fq = rgb.features(('q', i), img_q, cam.image_valid)
            n_events = 0
        if args.retrieval == 'oracle':
            kfs = fd.nearest_keyframes(i, args.num_keyframes, candidates, qd)
        elif args.retrieval == 'netvlad':
            from src.WP3.retrieval import event_image
            kfs = retriever.topk(event_image(count) if args.mode == 'events' else img_q, candidates,
                                 args.num_keyframes)
        else:
            if args.mode == 'events':
                qdesc = model.extract_events(vox, mask)[1] if n_events >= 100 else np.zeros((0, 256), np.float32)
            else:
                qdesc = fq['descriptors'][0].cpu().numpy()
            kfs = retriever.topk(qdesc, candidates, args.num_keyframes)
        map_gap = fd.pose_gap(i, fd.nearest_keyframes(i, 1, candidates, qd), qd)    # best possible keyframe
        ret_dp, ret_dr = fd.pose_gap(i, kfs, qd)
        best = np.argmin(ret_dp / 0.05 + ret_dr / 5.0)                      # best retrieved keyframe
        for k in kfs:
            img_k = cam.image(fd.names[k])
            if args.mode == 'events':
                if n_events < 100:
                    continue
                res = model.match(vox, img_k, mask, cam.image_valid)
                pts_q, pts_k, s = res.matched_events, res.matched_image, res.scores
            else:
                pts_q, pts_k, s = rgb.match(fq, rgb.features(('k', k), img_k, cam.image_valid))
            n_matches += len(pts_q)
            ids.append(map_index.lift(fd.names[k], to_map_px(pts_k), args.lift_radius))
            q2d.append(pts_q)
            scores.append(s)
        if q2d:
            ids_all = np.concatenate(ids)
            pts2d, pts3d = collect_correspondences(np.concatenate(q2d), ids_all, np.concatenate(scores), map_index)
        else:
            ids_all, pts2d, pts3d = np.zeros(0), np.zeros((0, 2)), np.zeros((0, 3))
        pose = pnp(pts2d, pts3d)
        R_gt, p_gt = (qd.R1[i], qd.p1[i]) if args.mode == 'events' else (qd.R0[i], qd.p0[i])
        pe, re_ = pose_error(pose.R_cw, pose.t_cw, R_gt, p_gt) if pose.success else (np.inf, np.inf)
        # diagnostic: how many 2D-3D correspondences does the GT pose explain (reprojection < ransac px)?
        # GT pose explains many but PnP fails -> estimation problem; GT explains few -> matching (or GT frame) problem
        xc = R_gt.inv().apply(pts3d - p_gt)
        front = xc[:, 2] > 1e-6
        proj = (xc[front] @ cam.K.T)[:, :2] / xc[front, 2:3]
        gt_inliers = int(np.sum(np.linalg.norm(proj - pts2d[front], axis=1) < args.ransac_px))
        if pose.R_cw is not None:   # estimated pose (world <- camera), kept even if not "successful" (D analysis)
            C = -pose.R_cw.T @ pose.t_cw
            q = Rotation.from_matrix(pose.R_cw.T).as_quat()
        else:
            C, q = np.full(3, np.nan), np.full(4, np.nan)
        rows.append(dict(query=qd.names[i], t_s=float(qd.t[i]), speed_deg_s=float(qd.speed_deg_s[i]),
                         num_events=n_events, keyframes=' '.join(fd.names[kfs]),
                         map_gap_cm=float(map_gap[0][0] * 100), map_gap_deg=float(map_gap[1][0]),
                         ret_gap_cm=float(ret_dp[best] * 100), ret_gap_deg=float(ret_dr[best]),
                         num_matches=n_matches,
                         lift_coverage=float((ids_all >= 0).mean()) if len(ids_all) else 0.0,
                         num_2d3d=len(pts2d), num_inliers=pose.num_inliers, gt_inliers=gt_inliers,
                         success=pose.success,
                         pos_err_m=pe, rot_err_deg=re_,
                         est_x=C[0], est_y=C[1], est_z=C[2], est_qx=q[0], est_qy=q[1], est_qz=q[2], est_qw=q[3],
                         time_s=time.perf_counter() - t0))
        r = rows[-1]
        print(f'{qn + 1}/{len(queries)} {r["query"]} {r["speed_deg_s"]:.0f} deg/s: {r["num_matches"]} matches, '
              f'{r["num_2d3d"]} 2D-3D, {r["num_inliers"]} inl, err {pe * 100:.1f} cm / {re_:.2f} deg', flush=True)
    reader.close()

    with open(out_dir / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    table = summarise(rows)
    header = f'{name}: {vars(args)}\n\n'
    (out_dir / 'summary.md').write_text(header + table + '\n')
    print(header + table)


if __name__ == '__main__':
    main()

"""D: is the ~0.8 cm error of event queries (vs 0.3 cm for RGB queries) systematic? Offline, from saved poses.

Inputs: results.csv of localization runs (estimated camera centre + orientation per query, see run_localization.py).
  1. bias      error vector C_est - C_gt expressed in the GT camera frame: a constant offset (mean >> spread / sqrt(n))
               points at extrinsics / GT-convention problems, zero mean at noise.
  2. time      median position error against the GT at t_frame + tau for tau in [-60, 60] ms: a minimum away from 0
               means the estimate belongs to another instant (clock offset or window lag).
  3. extrinsics same query, same map: relative pose between the event-camera estimate (E1) and the RGB-camera
               estimate (E0), compared with the calibrated cam0 -> cam1 transform T_cn_cnm1. Independent of the GT.

Usage: python -m src.WP3.analyze_offset [--events RUN ...] [--rgb RUN]
Output: outputs/WP3/offset_analysis/summary.md, time_offset.png
"""
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from src.constants import OUTPUT_DIR
from src.SfM.data import correct_poses_to_cam0, interpolate_poses, load_cam0_to_cam1_transformation, load_groundtruth

RUNS = OUTPUT_DIR / 'WP3' / 'localization'
OUT = OUTPUT_DIR / 'WP3' / 'offset_analysis'
GOOD_M = 0.05   # only poses within 5 cm are analysed (excludes gross failures)


def load_run(name):
    rows = list(csv.DictReader(open(RUNS / name / 'results.csv')))
    rows = [r for r in rows if r['success'] == 'True' and float(r['pos_err_m']) < GOOD_M]
    t = np.array([float(r['t_s']) for r in rows])
    C = np.array([[float(r[k]) for k in ('est_x', 'est_y', 'est_z')] for r in rows])
    R = Rotation.from_quat([[float(r[k]) for k in ('est_qx', 'est_qy', 'est_qz', 'est_qw')] for r in rows])
    return [r['query'] for r in rows], t, R, C


def gt(t, cam):
    R1, p1, _ = interpolate_poses(*load_groundtruth(), t)
    return (R1, p1) if cam == 1 else correct_poses_to_cam0(R1, p1)


def bias(t, R, C, cam):
    R_gt, p_gt = gt(t, cam)
    e = R_gt.inv().apply(C - p_gt) * 100        # cm, GT camera frame (x right, y down, z forward)
    n = len(e)
    return (f'n={n}, mean [{", ".join(f"{v:+.2f}" for v in e.mean(0))}] cm '
            f'(+-{", ".join(f"{v:.2f}" for v in e.std(0) / np.sqrt(n))} s.e.), '
            f'std [{", ".join(f"{v:.2f}" for v in e.std(0))}] cm, |mean| {np.linalg.norm(e.mean(0)):.2f} cm, '
            f'median |e| {np.median(np.linalg.norm(e, axis=1)):.2f} cm, '
            f'median |e - mean| {np.median(np.linalg.norm(e - e.mean(0), axis=1)):.2f} cm'), e.mean(0)


def time_sweep(t, C, cam, taus):
    return np.array([np.median(np.linalg.norm(C - gt(t + tau, cam)[1], axis=1)) * 100 for tau in taus])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--events', nargs='+', default=['E1_events_40ms_MVSEC_SP_LG_spmap',
                                                    'E1_events_80ms_MVSEC_SP_LG_spmap',
                                                    'E1_events_160ms_MVSEC_SP_LG_spmap'])
    ap.add_argument('--rgb', default='E0_rgb_spmap')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    taus = np.arange(-60, 61, 1) / 1000.0
    T = load_cam0_to_cam1_transformation()          # x_c1 = T x_c0
    lines = [f'# Offset analysis ({len(args.events)} event runs vs {args.rgb}); poses with error < {GOOD_M * 100:g} cm',
             '', f'Calibration T_c1c0: t = [{", ".join(f"{v * 100:+.2f}" for v in T[:3, 3])}] cm, '
             f'rotation {np.degrees(Rotation.from_matrix(T[:3, :3]).magnitude()):.2f} deg', '']

    plt.figure(figsize=(6, 4))
    runs = [(args.rgb, 0)] + [(e, 1) for e in args.events]
    loaded = {}
    for name, cam in runs:
        q, t, R, C = load_run(name)
        loaded[name] = dict(zip(q, zip(t, R, C)))
        txt, _ = bias(t, R, C, cam)
        sweep = time_sweep(t, C, cam, taus)
        k = np.argmin(sweep)
        lines += [f'## {name} (cam{cam} GT)', f'- bias in camera frame: {txt}',
                  f'- time offset: median error {sweep[taus == 0][0]:.2f} cm at tau = 0, '
                  f'minimum {sweep[k]:.2f} cm at tau = {taus[k] * 1000:+.0f} ms', '']
        plt.plot(taus * 1000, sweep, label=name.replace('_MVSEC_SP_LG_spmap', '').replace('_spmap', ''))
    plt.xlabel('GT evaluated at t_frame + tau [ms]')
    plt.ylabel('median position error [cm]')
    plt.legend(fontsize=7)
    plt.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT / 'time_offset.png', dpi=150)

    # 3. relative pose E0 (cam0) vs E1 (cam1) on the same queries: T_c1c0 = T_c1w * T_wc0
    lines.append('## Relative pose event estimate vs RGB estimate (same query), compared with the calibration')
    rgb = loaded[args.rgb]
    for name in args.events:
        common = sorted(set(rgb) & set(loaded[name]))
        ts, rots = [], []
        for qn in common:
            _, R0, C0 = rgb[qn]
            _, R1, C1 = loaded[name][qn]
            R_c1c0 = R1.inv() * R0
            ts.append(R1.inv().apply(C0 - C1))          # cam0 centre in cam1 frame = t of T_c1c0
            rots.append((Rotation.from_matrix(T[:3, :3]).inv() * R_c1c0).magnitude())
        ts = np.array(ts) * 100
        med = np.median(ts, axis=0)
        lines.append(f'- {name}: n={len(common)}, median t_c1c0 [{", ".join(f"{v:+.2f}" for v in med)}] cm '
                     f'(calibration [{", ".join(f"{v * 100:+.2f}" for v in T[:3, 3])}]), '
                     f'MAD [{", ".join(f"{v:.2f}" for v in np.median(np.abs(ts - med), axis=0))}] cm, '
                     f'median rotation difference to calibration {np.degrees(np.median(rots)):.2f} deg')
    text = '\n'.join(lines)
    (OUT / 'summary.md').write_text(text + '\n')
    print(text)


if __name__ == '__main__':
    main()

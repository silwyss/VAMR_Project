"""E7 diagnosis: why do dark event queries fail against the light map? Separates lighting change from viewpoint gap.

For each event query (40 ms window, MVSEC_SP_LG) three image partners at a *similar viewpoint gap*:
  (a) dark events  <-> dark RGB frame of the same sequence (other time)  -> viewpoint gap only (same light + flicker)
  (b) dark events  <-> light map keyframe (oracle-nearest)               -> lighting change + viewpoint gap
  (c) light events <-> light map keyframe at a matched gap               -> control: viewpoint gap only
Partner frames are chosen so that their gap score (dp / 5 cm + dr / 5 deg) to the query is as close as possible to
that of (b). A match is counted correct if its symmetric epipolar distance under the GT relative pose is < 2 px
(necessary, not sufficient -> an upper bound on correct matches, same for all three).

Usage: python -m src.WP3.exp_illumination  ->  outputs/WP3/dark_analysis/illumination.md
"""
from pathlib import Path

import numpy as np

from src.constants import OUTPUT_DIR, ROOT_DIR
from src.WP3.ei_nexus import EINexus
from src.WP3.events import EventReader, event_count_image, voxel_grid
from src.WP3.geometry import FrameData, VirtualCamera

N_QUERIES, TOL_PX, MIN_DT_S = 40, 2.0, 0.5


def gap_score(Ra, pa, Rb, pb):
    return np.linalg.norm(pa - pb, axis=-1) / 0.05 + np.degrees((Ra.inv() * Rb).magnitude()) / 5.0


def epipolar_correct(K, R_wq, p_wq, R_wi, p_wi, xq, xi):
    """Symmetric epipolar distance < TOL between query (events, cam1 pose) and image (cam0 pose) points."""
    if len(xq) == 0:
        return np.zeros(0, bool)
    R = (R_wq.inv() * R_wi).as_matrix()                 # image -> query camera
    t = R_wq.inv().apply(p_wi - p_wq)
    E = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]]) @ R
    Ki = np.linalg.inv(K)
    F = Ki.T @ E @ Ki                                   # x_q^T F x_i = 0
    hq, hi = np.c_[xq, np.ones(len(xq))], np.c_[xi, np.ones(len(xi))]
    l_q, l_i = hi @ F.T, hq @ F                         # epipolar lines in query / image
    num = np.abs(np.sum(hq * l_q, axis=1))
    d = num / np.hypot(l_q[:, 0], l_q[:, 1]) + num / np.hypot(l_i[:, 0], l_i[:, 1])
    return d / 2 < TOL_PX


def main():
    cam, model = VirtualCamera(), EINexus()
    light = FrameData.load()
    dark = FrameData.load_sequence(ROOT_DIR / 'Dataset_Dark')
    readers = {'light': EventReader(), 'dark': EventReader(ROOT_DIR / 'Dataset_Dark' / 'events.h5')}
    rng = np.random.default_rng(0)

    def run(seq, rd, i, partners):
        ev = cam.events(rd.window_by_time(seq.t_us(i), 40))
        vox, mask = voxel_grid(ev, model.bins, cam.size), event_count_image(ev, cam.size) > 0
        out = []
        for pseq, j in partners:
            img = cam.image(pseq.names[j], images_dir=pseq.images_dir)
            r = model.match(vox, img, mask, cam.image_valid)
            ok = epipolar_correct(cam.K, seq.R1[i], seq.p1[i], pseq.R0[j], pseq.p0[j], r.matched_events, r.matched_image)
            out.append((len(ok), int(ok.sum()), float(gap_score(seq.R1[i], seq.p1[i], pseq.R0[j], pseq.p0[j]))))
        return out

    def usable(seq, rd):
        t = seq.t[seq.query_idx] * 1e6
        return seq.query_idx[(t - 80e3 >= rd.t_start) & (t <= rd.t_end)]

    res = {'a': [], 'b': [], 'c': []}
    qd = usable(dark, readers['dark'])
    for i in qd[np.linspace(0, len(qd) - 1, N_QUERIES).astype(int)]:
        k = light.nearest_keyframes(i, 1, query=dark)[0]
        target = gap_score(dark.R1[i], dark.p1[i], light.R0[k], light.p0[k])
        far = np.abs(dark.t - dark.t[i]) > MIN_DT_S
        cand = np.flatnonzero(far)
        j = cand[np.argmin(np.abs(gap_score(dark.R1[i], dark.p1[i], dark.R0[cand], dark.p0[cand]) - target))]
        (a, b) = run(dark, readers['dark'], i, [(dark, j), (light, k)])
        res['a'].append(a)
        res['b'].append(b)
        # control: a light query and a light keyframe at the same gap score
        ql = usable(light, readers['light'])
        il = ql[rng.integers(len(ql))]
        g = gap_score(light.R1[il], light.p1[il], light.R0[light.key_idx], light.p0[light.key_idx])
        kl = light.key_idx[np.argmin(np.abs(g - target))]
        res['c'].append(run(light, readers['light'], il, [(light, kl)])[0])
        print(f'{len(res["a"])}/{N_QUERIES} gap {target:.1f}: a {a[1]}/{a[0]} (gap {a[2]:.1f}), '
              f'b {b[1]}/{b[0]}, c {res["c"][-1][1]}/{res["c"][-1][0]} (gap {res["c"][-1][2]:.1f})', flush=True)

    names = {'a': 'dark events ↔ dark RGB (other time)', 'b': 'dark events ↔ light map keyframe',
             'c': 'light events ↔ light map keyframe'}
    lines = [f'# Lighting vs viewpoint: {N_QUERIES} queries, 40 ms, epipolar < {TOL_PX:g} px (GT relative pose)', '',
             '| pair | median gap score | median matches | median epipolar-consistent | median fraction |',
             '|---|---|---|---|---|']
    for key, v in res.items():
        v = np.array(v, dtype=float)
        lines.append(f'| ({key}) {names[key]} | {np.median(v[:, 2]):.1f} | {np.median(v[:, 0]):.0f} | '
                     f'{np.median(v[:, 1]):.0f} | {np.median(v[:, 1] / np.maximum(v[:, 0], 1)):.2f} |')
    text = '\n'.join(lines)
    out = OUTPUT_DIR / 'WP3' / 'dark_analysis'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'illumination.md').write_text(text + '\n', encoding='utf-8')
    print(text.encode('ascii', 'replace').decode())


if __name__ == '__main__':
    main()

"""E7: peanuts_dark queries against the peanuts_light map.

The dark sequence follows a different path, so its queries are further from the map keyframes than the interleaved
light queries. To separate "darkness" from "viewpoint gap", dark queries are compared with light queries *at the same
gap*: the light pool is all stride runs of A (strides 1-32), binned by the same gap measure
(gap score = position gap / 5 cm + rotation gap / 5 deg to the nearest available keyframe).

Usage: python -m src.WP3.analyze_dark  ->  outputs/WP3/dark_analysis/summary.md
"""
import sys

import numpy as np

from src.constants import OUTPUT_DIR
from src.WP3.analyze_gap import BASE, RUNS, STRIDES, THR, load, recall

OUT = OUTPUT_DIR / 'WP3' / 'dark_analysis'
PREFIX = 'Dataset_Dark_'
BINS = (0, 1, 2, 4, 8, 16, 1e9)


def score(r):
    return r['map_gap_cm'] / 5 + r['map_gap_deg'] / 5


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ['# E7: peanuts_dark queries vs peanuts_light map (100 queries each)', '',
             '| run | median gap nearest kf | median gap best retrieved | ' + ' | '.join(
                 f'recall {m * 100:g}cm/{d}°' for m, d in THR) + ' | median err (all) | PnP inl | GT-pose inl |',
             '|' + '---|' * 9]
    for p in sorted(RUNS.glob(PREFIX + '*')):
        rows = load(p.name)
        if rows is None:
            continue
        med = lambda k: np.median([float(r[k]) for r in rows])
        err = np.median([r['pos_err_m'] * 100 for r in rows])
        lines.append(f'| {p.name[len(PREFIX):]} | {med("map_gap_cm"):.0f} cm / {med("map_gap_deg"):.0f}° | '
                     f'{med("ret_gap_cm"):.0f} cm / {med("ret_gap_deg"):.0f}° | '
                     + ' | '.join(f'{recall(rows, m, d):.2f}' for m, d in THR)
                     + f' | {err:.1f} cm | {med("num_inliers"):.0f} | {med("gt_inliers"):.0f} |')

    lines += ['', '## Same viewpoint gap: dark vs light (oracle retrieval, recall 5cm/5°)', '',
              'gap score = position gap / 5 cm + rotation gap / 5° to the nearest available keyframe', '',
              '| gap score | light events (n) | dark events 40 ms (n) | light RGB (n) | dark RGB (n) |', '|---|---|---|---|---|']
    pools = {}
    for mod, base in BASE.items():
        pools[f'light {mod}'] = sum((load(base + (f'_stride{s}' if s > 1 else '')) or [] for s in STRIDES), [])
    pools['dark events'] = load(PREFIX + BASE['events']) or []
    pools['dark rgb'] = load(PREFIX + BASE['rgb']) or []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        cells = []
        for k in ('light events', 'dark events', 'light rgb', 'dark rgb'):
            sel = [r for r in pools[k] if lo <= score(r) < hi]
            cells.append(f'{recall(sel, 0.05, 5):.2f} ({len(sel)})' if len(sel) >= 5 else f'- ({len(sel)})')
        lines.append(f'| {lo:g}–{hi:g} | ' + ' | '.join(cells) + ' |')
    text = '\n'.join(lines)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

"""A (viewpoint gap via map stride) and B / E4 (retrieval): tables + plots from the localization runs.

  A  runs E1_events_40ms_..._spmap[_strideK] and E0_rgb_spmap[_strideK]; all queries of all strides are pooled and
     binned by the gap between the query and the nearest *available* keyframe (GT pose) -> recall vs gap.
  B  runs ..._{netvlad,einexus}[_stride4] vs oracle: recall, and how far the best retrieved keyframe is.

Usage: python -m src.WP3.analyze_gap   ->  outputs/WP3/gap_analysis/summary.md, recall_vs_gap.png
"""
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.constants import OUTPUT_DIR

RUNS = OUTPUT_DIR / 'WP3' / 'localization'
OUT = OUTPUT_DIR / 'WP3' / 'gap_analysis'
STRIDES = (1, 2, 4, 8, 16, 32)
THR = ((0.02, 2), (0.05, 5), (0.10, 10))
BASE = {'events': 'E1_events_40ms_MVSEC_SP_LG_spmap', 'rgb': 'E0_rgb_spmap'}


def load(name):
    p = RUNS / name / 'results.csv'
    if not p.is_file():
        return None
    rows = list(csv.DictReader(open(p)))
    for r in rows:
        for k in ('pos_err_m', 'rot_err_deg', 'map_gap_cm', 'map_gap_deg', 'ret_gap_cm', 'ret_gap_deg',
                  'speed_deg_s', 'num_inliers'):
            if k in r:      # older runs have no gap columns
                r[k] = float(r[k])
        r['success'] = r['success'] == 'True'
    return rows


def recall(rows, m, d):
    return np.mean([r['success'] and r['pos_err_m'] < m and r['rot_err_deg'] < d for r in rows])


def row_stats(rows):
    ok = [r for r in rows if r['success']]
    med = lambda k, s=1: np.median([r[k] * s for r in ok]) if ok else np.nan
    return (f'{np.mean([r["success"] for r in rows]):.2f} | ' + ' | '.join(f'{recall(rows, m, d):.2f}' for m, d in THR)
            + f' | {med("pos_err_m", 100):.1f} cm / {med("rot_err_deg"):.2f}° | {np.median([r["num_inliers"] for r in rows]):.0f}')


HEAD = '| success (≥12 inl) | recall 2cm/2° | recall 5cm/5° | recall 10cm/10° | median error (successes) | inliers |'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ['# A: map stride (viewpoint gap), oracle retrieval, 100 queries', '',
             '| modality | stride | median gap to nearest keyframe ' + HEAD, '|' + '---|' * 9]
    pooled = {}
    for mod, base in BASE.items():
        pooled[mod] = []
        for s in STRIDES:
            rows = load(base + (f'_stride{s}' if s > 1 else ''))
            if rows is None:
                continue
            pooled[mod] += rows
            gap = f'{np.median([r["map_gap_cm"] for r in rows]):.1f} cm / {np.median([r["map_gap_deg"] for r in rows]):.1f}°'
            lines.append(f'| {mod} | {s} | {gap} | ' + row_stats(rows) + ' |')

    # pooled recall vs gap (position gap bins; rotation gap is strongly correlated)
    edges = np.array([0, 2, 5, 10, 20, 40, 80, 1000])
    lines += ['', '## Pooled over all strides: recall vs gap to nearest available keyframe', '',
              '| gap bin | n (events / rgb) | events recall 2cm/2° / 5cm/5° | rgb recall 2cm/2° / 5cm/5° |',
              '|---|---|---|---|']
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    for mod, style in (('events', 'o-'), ('rgb', 's--')):
        rows = pooled[mod]
        g = np.array([r['map_gap_cm'] for r in rows])
        for (m, d), c in zip(THR[:2], ('C0', 'C1')):
            xs, ys = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                sel = [r for r, gg in zip(rows, g) if lo <= gg < hi]
                if len(sel) >= 10:
                    xs.append(np.median([r['map_gap_cm'] for r in sel]))
                    ys.append(recall(sel, m, d))
            ax.plot(xs, ys, style, color=c, label=f'{mod}, {m * 100:g} cm / {d}°')
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = {mod: [r for r in pooled[mod] if lo <= r['map_gap_cm'] < hi] for mod in pooled}
        if not sel['events'] and not sel['rgb']:
            continue
        f = lambda rs: ' / '.join(f'{recall(rs, m, d):.2f}' for m, d in THR[:2]) if rs else '-'
        lines.append(f'| {lo}–{hi} cm | {len(sel["events"])} / {len(sel["rgb"])} | {f(sel["events"])} | {f(sel["rgb"])} |')
    ax.set_xscale('log')
    ax.set_xlabel('gap query → nearest available map keyframe [cm] (bin median)')
    ax.set_ylabel('recall')
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / 'recall_vs_gap.png', dpi=150)

    # B: retrieval
    lines += ['', '# B: retrieval (top-3 keyframes), 100 queries', '',
              '| modality | stride | retrieval | median gap best retrieved kf | retrieved kf within 2× oracle gap '
              + HEAD, '|' + '---|' * 11]
    for mod, base in BASE.items():
        for s in (1, 4):
            for ret in ('oracle', 'einexus', 'netvlad'):
                rows = load(base + (f'_stride{s}' if s > 1 else '') + (f'_{ret}' if ret != 'oracle' else ''))
                if rows is None:
                    continue
                gap = f'{np.median([r["ret_gap_cm"] for r in rows]):.1f} cm / {np.median([r["ret_gap_deg"] for r in rows]):.1f}°'
                near = np.mean([r['ret_gap_cm'] <= 2 * r['map_gap_cm'] + 1 for r in rows])
                lines.append(f'| {mod} | {s} | {ret} | {gap} | {near:.2f} | ' + row_stats(rows) + ' |')

    # wrong poses counted as "success": the 12-inlier rule as a confidence measure
    lines += ['', '## "Successes" (≥ 12 inliers) that are wrong by > 10 cm or 10°, per run', '']
    for p in sorted(RUNS.iterdir()):
        rows = load(p.name)
        if rows is None or 'map_gap_cm' not in rows[0]:
            continue
        bad = [r for r in rows if r['success'] and (r['pos_err_m'] > .1 or r['rot_err_deg'] > 10)]
        if bad:
            lines.append(f'- {p.name}: {len(bad)} (inliers of these: '
                         f'{", ".join(str(int(r["num_inliers"])) for r in sorted(bad, key=lambda r: -r["num_inliers"])[:8])})')
    text = '\n'.join(lines)
    (OUT / 'summary.md').write_text(text + '\n', encoding='utf-8')
    sys.stdout.reconfigure(encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()

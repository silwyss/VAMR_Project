"""E2 / E3: matching quality of EI-Nexus vs. event-window length and model variant.

Every query is matched against the RGB frame taken at the same instant, where the correct correspondence is known
(rotation homography between the two beam-splitter cameras, see geometry.VirtualCamera.same_time_homography).
Metrics per (query, config): #events, #matches, #correct matches (< TOL px from H), precision = correct / matches.
This isolates the cross-modal *matching* from viewpoint change, retrieval and PnP (those come in M3+).

  E2a  fixed duration windows      (MVSEC_SP_LG)
  E2b  fixed event-count windows   (MVSEC_SP_LG)
  E3   model variants: LightGlue (stage 2) vs mutual-NN (stage 1), MVSEC vs EC trained event extractor

Outputs: outputs/WP3/exp_matching/results.csv, summary.md, *.png
"""
import csv
import time

import matplotlib.pyplot as plt
import numpy as np

from src.constants import OUTPUT_DIR
from src.WP3.ei_nexus import EINexus
from src.WP3.events import EventReader, event_count_image, voxel_grid
from src.WP3.geometry import FrameData, VirtualCamera, homography_precision

OUT_DIR = OUTPUT_DIR / 'WP3' / 'exp_matching'
NUM_QUERIES = 60
TOL_PX = 5.0
DURATIONS_MS = (2.5, 5, 10, 20, 40, 80)
COUNTS = (25_000, 50_000, 100_000, 200_000, 400_000, 800_000)
VARIANT_DURATIONS_MS = (10, 40)



def build_configs(durations, counts, variant_durations):
    return ([('E2a', 'MVSEC_SP_LG', 'time', d) for d in durations]
            + [('E2b', 'MVSEC_SP_LG', 'count', n) for n in counts]
            + [('E3', v, 'time', d) for v in ('MVSEC_SP_MNN', 'EC_SP_MNN') for d in variant_durations])


CONFIGS = build_configs(DURATIONS_MS, COUNTS, VARIANT_DURATIONS_MS)


def run() -> list[dict]:
    cam = VirtualCamera()
    fd = FrameData.load()
    H = cam.same_time_homography()
    models = {v: EINexus(v) for v in sorted({c[1] for c in CONFIGS})}
    rows = []
    with EventReader() as reader:
        # the event stream starts ~4 ms before the first frame and ends ~0.75 s before the last frame:
        # only queries with a full event history (longest window) are usable
        t_q = fd.t[fd.query_idx] * 1e6
        usable = fd.query_idx[(t_q - max(DURATIONS_MS) * 1e3 >= reader.t_start) & (t_q <= reader.t_end)]
        queries = usable[np.linspace(0, len(usable) - 1, NUM_QUERIES).astype(int)]
        print(f'{len(usable)}/{len(fd.query_idx)} queries inside the event stream')
        for qi, i in enumerate(queries):
            img = cam.image(fd.names[i])
            cache = {}
            for exp, variant, mode, value in CONFIGS:
                key = (mode, value)
                if key not in cache:   # window + representation shared between model variants
                    raw = (reader.window_by_time(fd.t_us(i), value) if mode == 'time'
                           else reader.window_by_count(fd.t_us(i), value))
                    ev = cam.events(raw)
                    cache[key] = (ev, voxel_grid(ev, 16, cam.size), event_count_image(ev, cam.size) > 0)
                ev, vox, mask = cache[key]
                t0 = time.perf_counter()
                res = models[variant].match(vox, img, mask, cam.image_valid)
                dt = time.perf_counter() - t0
                good = homography_precision(H, res.matched_image, res.matched_events, TOL_PX)
                rows.append(dict(exp=exp, variant=variant, mode=mode, value=value, query=fd.names[i],
                                 speed_deg_s=float(fd.speed_deg_s[i]), num_events=len(ev),
                                 duration_ms=ev.duration_ms, num_kpts_events=len(res.kpts_events),
                                 num_matches=len(res.matches), num_correct=int(good.sum()),
                                 precision=float(good.mean()) if len(good) else 0.0, match_time_s=dt))
            print(f'{qi + 1}/{len(queries)} {fd.names[i]} ({fd.speed_deg_s[i]:.0f} deg/s)', flush=True)
    return rows


def summarise(rows: list[dict]) -> str:
    speeds = np.array([r['speed_deg_s'] for r in rows if r['exp'] == 'E2a' and r['value'] == DURATIONS_MS[0]])
    edges = np.percentile(speeds, [100 / 3, 200 / 3])
    groups = (('slow', -np.inf, edges[0]), ('medium', edges[0], edges[1]), ('fast', edges[1], np.inf))
    lines = [f'Speed tertiles (deg/s): slow < {edges[0]:.0f} <= medium < {edges[1]:.0f} <= fast', '',
             '| exp | variant | window | events (median) | duration ms (median) | matches | correct | precision | '
             'correct slow / med / fast | precision slow / med / fast |',
             '|---|---|---|---|---|---|---|---|---|---|']
    for exp, variant, mode, value in CONFIGS:
        sel = [r for r in rows if (r['exp'], r['variant'], r['mode'], r['value']) == (exp, variant, mode, value)]
        med = lambda k, s=sel: np.median([r[k] for r in s])
        by = [[r for r in sel if lo <= r['speed_deg_s'] < hi] for _, lo, hi in groups]
        win = f'{value:g} ms' if mode == 'time' else f'{value // 1000}k ev'
        lines.append(f'| {exp} | {variant} | {win} | {med("num_events"):.0f} | {med("duration_ms"):.1f} | '
                     f'{med("num_matches"):.0f} | {med("num_correct"):.0f} | {med("precision"):.2f} | '
                     + ' / '.join(f'{np.median([r["num_correct"] for r in b]):.0f}' for b in by) + ' | '
                     + ' / '.join(f'{np.median([r["precision"] for r in b]):.2f}' for b in by) + ' |')
    return '\n'.join(lines), groups


def plot(rows: list[dict], groups):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for col, (exp, mode, values, xlabel) in enumerate((('E2a', 'time', DURATIONS_MS, 'window length [ms]'),
                                                      ('E2b', 'count', COUNTS, 'events per window'))):
        for name, lo, hi in groups:
            for row_ax, key in ((0, 'num_correct'), (1, 'precision')):
                y = [np.median([r[key] for r in rows if r['exp'] == exp and r['value'] == v
                                and lo <= r['speed_deg_s'] < hi]) for v in values]
                axes[row_ax, col].plot(values, y, 'o-', label=f'{name} motion')
        for row_ax, ylabel in ((0, 'correct matches (median)'), (1, 'precision (median)')):
            ax = axes[row_ax, col]
            ax.set_xscale('log')
            ax.set(xlabel=xlabel, ylabel=ylabel)
            ax.grid(alpha=0.3)
            ax.legend()
    axes[0, 0].set_title('E2a: fixed-duration windows')
    axes[0, 1].set_title('E2b: fixed event-count windows')
    fig.suptitle(f'EI-Nexus (MVSEC_SP_LG) event window vs same-time RGB frame, correct = within {TOL_PX:g} px of H')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'e2_window_length.png', dpi=120)

    # precision vs speed scatter for the 10 ms / 40 ms windows
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for v in (10, 40):
        sel = [r for r in rows if r['exp'] == 'E2a' and r['value'] == v]
        ax.scatter([r['speed_deg_s'] for r in sel], [r['precision'] for r in sel], s=14, label=f'{v} ms')
    ax.set(xlabel='angular speed [deg/s]', ylabel='precision', title='Precision vs motion speed')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'precision_vs_speed.png', dpi=120)


def main():
    import argparse
    global CONFIGS, DURATIONS_MS, COUNTS, OUT_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument('--durations', type=float, nargs='*', default=DURATIONS_MS)
    ap.add_argument('--counts', type=int, nargs='*', default=COUNTS)
    ap.add_argument('--variant-durations', type=float, nargs='*', default=VARIANT_DURATIONS_MS)
    ap.add_argument('--name', default='exp_matching', help='output folder in outputs/WP3')
    args = ap.parse_args()
    DURATIONS_MS, COUNTS = tuple(args.durations), tuple(args.counts)
    CONFIGS = build_configs(DURATIONS_MS, COUNTS, tuple(args.variant_durations))
    OUT_DIR = OUTPUT_DIR / 'WP3' / args.name
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = run()
    with open(OUT_DIR / 'results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    table, groups = summarise(rows)
    (OUT_DIR / 'summary.md').write_text(table + '\n')
    plot(rows, groups)
    print(table)
    print(f'median match time: {np.median([r["match_time_s"] for r in rows]) * 1e3:.0f} ms')


if __name__ == '__main__':
    main()

"""M1: run EI-Nexus on EDS event windows and RGB frames and visualise the matches.

For a few query frames (slow -> fast motion) it matches the event window ending at the frame timestamp against
  (a) the RGB frame taken at the same time   -> the two cameras share a beam splitter (4.5 mm, 1.6 deg apart), so
      correct matches must follow the rotation homography H = K R_c1c0 K^-1 (up to ~2-3 px parallax): this gives a
      ground-truth precision for the matches, and tells whether the RGB image is mirrored w.r.t. the events.
  (b) the nearest map keyframe (by GT pose) -> the actual relocalization setting; checked with a RANSAC
      fundamental matrix (inlier ratio).
Both modalities are undistorted into one virtual pinhole camera (the event camera's K) so they have the same scale.

Outputs: outputs/WP3/m1/*.png and a table on stdout.
"""
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.constants import OUTPUT_DIR
from src.WP3.ei_nexus import EINexus
from src.WP3.events import EventReader, event_color_image, event_count_image, voxel_grid
from src.WP3.geometry import FrameData, VirtualCamera, homography_precision

OUT_DIR = OUTPUT_DIR / 'WP3' / 'm1'
WINDOW_MS = 40.0
HOMOGRAPHY_TOL_PX = 5.0


def draw_matches(ax, left, right, pts_l, pts_r, good=None, title=''):
    w = left.shape[1]
    canvas = np.hstack([left if left.ndim == 3 else np.dstack([left] * 3), np.dstack([right] * 3)])
    ax.imshow(canvas)
    good = np.ones(len(pts_l), bool) if good is None else good
    for (xl, yl), (xr, yr), g in zip(pts_l, pts_r, good):
        ax.plot([xl, xr + w], [yl, yr], '-', color='lime' if g else 'red', lw=0.6, alpha=0.8)
    ax.set_title(title, fontsize=9)
    ax.axis('off')


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cam = VirtualCamera()
    fd = FrameData.load()
    model = EINexus('MVSEC_SP_LG')
    H_same = cam.same_time_homography()

    q = fd.query_idx
    picks = [q[np.argmin(np.abs(fd.speed_deg_s[q] - np.percentile(fd.speed_deg_s[q], pc)))] for pc in (10, 50, 90)]

    rows = []
    with EventReader() as reader:
        for i in picks:
            name, speed = fd.names[i], fd.speed_deg_s[i]
            ev = cam.events(reader.window_by_time(fd.t_us(i), WINDOW_MS))
            vox = voxel_grid(ev, model.bins, cam.size)
            ev_mask = event_count_image(ev, cam.size) > 0
            ev_vis = event_color_image(ev, cam.size)
            k = fd.nearest_keyframe(i)

            fig, axes = plt.subplots(3, 1, figsize=(13, 15))
            # (a) same-time RGB frame, normal and mirrored
            for ax, flip in zip(axes[:2], (False, True)):
                img = cam.image(name, flip)
                res = model.match(vox, img, ev_mask, cam.image_valid)
                good = homography_precision(H_same, res.matched_image, res.matched_events, HOMOGRAPHY_TOL_PX)
                prec = good.mean() if len(good) else 0.0
                rows.append((name, f'{speed:.0f}', len(ev), f'same-time{" flipped" if flip else ""}',
                             len(res.matches), f'{prec:.2f}'))
                draw_matches(ax, ev_vis, img, res.matched_events, res.matched_image, good,
                             f'{name} events ({len(ev)} ev, {WINDOW_MS:g} ms, {speed:.0f} deg/s) vs same-time RGB'
                             f'{" (mirrored)" if flip else ""}: {len(res.matches)} matches, '
                             f'{prec:.0%} consistent with H (<{HOMOGRAPHY_TOL_PX:g} px)')
            # (b) nearest map keyframe
            img = cam.image(fd.names[k])
            res = model.match(vox, img, ev_mask, cam.image_valid)
            inl = np.zeros(len(res.matches), bool)
            if len(res.matches) >= 8:
                _, mask = cv2.findFundamentalMat(res.matched_events, res.matched_image, cv2.USAC_MAGSAC, 2.0, 0.999)
                inl = mask.ravel().astype(bool) if mask is not None else inl
            dt = abs(fd.t[k] - fd.t[i]) * 1e3
            rows.append((name, f'{speed:.0f}', len(ev), f'keyframe {fd.names[k]} ({dt:.0f} ms)',
                         len(res.matches), f'{inl.mean() if len(inl) else 0:.2f} (F inl.)'))
            draw_matches(axes[2], ev_vis, img, res.matched_events, res.matched_image, inl,
                         f'vs nearest map keyframe {fd.names[k]} ({dt:.0f} ms apart): {len(res.matches)} matches, '
                         f'{inl.sum()} fundamental-matrix inliers')
            fig.tight_layout()
            fig.savefig(OUT_DIR / f'{name[:-4]}.png', dpi=110)
            plt.close(fig)

    print(f'{"query":22s} {"deg/s":>6s} {"events":>8s}  {"pair":38s} {"matches":>7s}  precision')
    for r in rows:
        print(f'{r[0]:22s} {r[1]:>6s} {r[2]:>8d}  {r[3]:38s} {r[4]:>7d}  {r[5]}')
    print(f'figures in {OUT_DIR}')


if __name__ == '__main__':
    main()

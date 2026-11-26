"""Event loading, windowing and representations for WP3.

events.h5 (EDS) stores flat arrays t [us, int64], x, y [uint16], p [uint8 in {0, 1}] with ~341M events, so it is
never loaded completely: windows are located by binary search directly on the (sorted) h5 timestamp dataset.
"""
from dataclasses import dataclass

import cv2
import h5py
import numpy as np
import torch

from src.constants import DATASET_DIR


@dataclass
class EventWindow:
    x: np.ndarray   # [N] float32 pixel coordinates (raw or undistorted)
    y: np.ndarray   # [N] float32
    t: np.ndarray   # [N] int64 timestamps [us]
    p: np.ndarray   # [N] int8 polarity in {0, 1}

    def __len__(self):
        return len(self.t)

    @property
    def duration_ms(self) -> float:
        return float(self.t[-1] - self.t[0]) / 1e3 if len(self) > 1 else 0.0


class EventReader:
    """Random access to event windows of an EDS events.h5 file."""

    def __init__(self, path=DATASET_DIR / 'events.h5'):
        self.file = h5py.File(path, 'r')
        self._t, self._x, self._y, self._p = (self.file[k] for k in ('t', 'x', 'y', 'p'))
        self.num_events = self._t.shape[0]
        self.t_start, self.t_end = int(self._t[0]), int(self._t[-1])

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def index_of(self, t_us: int) -> int:
        """First event index with t >= t_us (like np.searchsorted(t, t_us, 'left')) via bisection on disk."""
        lo, hi = 0, self.num_events
        while hi - lo > 4096:                       # coarse: single-element reads
            mid = (lo + hi) // 2
            if self._t[mid] < t_us:
                lo = mid + 1
            else:
                hi = mid
        return lo + int(np.searchsorted(self._t[lo:hi], t_us, side='left'))  # fine: one small block

    def slice(self, i0: int, i1: int) -> EventWindow:
        i0, i1 = max(i0, 0), min(i1, self.num_events)
        return EventWindow(self._x[i0:i1].astype(np.float32), self._y[i0:i1].astype(np.float32),
                           self._t[i0:i1].astype(np.int64), self._p[i0:i1].astype(np.int8))

    def window_by_time(self, t_end_us: int, duration_ms: float) -> EventWindow:
        """All events in [t_end - duration, t_end)."""
        return self.slice(self.index_of(int(t_end_us - duration_ms * 1e3)), self.index_of(int(t_end_us)))

    def window_by_count(self, t_end_us: int, num_events: int) -> EventWindow:
        """The last `num_events` events before t_end."""
        i1 = self.index_of(int(t_end_us))
        return self.slice(i1 - num_events, i1)

    def window(self, t_end_us: int, duration_ms: float | None = None, num_events: int | None = None,
               flicker_hz: float | None = None) -> EventWindow:
        """Time window (duration_ms) or count window (num_events) ending at t_end, optionally with flicker removal
        (see flicker_filter; reads one flicker period of extra history so that the filter stays causal)."""
        if flicker_hz is None:
            return (self.window_by_time(t_end_us, duration_ms) if num_events is None
                    else self.window_by_count(t_end_us, num_events))
        period, tol = 1e6 / flicker_hz, 1e3
        if num_events is None:
            t_min = int(t_end_us - duration_ms * 1e3)
            return flicker_filter(self.window_by_time(t_end_us, (t_end_us - t_min + period + tol) / 1e3), t_min,
                                  period, tol)
        n = 2 * num_events
        while True:                                  # count window: grow the history until enough events survive
            raw = self.window_by_count(t_end_us, n)
            out = flicker_filter(raw, int(raw.t[0] + period + tol) if len(raw) else 0, period, tol)
            if len(out) >= num_events or len(raw) < n:
                return EventWindow(*(a[-num_events:] for a in (out.x, out.y, out.t, out.p)))
            n *= 2


def flicker_phase_gate(ev: EventWindow, freq_hz: float = 100.0, keep_fraction: float = 0.5) -> EventWindow:
    """Drop the flicker-dominated half of each polarity's cycle.

    Under a flickering lamp ON events bunch around the brightening phase and OFF events around the dimming phase of
    every cycle (period 1/freq). The peak phase of each polarity is estimated from the window itself (circular mean
    of the event phases); events of that polarity within (1 - keep_fraction)/2 cycles of their peak are dropped.
    Motion events are spread uniformly over the phase, so they lose (1 - keep_fraction) of their events."""
    if len(ev) == 0:
        return ev
    phase = ((ev.t - ev.t[0]) * (freq_hz / 1e6)) % 1.0
    keep = np.ones(len(ev), bool)
    for pol in (0, 1):
        sel = ev.p == pol
        if sel.sum() < 100:
            continue
        peak = np.angle(np.exp(2j * np.pi * phase[sel]).mean()) / (2 * np.pi) % 1.0
        dist = np.abs((phase[sel] - peak + 0.5) % 1.0 - 0.5)
        keep[np.flatnonzero(sel)[dist < (1 - keep_fraction) / 2]] = False
    return EventWindow(ev.x[keep], ev.y[keep], ev.t[keep], ev.p[keep])


def flicker_filter(ev: EventWindow, t_min: int, period_us: float = 1e4, tol_us: float = 1e3,
                   width: int = 1280) -> EventWindow:
    """Causal comb filter against light flicker (e.g. 100 Hz from 50 Hz mains lamps).

    Flicker makes every lit pixel fire the same polarity once per flicker period, independent of motion; a moving
    edge passes a pixel once. An event is dropped if the same pixel fired with the same polarity one period earlier
    (t - period +- tol). Returns the surviving events with t >= t_min (events before t_min only serve as history).
    Needs raw (integer) sensor coordinates, i.e. apply before undistortion. A simplified event-level version of the
    comb filtering idea of Wang et al., "Linear comb filter for event flicker removal", ICRA 2022."""
    if len(ev) == 0:
        return ev
    key = (ev.y.astype(np.int64) * width + ev.x.astype(np.int64)) * 2 + ev.p.astype(np.int64)
    order = np.lexsort((ev.t, key))                  # sorted by pixel/polarity, then time
    code = key[order] * (1 << 40) + (ev.t[order] - ev.t.min())   # relative t < 2^40 us (~12 days)
    lo = np.searchsorted(code, code - int(period_us + tol_us), 'left')
    hi = np.searchsorted(code, code - int(period_us - tol_us), 'right')
    periodic = np.empty(len(ev), bool)
    periodic[order] = hi > lo
    keep = ~periodic & (ev.t >= t_min)
    return EventWindow(ev.x[keep], ev.y[keep], ev.t[keep], ev.p[keep])


# --------------------------------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------------------------------

def undistort_events(ev: EventWindow, K: np.ndarray, D: np.ndarray, K_new: np.ndarray, size: tuple[int, int]) -> EventWindow:
    """Map event coordinates through the radtan undistortion into the pinhole camera K_new (size = (W, H)).
    Events that fall outside the new image are dropped."""
    if len(ev) == 0:
        return ev
    pts = np.stack([ev.x, ev.y], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, D, P=K_new).reshape(-1, 2)
    keep = (und[:, 0] >= 0) & (und[:, 0] <= size[0] - 1) & (und[:, 1] >= 0) & (und[:, 1] <= size[1] - 1)
    return EventWindow(und[keep, 0].astype(np.float32), und[keep, 1].astype(np.float32), ev.t[keep], ev.p[keep])


# --------------------------------------------------------------------------------------------------
# Representations
# --------------------------------------------------------------------------------------------------

@torch.no_grad()
def voxel_grid(ev: EventWindow, bins: int, size: tuple[int, int], normalize: bool = True,
               rank_time: bool = False) -> torch.Tensor:
    """[bins, H, W] voxel grid with bilinear interpolation in x, y and t, polarity as +-1, then normalised to
    zero mean / unit std over the non-zero cells. Same definition as EI-Nexus' `events_to_voxel_grid`
    (datasets/representations.py), vectorised and with sub-pixel (undistorted) coordinates.
    rank_time: bin by event index instead of timestamp (every bin gets the same number of events), which removes
    rate modulations such as 100 Hz flicker from the temporal axis."""
    W, H = size
    grid = torch.zeros(bins * H * W, dtype=torch.float32)
    if len(ev) < 2:
        return grid.view(bins, H, W)
    x, y = torch.from_numpy(ev.x), torch.from_numpy(ev.y)
    t = torch.from_numpy(np.arange(len(ev), dtype=np.float64) if rank_time else (ev.t - ev.t[0]).astype(np.float64))
    t = ((bins - 1) * t / max(float(t[-1]), 1e-9)).float()
    val = torch.from_numpy(np.where(ev.p > 0, 1.0, -1.0).astype(np.float32))
    x0, y0, t0 = x.floor(), y.floor(), t.floor()
    for xi in (x0, x0 + 1):
        for yi in (y0, y0 + 1):
            for ti in (t0, t0 + 1):
                w = val * (1 - (xi - x).abs()) * (1 - (yi - y).abs()) * (1 - (ti - t).abs())
                ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H) & (ti >= 0) & (ti < bins)
                idx = (ti * H * W + yi * W + xi).long()
                grid.index_add_(0, idx[ok], w[ok])
    grid = grid.view(bins, H, W)
    if normalize:
        nz = grid != 0
        if nz.any():
            m, s = grid[nz].mean(), grid[nz].std()
            grid[nz] = (grid[nz] - m) / s if s > 0 else grid[nz] - m
    return grid


def event_count_image(ev: EventWindow, size: tuple[int, int], signed: bool = False) -> np.ndarray:
    """[H, W] float32 number of events per pixel (signed: +1 / -1 per polarity)."""
    W, H = size
    img = np.zeros((H, W), np.float32)
    xi = np.clip(np.round(ev.x).astype(np.int64), 0, W - 1)
    yi = np.clip(np.round(ev.y).astype(np.int64), 0, H - 1)
    np.add.at(img, (yi, xi), np.where(ev.p > 0, 1.0, -1.0) if signed else 1.0)
    return img


def event_color_image(ev: EventWindow, size: tuple[int, int]) -> np.ndarray:
    """[H, W, 3] uint8 visualisation: positive events blue, negative red, on white."""
    W, H = size
    pos = event_count_image(EventWindow(ev.x[ev.p > 0], ev.y[ev.p > 0], ev.t[ev.p > 0], ev.p[ev.p > 0]), size)
    neg = event_count_image(EventWindow(ev.x[ev.p == 0], ev.y[ev.p == 0], ev.t[ev.p == 0], ev.p[ev.p == 0]), size)
    img = np.full((H, W, 3), 255, np.uint8)
    img[pos > neg] = (40, 80, 255)
    img[neg > pos] = (255, 60, 40)
    return img

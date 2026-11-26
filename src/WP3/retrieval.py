"""Image retrieval for WP3 (E4): which map keyframes does the query see? Replaces the oracle (nearest by GT pose).

Methods (all return the k best of a candidate set of map keyframes):
  netvlad   global descriptor (hloc's VGG16-NetVLAD, Pitts30K) of the query vs. each keyframe image. RGB queries use
            the RGB frame; event queries use an event-count image (edges bright on black) -> a modality gap NetVLAD was
            never trained for. This is the "standard pipeline" baseline.
  einexus   voting with EI-Nexus' local descriptors: the query's keypoint descriptors (event extractor for events,
            image extractor for RGB) are compared with the SuperPoint descriptors of every keyframe (from the sp map's
            features.h5); score = number of mutual nearest neighbours. Works across modalities because EI-Nexus'
            event descriptors are distilled into SuperPoint's descriptor space.
"""
import h5py
import numpy as np
import torch

from src.constants import OUTPUT_DIR
from src.WP3.geometry import FrameData, VirtualCamera

CACHE_DIR = OUTPUT_DIR / 'WP3' / 'retrieval'


def event_image(count: np.ndarray) -> np.ndarray:
    """Event-count image -> uint8 grayscale (clipped at the 99th percentile of non-zero pixels)."""
    nz = count[count > 0]
    hi = np.percentile(nz, 99) if len(nz) else 1.0
    return (np.clip(count / hi, 0, 1) * 255).astype(np.uint8)


class NetVLADRetrieval:
    def __init__(self, cam: VirtualCamera, fd: FrameData, device='cuda'):
        from hloc.extractors.netvlad import NetVLAD
        self.device = device
        self.net = NetVLAD(NetVLAD.default_conf).eval().to(device)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache = CACHE_DIR / 'netvlad_keyframes.npz'
        names = fd.names[fd.key_idx]
        if cache.is_file() and list(np.load(cache)['names']) == list(names):
            desc = np.load(cache)['desc']
        else:
            desc = np.stack([self.describe(cam.image(n)) for n in names])
            np.savez(cache, names=names, desc=desc)
        self.key_desc = dict(zip(fd.key_idx, desc))

    @torch.no_grad()
    def describe(self, gray: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(gray.astype(np.float32) / 255.0)[None, None].repeat(1, 3, 1, 1).to(self.device)
        return self.net({'image': x})['global_descriptor'][0].cpu().numpy()

    def topk(self, query_gray: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
        q = self.describe(query_gray)
        sim = np.stack([self.key_desc[c] for c in candidates]) @ q
        return candidates[np.argsort(-sim)[:k]]


class EINexusRetrieval:
    def __init__(self, fd: FrameData, features_h5, device='cuda'):
        self.device = device
        self.key_desc = {}
        with h5py.File(features_h5, 'r') as f:
            for i in fd.key_idx:
                d = f[fd.names[i]]['descriptors'][()].T        # stored [D, N] (hloc format)
                self.key_desc[i] = torch.from_numpy(d).to(device)

    @torch.no_grad()
    def scores(self, query_desc: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        q = torch.from_numpy(query_desc).to(self.device)                      # [N, D], L2-normalised
        out = np.zeros(len(candidates))
        for j, c in enumerate(candidates):
            sim = q @ self.key_desc[c].T                                      # [N, M]
            nn01, nn10 = sim.argmax(1), sim.argmax(0)
            out[j] = int((nn10[nn01] == torch.arange(len(q), device=self.device)).sum())
        return out

    def topk(self, query_desc: np.ndarray, candidates: np.ndarray, k: int) -> np.ndarray:
        return candidates[np.argsort(-self.scores(query_desc, candidates), kind='stable')[:k]]

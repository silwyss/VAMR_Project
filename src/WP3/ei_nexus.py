"""Inference wrapper around EI-Nexus (third_party/EI-Nexus, git submodule).

Only the SuperPoint-based variants are supported: the SiLK image extractor pulls in JAX / pytorch-lightning, so its
module is replaced by a stub before EI-Nexus is imported (the submodule itself stays unmodified).

Variants (checkpoints in third_party/checkpoints, see docs/WP3_plan.md):
    MVSEC_SP_LG   event extractor + LightGlue matcher trained on MVSEC (stage 2)   <- default
    MVSEC_SP_MNN  event extractor trained on MVSEC, mutual nearest neighbour matching (stage 1)
    EC_SP_MNN     event extractor trained on EC, mutual nearest neighbour matching (stage 1)
"""
import sys
import types
from dataclasses import dataclass

import numpy as np
import torch

from src.constants import ROOT_DIR

EI_NEXUS_DIR = ROOT_DIR / 'third_party' / 'EI-Nexus'
CHECKPOINT_DIR = ROOT_DIR / 'third_party' / 'checkpoints'

VARIANTS = {
    'MVSEC_SP_LG': ('configs/model/test/EI_SP_LG.yaml', 'MVSEC_EI_SP_LG_ft_Stage2.pth'),
    'MVSEC_SP_MNN': ('configs/model/test/EI_SP_MNN.yaml', 'MVSEC_EI_SP_MNN_Stage1.pth'),
    'EC_SP_MNN': ('configs/model/test/EI_SP_MNN.yaml', 'EC_EI_SP_MNN_Stage1.pth'),
}
CHECKPOINT_URL = 'https://github.com/ZhonghuaYi/EI-Nexus_official/releases/download/v1.0%2C0/'


def _import_eim():
    if not (EI_NEXUS_DIR / 'core').is_dir():
        raise FileNotFoundError(f'{EI_NEXUS_DIR} is empty - run `git submodule update --init`.')
    if str(EI_NEXUS_DIR) not in sys.path:
        sys.path.insert(0, str(EI_NEXUS_DIR))
    stub = types.ModuleType('core.modules.image_extractors.silk_extractor')
    stub.SiLKModel = None
    sys.modules.setdefault('core.modules.image_extractors.silk_extractor', stub)
    from core.modules.EIM import EIM
    return EIM


@dataclass
class MatchResult:
    kpts_events: np.ndarray   # [N, 2] xy of all event keypoints
    kpts_image: np.ndarray    # [M, 2] xy of all image keypoints
    matches: np.ndarray       # [K, 2] index pairs (event kpt, image kpt)
    scores: np.ndarray        # [K] matching confidence

    @property
    def matched_events(self) -> np.ndarray:
        return self.kpts_events[self.matches[:, 0]]

    @property
    def matched_image(self) -> np.ndarray:
        return self.kpts_image[self.matches[:, 1]]


class EINexus:
    def __init__(self, variant: str = 'MVSEC_SP_LG', device: str = 'cuda'):
        from omegaconf import OmegaConf
        EIM = _import_eim()
        cfg_path, ckpt_name = VARIANTS[variant]
        ckpt = CHECKPOINT_DIR / ckpt_name
        if not ckpt.is_file():
            print(f'downloading {ckpt_name} ...')
            ckpt.parent.mkdir(parents=True, exist_ok=True)
            torch.hub.download_url_to_file(CHECKPOINT_URL + ckpt_name, str(ckpt))
        cfg = OmegaConf.load(EI_NEXUS_DIR / cfg_path)
        cfg.pretrain_stage1.model_path = None
        cfg.pretrain_stage2.model_path = None
        self.device = device
        self.variant = variant
        self.model = EIM(cfg, device)
        state = torch.load(ckpt, map_location=device, weights_only=False)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        # the image extractor (SuperPoint) keeps its own pretrained weights if absent from the checkpoint
        missing = [k for k in missing if not k.startswith('image_extractor')]
        if missing or unexpected:
            raise RuntimeError(f'checkpoint mismatch: missing={missing[:5]}, unexpected={unexpected[:5]}')
        self.model.eval()
        self.bins = cfg.event_extractor[cfg.event_extractor.type].in_channels

    @torch.no_grad()
    def extract_image(self, image: np.ndarray, image_mask: np.ndarray | None = None):
        """EI-Nexus' (frozen SuperPoint) image keypoints exactly as used inside match():
        returns xy [N, 2], descriptors [N, D], scores [N]. Deterministic for the same image + mask."""
        img = torch.from_numpy(image.astype(np.float32))[None, None].to(self.device)
        im = torch.from_numpy(image_mask)[None, None].to(self.device) if image_mask is not None else None
        f = self.model.image_extractor(img, im)
        pos = f['sparse_positions'][0].cpu().numpy()
        return pos[:, 1::-1].copy(), f['sparse_descriptors'][0].cpu().numpy(), pos[:, 2].copy()

    @torch.no_grad()
    def extract_events(self, voxel: torch.Tensor, event_mask: np.ndarray | None = None):
        """EI-Nexus' event keypoints exactly as used inside match(): xy [N, 2], descriptors [N, D], scores [N]."""
        ev = voxel[None].to(self.device)
        em = torch.from_numpy(event_mask)[None, None].to(self.device) if event_mask is not None else None
        f = self.model.event_extractor(ev, em)
        pos = f['sparse_positions'][0].cpu().numpy()
        return pos[:, 1::-1].copy(), f['sparse_descriptors'][0].cpu().numpy(), pos[:, 2].copy()

    @torch.no_grad()
    def match(self, voxel: torch.Tensor, image: np.ndarray, event_mask: np.ndarray | None = None,
              image_mask: np.ndarray | None = None) -> MatchResult:
        """voxel [bins, H, W] (see events.voxel_grid), image [H, W] uint8 grayscale.
        Masks [H, W] bool restrict where keypoints may be detected (pixels with events / valid image area)."""
        ev = voxel[None].to(self.device)
        img = torch.from_numpy(image.astype(np.float32))[None, None].to(self.device)  # 0..255, EIM divides by 255
        em = torch.from_numpy(event_mask)[None, None].to(self.device) if event_mask is not None else None
        im = torch.from_numpy(image_mask)[None, None].to(self.device) if image_mask is not None else None
        e_feats, i_feats, out = self.model(ev, img, em, im)

        def xy(pos):  # EI-Nexus positions are (y, x, score)
            p = pos[0][:, :2].cpu().numpy()
            return p[:, ::-1].copy()

        kpts_e, kpts_i = xy(e_feats['sparse_positions']), xy(i_feats['sparse_positions'])
        m0 = out['matches0'][0].reshape(-1).cpu().numpy()
        s0 = out['matching_scores0'][0].reshape(-1).cpu().numpy()
        valid = m0 > -1
        matches = np.stack([np.flatnonzero(valid), m0[valid]], axis=1).astype(np.int64)
        return MatchResult(kpts_e, kpts_i, matches, s0[valid])

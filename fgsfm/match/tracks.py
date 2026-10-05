"""Tracks from pairwise matches (union-find) and multi-view triangulation from initial poses."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TrackSet:
    obs_img: list[np.ndarray]      # per track: image indices
    obs_feat: list[np.ndarray]     # per track: feature indices in that image
    xyz: np.ndarray                # (T, 3)

    def __len__(self) -> int:
        return len(self.obs_img)


def build_tracks(matches: dict[tuple[int, int], np.ndarray], num_feats: list[int], min_len: int = 2) -> TrackSet:
    """Union-find over (image, feature) nodes; tracks with two features in one image are dropped."""
    offset = np.concatenate([[0], np.cumsum(num_feats)])
    parent = np.arange(offset[-1])

    def find(x):
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for (i, j), m in matches.items():
        for a, b in zip(offset[i] + m[:, 0], offset[j] + m[:, 1]):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    used = np.unique(np.concatenate([np.concatenate([offset[i] + m[:, 0], offset[j] + m[:, 1]])
                                     for (i, j), m in matches.items()]))
    roots = np.array([find(x) for x in used])
    img_of = np.searchsorted(offset, used, side="right") - 1
    feat_of = used - offset[img_of]
    order = np.argsort(roots, kind="stable")
    roots, img_of, feat_of = roots[order], img_of[order], feat_of[order]
    bounds = np.flatnonzero(np.diff(roots)) + 1
    obs_img, obs_feat = [], []
    for imgs, fts in zip(np.split(img_of, bounds), np.split(feat_of, bounds)):
        if len(imgs) < min_len or len(np.unique(imgs)) != len(imgs):
            continue
        obs_img.append(imgs)
        obs_feat.append(fts)
    return TrackSet(obs_img, obs_feat, np.zeros((len(obs_img), 3)))


def triangulate(tracks: TrackSet, feats: list[dict], w2c: np.ndarray, K: np.ndarray,
                max_reproj_px: float, min_angle_deg: float = 1.0) -> np.ndarray:
    """Linear multi-view DLT per track; returns a keep mask (depth > 0, reprojection, triangulation angle)."""
    keep = np.zeros(len(tracks), bool)
    P = K[None] @ w2c[:, :3, :]                              # (N, 3, 4)
    C = -np.einsum("nji,nj->ni", w2c[:, :3, :3], w2c[:, :3, 3])
    for t, (imgs, fts) in enumerate(zip(tracks.obs_img, tracks.obs_feat)):
        uv = np.stack([feats[i]["xy"][f] for i, f in zip(imgs, fts)]).astype(np.float64)
        Ps = P[imgs]
        A = np.concatenate([uv[:, :1] * Ps[:, 2] - Ps[:, 0], uv[:, 1:] * Ps[:, 2] - Ps[:, 1]])
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1, :3] / Vt[-1, 3]
        x = np.einsum("nij,j->ni", Ps, np.append(X, 1.0))
        if (x[:, 2] <= 0).any():
            continue
        err = np.linalg.norm(x[:, :2] / x[:, 2:] - uv, axis=1)
        rays = X - C[imgs]
        rays /= np.linalg.norm(rays, axis=1, keepdims=True)
        cosang = rays @ rays.T
        angle = np.degrees(np.arccos(np.clip(cosang.min(), -1, 1)))
        if err.max() <= max_reproj_px and angle >= min_angle_deg:
            tracks.xyz[t] = X
            keep[t] = True
    return keep


def subset(tracks: TrackSet, mask: np.ndarray) -> TrackSet:
    idx = np.flatnonzero(mask)
    return TrackSet([tracks.obs_img[i] for i in idx], [tracks.obs_feat[i] for i in idx], tracks.xyz[idx])

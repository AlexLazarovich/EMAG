"""Interpolate sparse LD EEG to a full HD channel layout (Ablation 4.3).

Uses inverse-distance weighting in 3D electrode space. LD values are placed at
their true electrode positions; each HD channel receives a weighted blend of
nearby LD channels.
"""
from __future__ import annotations

import numpy as np


def interpolate_ld_to_hd(
    x_ld: np.ndarray | torch.Tensor,
    ld_indices: np.ndarray,
    pos_hd_mm: np.ndarray,
    eps_mm: float = 1.0,
) -> np.ndarray | torch.Tensor:
    """
    Map LD channels to an HD-sized layout (n_hd, T).

    Args:
        x_ld: (n_ld, T) signal at LD electrodes.
        ld_indices: (n_ld,) integer indices into HD montage order [0, n_hd).
        pos_hd_mm: (n_hd, 3) positions of all HD electrodes in mm.
        eps_mm: regularizer for IDW distances.

    Returns:
        (n_hd, T) float32 array or tensor matching input type.
    """
    return_numpy = isinstance(x_ld, np.ndarray)
    if return_numpy:
        ld = np.asarray(x_ld, dtype=np.float64)
        device = None
        dtype = None
    else:
        import torch

        ld = x_ld.detach().float().cpu().numpy()
        device = x_ld.device
        dtype = x_ld.dtype

    n_ld, T = ld.shape
    n_hd = pos_hd_mm.shape[0]
    ld_idx = np.asarray(ld_indices, dtype=np.int64).reshape(-1)
    if ld_idx.shape[0] != n_ld:
        raise ValueError(f"ld_indices length {ld_idx.shape[0]} != n_ld {n_ld}")

    pos_ld = pos_hd_mm[ld_idx].astype(np.float64)  # (n_ld, 3)
    pos_all = pos_hd_mm.astype(np.float64)  # (n_hd, 3)

    # Pairwise HD -> LD distances (n_hd, n_ld)
    diff = pos_all[:, None, :] - pos_ld[None, :, :]
    dist = np.linalg.norm(diff, axis=-1) + float(eps_mm)
    w = 1.0 / dist
    w /= np.sum(w, axis=1, keepdims=True) + 1e-12

    out = np.zeros((n_hd, T), dtype=np.float64)
    for j in range(n_hd):
        out[j] = w[j] @ ld

    out = out.astype(np.float32)
    if return_numpy:
        return out
    return torch.from_numpy(out).to(device=device, dtype=dtype)

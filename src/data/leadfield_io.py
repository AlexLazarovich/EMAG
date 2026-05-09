"""Load and validate precomputed leadfield matrices (Ablation 5.3)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def load_leadfield_matrix(
    path: str | Path,
    n_electrodes: int,
    n_sources: int,
    device: str | torch.device,
) -> torch.Tensor:
    """
    Load a leadfield matrix from .npy or .npz.

    Accepts:
      - (M, N) array
      - .npz with key 'L' or first array

    Raises:
        FileNotFoundError, ValueError on shape mismatch.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Leadfield file not found: {path}")

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=False)
        if "L" in data:
            L = np.asarray(data["L"], dtype=np.float32)
        else:
            keys = list(data.keys())
            if not keys:
                raise ValueError(f"Empty npz: {path}")
            L = np.asarray(data[keys[0]], dtype=np.float32)
    else:
        L = np.load(path, allow_pickle=False)
        L = np.asarray(L, dtype=np.float32)

    if L.ndim != 2:
        raise ValueError(f"Leadfield must be 2D, got shape {L.shape}")
    M, N = L.shape
    if M != n_electrodes:
        raise ValueError(f"Leadfield rows M={M} != n_electrodes={n_electrodes}")
    if N != n_sources:
        raise ValueError(f"Leadfield cols N={N} != n_sources={n_sources}")
    return torch.from_numpy(L.copy()).float().to(device)

"""Pre-render a trained EMAG checkpoint into a JSON file consumed by site/viewer.html.

Dumps:
    Gaussians:        per-Gaussian (xyz, sigma_xyz, mu_t, sigma_t, amp)
    Electrodes (2D):  azimuthal-equidistant projection of HD electrode positions
    HD truth:         actual HD signal of one chosen epoch
    HD prediction:    EMAG forward pass over the same epoch
    LD input:         the LD channels seen by the model

Usage:
    PYTHONPATH=src python scripts/build_site_data.py \
        --checkpoint /path/to/ckpt.pt \
        --dataset localize_mi --subject sub-01 --sr_factor 4 \
        --data_root /path/to/Localize-MI/derivatives/epochs \
        --out site/data/<id>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import inspect

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from models.model import EMAG  # noqa: E402
from models.base import BrainGrid  # noqa: E402


def _spatial_temporal_sigma(chol_log_diag: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    cd = chol_log_diag.detach().cpu()
    if cd.ndim == 1:
        cd = cd.unsqueeze(-1)
    sigmas = (1.0 / torch.exp(cd)).numpy()
    if sigmas.shape[-1] == 1:
        sigmas = np.broadcast_to(sigmas, (cd.shape[0], 4)).copy()
    elif sigmas.shape[-1] == 3:
        st = np.full((cd.shape[0], 1), sigmas[:, :3].mean())
        sigmas = np.concatenate([sigmas, st], axis=-1)
    return sigmas[:, :3], sigmas[:, 3]


def _chol_to_spatial_cov(log_diag: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Cholesky of 4x4 precision -> 3x3 spatial covariance.

    Mirrors paper/media/make_ellipsoids.py: builds L (lower-triangular 4x4),
    P = L L^T, Sigma = P^-1, returns Sigma[:, :3, :3].
    """
    N = log_diag.shape[0]
    D = log_diag.shape[1]
    L = np.zeros((N, 4, 4), dtype=np.float64)
    diag = np.exp(log_diag)
    for d in range(min(D, 4)):
        L[:, d, d] = diag[:, d]
    if D == 1:
        for d in range(1, 4):
            L[:, d, d] = diag[:, 0]
    if off.shape[1] >= 6:
        L[:, 1, 0] = off[:, 0]
        L[:, 2, 0] = off[:, 1]; L[:, 2, 1] = off[:, 2]
        L[:, 3, 0] = off[:, 3]; L[:, 3, 1] = off[:, 4]; L[:, 3, 2] = off[:, 5]
    P = L @ L.transpose(0, 2, 1) + 1e-9 * np.eye(4)
    Sigma = np.linalg.inv(P)
    return Sigma[:, :3, :3]


def _eig_spatial(cov3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose Nx3x3 symmetric covariances. Returns (eigvals (N,3), eigvecs (N,3,3))."""
    w, V = np.linalg.eigh(cov3)
    w = np.clip(w, 1e-6, None)
    return w, V


def _azimuthal_2d(pos3d: np.ndarray) -> np.ndarray:
    """Equidistant azimuthal projection from 3D head coords (mm).
    Center on mean, normalise to unit sphere, then (theta, phi) -> (theta*cos phi, theta*sin phi)."""
    p = pos3d - pos3d.mean(axis=0, keepdims=True)
    r = np.linalg.norm(p, axis=1, keepdims=True) + 1e-8
    pu = p / r
    theta = np.arccos(np.clip(pu[:, 2], -1.0, 1.0))   # polar angle from +z (top of head)
    phi = np.arctan2(pu[:, 1], pu[:, 0])
    x = theta * np.cos(phi)
    y = theta * np.sin(phi)
    return np.stack([x, y], axis=-1)


def _build_dataset(name, root, subject, sr_factor):
    if name == 'localize_mi':
        from data.localize_mi_dataloader import LocalizeMIDataset
        return LocalizeMIDataset(root=root, subject=subject, sr_factor=sr_factor,
                                 normalization='per_channel_zscore', return_metadata=True)
    if name == 'seed':
        from data.seed_dataloader import SEEDDataset
        return SEEDDataset(root=root, subject=subject, sr_factor=sr_factor,
                           normalization='per_channel_zscore', return_metadata=True)
    if name == 'seed_iv':
        from data.seed_iv_dataloader import SEEDIVDataset
        return SEEDIVDataset(root=root, subject=subject, sr_factor=sr_factor,
                             normalization='per_channel_zscore', return_metadata=True)
    raise ValueError(name)


def _electrode_positions(name, data_root, subject):
    if name == 'localize_mi':
        import pandas as pd
        path = Path(data_root) / subject / 'eeg' / f'{subject}_task-seegstim_electrodes.tsv'
        df = pd.read_csv(path, sep='\t')
        return df[['x', 'y', 'z']].values.astype(np.float64) * 1000
    if name == 'seed_iv':
        from data.seed_iv_dataloader import SEEDIVDataset
        return SEEDIVDataset.get_electrode_positions_mm(Path(data_root))
    if name == 'seed':
        from data.seed_dataloader import SEEDDataset
        return SEEDDataset.get_electrode_positions_mm(Path(data_root))
    raise ValueError(name)


# MUST match training (train.py:_get_sampling_rate). Wrong SR makes t/SR misalign
# with the model's learned mu_t and the reconstruction collapses after a few samples.
DATASET_SAMPLING_RATE = {
    'localize_mi': 8000,
    'seed':         200,
    'seed_iv':      200,
}

# Brain sphere radius used at training time (paper/media/make_ellipsoids.py).
BRAIN_R_MM = 90.0


def _build_model(cfg, n_hd, n_ld, electrode_positions, sampling_rate):
    accepted = set(inspect.signature(EMAG.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in accepted and k not in {
        'n_electrodes', 'n_ld_channels', 'device', 'electrode_positions', 'sampling_rate'}}
    kwargs.update(n_electrodes=n_hd, n_ld_channels=n_ld, device='cpu',
                  electrode_positions=electrode_positions, sampling_rate=sampling_rate)
    return EMAG(**kwargs)


def _forward_full(model, ld_t, T, time_chunk=20):
    """Run model over the full epoch, chunked along time."""
    out = []
    with torch.no_grad():
        for t0 in range(0, T, time_chunk):
            t1 = min(t0 + time_chunk, T)
            tin = torch.arange(t0, t1, dtype=torch.float32).unsqueeze(0)
            ld_chunk = ld_t[:, t0:t1].unsqueeze(0) if ld_t is not None else None
            pred = model(tin, x_ld=ld_chunk)              # (1, Tc, n_hd)
            out.append(pred.squeeze(0).transpose(0, 1).cpu().numpy())  # (n_hd, Tc)
    return np.concatenate(out, axis=1)                    # (n_hd, T)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--dataset', required=True, choices=['localize_mi', 'seed', 'seed_iv'])
    p.add_argument('--subject', required=True)
    p.add_argument('--sr_factor', type=int, required=True)
    p.add_argument('--data_root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epoch_idx', type=int, default=0)
    p.add_argument('--max_gaussians', type=int, default=4000)
    p.add_argument('--time_chunk', type=int, default=20)
    p.add_argument('--time_stride', type=int, default=1,
                   help='Sub-sample every Nth timestep (reduces JSON size).')
    p.add_argument('--label', default=None)
    args = p.parse_args()

    print(f"[1/5] Loading checkpoint {args.checkpoint}")
    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck.get('config') or ck.get('args') or {}
    sd = ck.get('model_state_dict', ck)

    print(f"[2/5] Loading dataset {args.dataset} subject={args.subject}")
    ds = _build_dataset(args.dataset, args.data_root, args.subject, args.sr_factor)
    item = ds[args.epoch_idx]
    ld_t = item['x_ld']    # (n_ld, T)
    hd_t = item['x_hd']    # (n_hd, T)
    n_ld, T = ld_t.shape
    n_hd = hd_t.shape[0]
    elec_pos_3d = _electrode_positions(args.dataset, args.data_root, args.subject)
    elec_2d = _azimuthal_2d(elec_pos_3d[:n_hd])

    sampling_rate = DATASET_SAMPLING_RATE[args.dataset]
    print(f"[3/5] Building model and running forward (n_hd={n_hd}, n_ld={n_ld}, T={T}, SR={sampling_rate})")
    model = _build_model(cfg, n_hd, n_ld, elec_pos_3d[:n_hd], sampling_rate)
    sd_filtered = {k: v for k, v in sd.items() if k in model.state_dict()}
    model.load_state_dict(sd_filtered, strict=False)
    model.eval()
    hd_pred = _forward_full(model, ld_t, T, time_chunk=args.time_chunk)  # (n_hd, T)

    if args.time_stride > 1:
        idx = np.arange(0, T, args.time_stride)
        hd_truth = hd_t.numpy()[:, idx]
        hd_pred  = hd_pred[:, idx]
        ld_arr   = ld_t.numpy()[:, idx]
        time_axis = idx.tolist()
    else:
        hd_truth = hd_t.numpy()
        ld_arr   = ld_t.numpy()
        time_axis = list(range(T))

    print(f"[4/5] Extracting Gaussian params")
    centers = sd.get('centers')
    if centers is None:
        bg = BrainGrid(grid_resolution=cfg.get('grid_resolution', 20),
                       grid_type=cfg.get('grid_type', 'cube'),
                       grid_support=cfg.get('grid_support', 'volume'))
        grid_pts = bg.get_grid_points().cpu().numpy()
        active_idx = sd.get('active_grid_idx').numpy()
        active_pts = grid_pts[active_idx]
        G = cfg.get('n_gaussians_per_point', 3)
        centers = np.repeat(active_pts, G, axis=0)
    else:
        centers = centers.detach().cpu().numpy()
    amplitude = sd['amplitude'].detach().cpu().numpy()
    mu_t = sd['mu_t'].detach().cpu().numpy()
    sigma_xyz, sigma_t = _spatial_temporal_sigma(sd['chol_log_diag'])
    log_diag = sd['chol_log_diag'].detach().cpu().numpy()
    off_diag = sd.get('chol_off_diag')
    if off_diag is None:
        off_diag = np.zeros((log_diag.shape[0], 6), dtype=np.float64)
    else:
        off_diag = off_diag.detach().cpu().numpy()
    cov3 = _chol_to_spatial_cov(log_diag, off_diag)        # (N, 3, 3)
    eigvals, eigvecs = _eig_spatial(cov3)                  # (N, 3), (N, 3, 3)

    G = int(cfg.get('n_gaussians_per_point', 3))
    parent_id_full = (np.arange(len(amplitude)) // G).astype(np.int32)

    order = np.argsort(-np.abs(amplitude))
    if args.max_gaussians and len(order) > args.max_gaussians:
        order = order[:args.max_gaussians]
    centers, amplitude, mu_t = centers[order], amplitude[order], mu_t[order]
    sigma_xyz, sigma_t = sigma_xyz[order], sigma_t[order]
    eigvals, eigvecs = eigvals[order], eigvecs[order]
    parent_id = parent_id_full[order]

    print(f"[5/5] Writing {args.out}")
    payload = {
        'id': args.out.stem,
        'label': args.label or f"{args.dataset} / {args.subject} / SR×{args.sr_factor}",
        'dataset': args.dataset, 'subject': args.subject, 'sr_factor': args.sr_factor,
        'n_time': len(time_axis), 'sampling_rate': float(sampling_rate),
        'time_axis': time_axis,
        # IMPORTANT: gaussians are stored sorted by |amp| desc (above), so the
        # original parent-grid grouping (G consecutive per point) is *broken*.
        # We tag each surviving entry with its parent_id so the viewer can
        # regroup the G mixture components for combined-blob rendering.
        'n_gaussians': int(len(amplitude)),
        'n_gaussians_per_point': int(cfg.get('n_gaussians_per_point', 3)),
        'gaussians': {
            'x':   centers[:, 0].round(2).tolist(),
            'y':   centers[:, 1].round(2).tolist(),
            'z':   centers[:, 2].round(2).tolist(),
            'sx':  sigma_xyz[:, 0].round(3).tolist(),
            'sy':  sigma_xyz[:, 1].round(3).tolist(),
            'sz':  sigma_xyz[:, 2].round(3).tolist(),
            'mu_t': mu_t.round(4).tolist(),
            'st':  sigma_t.round(4).tolist(),
            'amp': amplitude.round(4).tolist(),
            # Per-Gaussian eigendecomposition of the 3x3 spatial covariance
            # for client-side ellipsoid rendering.  Shapes: (N,3) and (N,3,3).
            'eigvals': eigvals.round(4).tolist(),
            'eigvecs': eigvecs.round(5).tolist(),
            'parent_id': parent_id.tolist(),
        },
        'electrodes': {
            'xy':  elec_2d.round(3).tolist(),
            'xyz': elec_pos_3d[:n_hd].round(2).tolist(),
            'n':   int(n_hd),
        },
        # Use the same brain sphere as training (paper convention r=90mm).
        # Center = mean of HD electrodes (head origin in dataset frame).
        'head': {
            'center': elec_pos_3d[:n_hd].mean(axis=0).round(2).tolist(),
            'radius': BRAIN_R_MM,
        },
        # Floats stored as 16-bit fixed-point ints scaled by 1000 for compactness:
        'hd_truth': (hd_truth * 1000).astype(np.int16).tolist(),
        'hd_pred':  (hd_pred  * 1000).astype(np.int16).tolist(),
        'signal_scale': 0.001,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(payload, f, separators=(',', ':'))
    sz_kb = args.out.stat().st_size / 1024
    print(f"Wrote {args.out}  ({payload['n_gaussians']} gaussians, {len(time_axis)} timesteps, {sz_kb:.0f} KB)")


if __name__ == '__main__':
    main()

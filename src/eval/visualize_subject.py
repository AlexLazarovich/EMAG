"""Visualize a single (subject, time) sample: LD input, HD ground truth, EMAG reconstruction.

Usage:
  PYTHONPATH=src python src/eval/visualize_subject.py \
      --checkpoint checkpoints/<run>/<subj>.pt \
      --dataset localize_mi --subject sub-01 --epoch_idx 0 --time_idx 1234 \
      --out viz.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import inspect

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from models.model import EMAG
from models.base import BrainGrid
from data.localize_mi_dataloader import LocalizeMIDataset
from data.seed_dataloader import SEEDDataset
from data.seed_iv_dataloader import SEEDIVDataset


def get_electrode_positions(dataset: str, data_root: Path, subject: str) -> np.ndarray:
    if dataset == 'localize_mi':
        elec_path = data_root / subject / 'eeg' / f'{subject}_task-seegstim_electrodes.tsv'
        df = pd.read_csv(elec_path, sep='\t')
        return df[['x', 'y', 'z']].values.astype(np.float64) * 1000
    if dataset == 'seed_iv':
        return SEEDIVDataset.get_electrode_positions_mm(data_root)
    if dataset == 'seed':
        return SEEDDataset.get_electrode_positions_mm(data_root)
    raise ValueError(f"Unknown dataset {dataset}")


def build_dataset(name: str, root: Path, subject: str, sr_factor: int):
    if name == 'localize_mi':
        return LocalizeMIDataset(root=root, subject=subject, sr_factor=sr_factor,
                                 normalization='per_channel_zscore', return_metadata=True)
    if name == 'seed':
        return SEEDDataset(root=root, subject=subject, sr_factor=sr_factor,
                           normalization='per_channel_zscore', return_metadata=True)
    if name == 'seed_iv':
        return SEEDIVDataset(root=root, subject=subject, sr_factor=sr_factor,
                             normalization='per_channel_zscore', return_metadata=True)
    raise ValueError(f"Unknown dataset {name}")


def load_model(ckpt_path: Path, device: torch.device,
               electrode_positions: np.ndarray) -> EMAG:
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ck.get('config') or ck.get('args') or {}
    sd = ck['model_state_dict'] if 'model_state_dict' in ck else ck

    n_hd = cfg.get('n_hd')
    n_ld = cfg.get('n_ld')
    if n_hd is None or n_ld is None:
        raise RuntimeError("Checkpoint missing n_hd/n_ld in config; re-run training to embed config.")

    accepted = set(inspect.signature(EMAG.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in accepted and k not in {
        'n_electrodes', 'n_ld_channels', 'device', 'electrode_positions'}}
    kwargs.update(
        n_electrodes=n_hd,
        n_ld_channels=n_ld,
        device=device,
        electrode_positions=electrode_positions,
    )
    model = EMAG(**kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[load_model] missing keys: {len(missing)} (e.g. {missing[:3]})")
    if unexpected:
        print(f"[load_model] unexpected keys: {len(unexpected)} (e.g. {unexpected[:3]})")
    model.to(device).eval()
    return model


def plot_triptych(ld: np.ndarray, hd_true: np.ndarray, hd_pred: np.ndarray,
                  time_idx: int, out_path: Path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, sig, title in zip(
        axes,
        [ld, hd_true, hd_pred],
        [f'LD input  ({ld.shape[0]} channels)',
         f'HD ground truth  ({hd_true.shape[0]} channels)',
         f'EMAG reconstruction  ({hd_pred.shape[0]} channels)'],
    ):
        ax.imshow(sig, aspect='auto', cmap='RdBu_r',
                  vmin=-np.max(np.abs(sig)), vmax=np.max(np.abs(sig)))
        ax.set_title(title)
        ax.set_ylabel('channel')
        ax.axvline(time_idx, color='k', lw=0.8, alpha=0.6)
    axes[-1].set_xlabel('time (samples)')
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=Path)
    p.add_argument('--dataset', required=True, choices=['localize_mi', 'seed', 'seed_iv'])
    p.add_argument('--data_root', type=Path, required=True,
                   help='Path to the dataset root (e.g. /path/to/Localize-MI/derivatives/epochs).')
    p.add_argument('--subject', required=True)
    p.add_argument('--sr_factor', type=int, default=4)
    p.add_argument('--epoch_idx', type=int, default=0,
                   help='Index into the dataset (which epoch of this subject).')
    p.add_argument('--time_idx', type=int, default=0,
                   help='Time sample to mark on the plot.')
    p.add_argument('--out', type=Path, default=Path('viz.png'))
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()

    device = torch.device(args.device)
    ds = build_dataset(args.dataset, args.data_root, args.subject, args.sr_factor)
    if args.epoch_idx >= len(ds):
        raise IndexError(f"epoch_idx {args.epoch_idx} >= dataset size {len(ds)}")
    item = ds[args.epoch_idx]
    ld_t, hd_t = item['x_ld'], item['x_hd']

    elec_pos = get_electrode_positions(args.dataset, args.data_root, args.subject)
    model = load_model(args.checkpoint, device, elec_pos)

    with torch.no_grad():
        T = hd_t.shape[-1]
        t_input = torch.arange(T, dtype=torch.float32, device=device).unsqueeze(0)
        x_ld = ld_t.unsqueeze(0).to(device)
        pred = model(t_input, x_ld=x_ld)             # (1, T, n_hd)
        pred = pred.squeeze(0).transpose(0, 1).cpu().numpy()  # (n_hd, T)

    plot_triptych(ld_t.numpy(), hd_t.numpy(), pred, args.time_idx, args.out)


if __name__ == '__main__':
    main()

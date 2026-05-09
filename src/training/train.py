"""Train a single model on Localize-MI / SEED / SEED-IV at a given SR factor.

Per-subject protocol (matches the paper):
  - Each subject is trained independently
  - 80/20 epoch split within each subject
  - LD electrode selection is persistent across all runs of a subject
  - Results are reported as mean+/-std across subjects

Backward compatible with the old benchmark scripts (defaults preserve prior
behaviour). Adds ablation flags (see ABLATIONS.md): --ablation_id,
--results_subdir, --covariance_mode, --offdiag_mode, --center_mode,
--ld_conditioning, --ld_input_mode, --mixture_assignment, --forward_operator,
--leadfield_path, --grid_support, etc.

Usage:
  python train.py --dataset localize_mi --subject sub-01
"""
import gc
import os
import sys
import json
import time
import signal
import logging
import argparse
import hashlib
import platform
import subprocess
from pathlib import Path
from typing import Optional


# --- Graceful preemption handling -------------------------------------------
# SLURM sends SIGTERM (and SIGUSR1 if --signal=B:USR1@N is set) before kill.
# We flip a flag and let the training loop exit cleanly after the current
# epoch's per-epoch resume.pt save, so requeue picks up exactly there.
_PREEMPT_FLAG = {'requested': False, 'sig': None}


def _preempt_handler(signum, frame):
    _PREEMPT_FLAG['requested'] = True
    _PREEMPT_FLAG['sig'] = signum


signal.signal(signal.SIGTERM, _preempt_handler)
try:
    signal.signal(signal.SIGUSR1, _preempt_handler)
except (AttributeError, ValueError):
    pass


def _atomic_save(obj, path: Path):
    """torch.save to a sibling .tmp then os.replace — never leaves a half-written file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from models.base import BrainGrid, create_leadfield  # noqa: F401
from models.model import EMAG
try:
    from data.ld_interpolation import interpolate_ld_to_hd
except Exception:  # pragma: no cover
    interpolate_ld_to_hd = None
try:
    from data.leadfield_io import load_leadfield_matrix
except Exception:  # pragma: no cover
    load_leadfield_matrix = None
from data.localize_mi_dataloader import LocalizeMIDataset
from data.seed_iv_dataloader import SEEDIVDataset
from data.seed_dataloader import SEEDDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_REGISTRY = {'emag': ('EMAG', EMAG)}


def parse_args():
    parser = argparse.ArgumentParser(description='Train a single 4DGSEEG model')
    parser.add_argument('--model', default='emag', choices=['emag'])
    parser.add_argument('--dataset', type=str, default='localize_mi',
                        choices=['localize_mi', 'seed_iv', 'seed'])
    parser.add_argument('--subject', type=str, default=None)
    parser.add_argument('--data_root', type=Path, required=True,
                        help='Path to the dataset root (e.g. /path/to/Localize-MI/derivatives/epochs).')
    parser.add_argument('--results_dir', type=Path, required=True,
                        help='Directory where per-subject results will be written.')
    parser.add_argument('--checkpoint_dir', type=Path, required=True,
                        help='Directory where checkpoints will be written.')
    parser.add_argument('--split_seed', type=int, default=42)
    parser.add_argument('--channel_seed', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--sr_factor', type=int, default=4)
    parser.add_argument('--time_chunk', type=int, default=10,
                        help='0 = auto-detect max chunk that fits in GPU memory')
    parser.add_argument('--batch_size', type=int, default=0,
                        help='0 = auto-detect after time_chunk probe')
    parser.add_argument('--patience', type=int, default=0)
    parser.add_argument('--ld_count_rounding', type=str, default='floor',
                        choices=['floor', 'ceil', 'round'],
                        help='How n_ld is computed from n_hd/sr_factor (default floor for backward compat).')
    parser.add_argument('--normalization', type=str, default='per_channel_zscore',
                        choices=['per_channel_zscore', 'none'],
                        help='Dataloader normalization. per_channel_zscore (default, backward compat) '
                             'destroys absolute amplitudes per window; "none" keeps raw scales — needed '
                             'for amplitude-sensitive downstream tasks like DE features.')
    parser.add_argument('--continue_training', action='store_true',
                        help='If a final checkpoint exists for a subject, load it and '
                             'continue training up to --epochs (instead of skipping).')
    parser.add_argument('--min_delta', type=float, default=1e-4)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--run_name', type=str, default=None)
    # Ablation bookkeeping
    parser.add_argument('--ablation_id', type=str, default=None)
    parser.add_argument('--ablation_group', type=str, default=None,
                        help='Tag stored in results JSON for aggregation.')
    parser.add_argument('--results_subdir', type=str, default=None,
                        help='Optional subdirectory under --results_dir')
    parser.add_argument('--checkpoint_subdir', type=str, default=None,
                        help='Optional subdirectory under --checkpoint_dir')
    parser.add_argument('--grid_search_json', type=str, default='grid_search_results.json')
    # NeRF-Hybrid
    parser.add_argument('--n_freq_bands', type=int, default=10)
    parser.add_argument('--mlp_hidden_dim', type=int, default=256)
    parser.add_argument('--mlp_depth', type=int, default=4)
    # Per-Voxel
    parser.add_argument('--sigma_base_mm', type=float, default=15.0)
    parser.add_argument('--temporal_mlp_hidden_dim', type=int, default=64)
    parser.add_argument('--temporal_mlp_depth', type=int, default=2)
    # Mixture 4D
    parser.add_argument('--n_components', type=int, default=1000)
    # Mixture 4D Cov
    parser.add_argument('--n_gaussians_per_point', type=int, default=3)
    # Grid
    parser.add_argument('--n_grid', type=int, default=8000)
    parser.add_argument('--grid_resolution', type=int, default=20)
    parser.add_argument('--grid_type', type=str, default='cube', choices=['cube', 'sphere'])
    parser.add_argument('--grid_support', type=str, default='volume',
                        choices=['volume', 'surface'])
    # LD pathway
    parser.add_argument('--disable_ld_conditioning', action='store_true',
                        help='Train/eval without LD input (legacy flag).')
    parser.add_argument('--ld_conditioning', type=str, default='per_grid',
                        choices=['per_grid', 'none', 'global_scalar'])
    parser.add_argument('--ld_input_mode', type=str, default='raw',
                        choices=['raw', 'interpolated'])
    # Cov ablations (legacy + new)
    parser.add_argument('--cov_ablation', type=str, default='full',
                        choices=['full', 'no_spacetime_coupling', 'diagonal_precision'],
                        help='Legacy alias; mapped to --offdiag_mode for backward compat.')
    parser.add_argument('--covariance_mode', type=str, default='full_4d',
                        choices=['full_4d', 'spatial_3d_only', 'diagonal_4d', 'isotropic_scalar'])
    parser.add_argument('--offdiag_mode', type=str, default='all',
                        choices=['all', 'none', 'spatial_only', 'spacetime_only'])
    parser.add_argument('--center_mode', type=str, default='fixed_grid',
                        choices=['fixed_grid', 'learnable'])
    parser.add_argument('--mixture_assignment', type=str, default='uniform',
                        choices=['uniform', 'subset_random', 'subset_attention'])
    parser.add_argument('--active_point_ratio', type=float, default=1.0)
    parser.add_argument('--forward_operator', type=str, default='gaussian_splat',
                        choices=['gaussian_splat', 'learned_linear', 'leadfield_matrix',
                                 'direct_electrode_mlp'])
    parser.add_argument('--learned_projection_rank', type=int, default=0)
    parser.add_argument('--leadfield_path', type=str, default=None)
    # Floating-cell (mixture_4d_floating)
    parser.add_argument('--cell_radius', type=float, default=None,
                        help='Override per-cell radius (mm). Default: scale * nearest-neighbor distance per point.')
    parser.add_argument('--cell_radius_scale', type=float, default=0.5,
                        help='Multiplier on nearest-neighbor distance for cell radius (default 0.5).')
    parser.add_argument('--init_mode', type=str, default='grid', choices=['grid', 'random'],
                        help='Init centers on grid (default) or random inside each cell.')
    parser.add_argument('--init_seed', type=int, default=0,
                        help='Seed for random init of mixture centers when --init_mode=random.')
    # Channel selection
    parser.add_argument('--channel_indices', type=str, default=None,
                        help='Optional comma-separated global channel indices for LD.')
    parser.add_argument('--constellation', type=str, default=None,
                        help='SEED / SEED-IV only: circumference, fronto_temporal, interior, random')
    parser.add_argument('--random_channel_selection', dest='random_channel_selection',
                        action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--bad_channel_policy', type=str, default='keep_all',
                        choices=['keep_all', 'drop_bad'],
                        help='Localize-MI only.')
    return parser.parse_args()


def _parse_channel_indices(s: Optional[str]):
    if s is None or not str(s).strip():
        return None
    return [int(x.strip()) for x in str(s).split(',') if x.strip()]


def compute_metrics(eeg_true, eeg_pred):
    mse = np.mean((eeg_pred - eeg_true)**2)
    mse_base = np.mean(eeg_true**2)
    nmse = mse / mse_base if mse_base > 0 else 0.0
    t_flat, p_flat = eeg_true.flatten(), eeg_pred.flatten()
    mt, mp = np.mean(t_flat), np.mean(p_flat)
    num = np.sum((t_flat - mt) * (p_flat - mp))
    den = np.sqrt(np.sum((t_flat - mt)**2) * np.sum((p_flat - mp)**2))
    pcc = num / den if den > 0 else 0.0
    sig_pow = np.mean(eeg_true**2)
    noise_pow = np.mean((eeg_pred - eeg_true)**2)
    snr = 10 * np.log10(sig_pow / noise_pow) if noise_pow > 0 and sig_pow > 0 else 0.0
    return {'NMSE': float(nmse), 'PCC': float(pcc), 'SNR': float(snr)}


def _mixture_seed(args, subject: str) -> int:
    h = int(hashlib.sha1(str(subject).encode('utf-8')).hexdigest()[:8], 16)
    return (int(args.channel_seed) + h) % (2**32)


def _mixture_n_total(args) -> int:
    bg = BrainGrid(args.grid_resolution, 'cpu',
                   grid_type=args.grid_type, grid_support=args.grid_support)
    n_full = bg.n_grid
    g = args.n_gaussians_per_point
    if args.mixture_assignment == 'uniform':
        n_active = n_full
    else:
        n_active = max(1, int(round(float(args.active_point_ratio) * n_full)))
    return int(n_active * g)


def _needs_metadata(args) -> bool:
    return args.ld_input_mode == 'interpolated'


def _make_datasets(args, subject, dataset_root):
    tc = args.time_chunk if args.time_chunk > 0 else None
    ch_idx = _parse_channel_indices(args.channel_indices)
    meta = _needs_metadata(args)
    if args.dataset == 'localize_mi':
        common = dict(
            root=dataset_root, sr_factor=args.sr_factor, split_mode='paper80_20',
            normalization=args.normalization, return_metadata=meta,
            time_chunk_size=tc, subject=subject,
            split_seed=args.split_seed, channel_seed=args.channel_seed,
            consistent_channels_within_recording=True,
            fixed_ld_channel_indices=ch_idx,
            random_channel_selection=args.random_channel_selection,
            bad_channel_policy=args.bad_channel_policy,
            ld_count_rounding=args.ld_count_rounding,
        )
        train_ds = LocalizeMIDataset(subset='train', **common)
        test_ds = LocalizeMIDataset(subset='test', **common)
    elif args.dataset == 'seed_iv':
        common = dict(
            root=dataset_root, sr_factor=args.sr_factor, split_mode='paper80_20',
            normalization=args.normalization, return_metadata=meta,
            subject=subject, split_seed=args.split_seed,
            channel_seed=args.channel_seed,
            constellation=args.constellation,
            fixed_ld_channel_indices=ch_idx,
            random_channel_selection=args.random_channel_selection,
            preload=True,
            ld_count_rounding=args.ld_count_rounding,
        )
        train_ds = SEEDIVDataset(subset='train', **common)
        test_ds = SEEDIVDataset(subset='test', **common)
    elif args.dataset == 'seed':
        common = dict(
            root=dataset_root, sr_factor=args.sr_factor, split_mode='paper80_20',
            normalization=args.normalization, return_metadata=meta,
            subject=subject, split_seed=args.split_seed,
            channel_seed=args.channel_seed,
            constellation=args.constellation,
            fixed_ld_channel_indices=ch_idx,
            random_channel_selection=args.random_channel_selection,
            preload=True,
            ld_count_rounding=args.ld_count_rounding,
        )
        train_ds = SEEDDataset(subset='train', **common)
        test_ds = SEEDDataset(subset='test', **common)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    return train_ds, test_ds


def _get_electrode_positions(args, subject, dataset_root):
    if args.dataset == 'localize_mi':
        elec_path = dataset_root / subject / 'eeg' / f'{subject}_task-seegstim_electrodes.tsv'
        elec_df = pd.read_csv(elec_path, sep='\t')
        return elec_df[['x', 'y', 'z']].values.astype(np.float64) * 1000
    elif args.dataset == 'seed_iv':
        return SEEDIVDataset.get_electrode_positions_mm(dataset_root)
    elif args.dataset == 'seed':
        return SEEDDataset.get_electrode_positions_mm(dataset_root)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def _get_sampling_rate(args):
    if args.dataset == 'localize_mi':
        return 8000
    elif args.dataset in ('seed_iv', 'seed'):
        return 200
    return 250


def _unwrap_batch(batch):
    if isinstance(batch, dict):
        return batch['x_ld'], batch['x_hd'], batch.get('meta')
    if len(batch) == 3:
        return batch[0], batch[1], batch[2]
    return batch[0], batch[1], None


def _collate_ld_for_model(ld_chunk, meta, args, pos_hd_mm, device):
    """Move LD to device. If ld_input_mode='interpolated', interpolate to HD layout."""
    ld = ld_chunk.to(device).float()
    if args.ld_input_mode != 'interpolated' or meta is None or interpolate_ld_to_hd is None:
        return ld
    B, _, T = ld.shape
    out = []
    for b in range(B):
        m = {k: (v[b] if torch.is_tensor(v) and v.dim() > 0 else v) for k, v in meta.items()}
        ld_idx = m['ld_indices'].detach().cpu().numpy().reshape(-1)
        x_interp = interpolate_ld_to_hd(ld[b].detach().cpu().numpy(), ld_idx, pos_hd_mm)
        out.append(torch.from_numpy(x_interp).to(device=device, dtype=ld.dtype))
    return torch.stack(out, dim=0)


def _probe_max_time_chunk(model, n_ld, n_time, device, safety_factor=0.85,
                          use_ld: bool = True):
    if device == 'cpu' or not torch.cuda.is_available():
        return n_time
    if isinstance(model, EMAG) and getattr(model, 'forward_operator', '') == 'learned_linear':
        return min(10, n_time)
    gpu_total = torch.cuda.get_device_properties(device).total_memory

    def measure_peak(tc):
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        t_input = torch.arange(0, tc, dtype=torch.float32, device=device).unsqueeze(0)
        if use_ld:
            ld_dummy = torch.randn(1, n_ld, tc, device=device)
            pred = model(t_input, x_ld=ld_dummy)
        else:
            ld_dummy = None
            pred = model(t_input, x_ld=None)
        loss = pred.pow(2).mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated(device)
        del t_input, pred, loss
        if ld_dummy is not None:
            del ld_dummy
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        return peak

    tc_small, tc_large = 10, 50
    tc_large = min(tc_large, n_time)
    if tc_large <= tc_small:
        tc_large = tc_small + 10
    try:
        peak_small = measure_peak(tc_small)
        peak_large = measure_peak(tc_large)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        logger.warning("Auto time_chunk probe OOM — falling back to tc=10")
        return 10
    per_step = (peak_large - peak_small) / (tc_large - tc_small)
    fixed_cost = peak_small - per_step * tc_small
    if per_step <= 0:
        return n_time
    usable = gpu_total * safety_factor
    max_tc = int((usable - fixed_cost) / per_step)
    max_tc = max(10, min(max_tc, n_time))
    logger.info(f"Auto time_chunk: GPU={gpu_total/1e9:.1f}GB, "
                f"fixed={fixed_cost/1e6:.0f}MB, per_step={per_step/1e3:.1f}KB, "
                f"max_tc={max_tc} (capped at n_time={n_time})")
    return max_tc


def _probe_max_batch_size(model, n_ld, time_chunk, device, safety_factor=0.85,
                          use_ld: bool = True, max_try: int = 256):
    if device == 'cpu' or not torch.cuda.is_available():
        return 1
    gpu_total = torch.cuda.get_device_properties(device).total_memory

    def measure_peak(B):
        gc.collect(); torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        t_input = torch.arange(0, time_chunk, dtype=torch.float32, device=device) \
            .unsqueeze(0).expand(B, -1)
        if use_ld:
            ld_dummy = torch.randn(B, n_ld, time_chunk, device=device)
            pred = model(t_input, x_ld=ld_dummy)
        else:
            ld_dummy = None
            pred = model(t_input, x_ld=None)
        loss = pred.pow(2).mean()
        loss.backward()
        peak = torch.cuda.max_memory_allocated(device)
        del t_input, pred, loss
        if ld_dummy is not None:
            del ld_dummy
        model.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        return peak

    try:
        peak_1 = measure_peak(1)
        peak_4 = measure_peak(4)
    except torch.cuda.OutOfMemoryError:
        gc.collect(); torch.cuda.empty_cache()
        logger.warning("Auto batch_size probe OOM at B<=4 — falling back to B=1")
        return 1
    per_sample = (peak_4 - peak_1) / 3
    fixed_cost = peak_1 - per_sample
    if per_sample <= 0:
        return min(max_try, 32)
    usable = gpu_total * safety_factor
    max_b = int((usable - fixed_cost) / per_sample)
    max_b = max(1, min(max_b, max_try))
    logger.info(f"Auto batch_size: GPU={gpu_total/1e9:.1f}GB, "
                f"fixed={fixed_cost/1e6:.0f}MB, per_sample={per_sample/1e6:.1f}MB, "
                f"max_B={max_b} (capped at {max_try})")
    return max_b


def _get_dataset_root(args):
    return Path(args.data_root)


def _list_subjects(args, dataset_root):
    if args.dataset == 'localize_mi':
        return LocalizeMIDataset.list_subjects(dataset_root)
    elif args.dataset == 'seed_iv':
        return SEEDIVDataset.list_subjects(dataset_root)
    elif args.dataset == 'seed':
        return SEEDDataset.list_subjects(dataset_root)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                      cwd=str(Path(__file__).resolve().parent), stderr=subprocess.DEVNULL, text=True)
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _write_run_manifest(path: Path, args, subjects: list, dataset_root: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'schema_version': '1.0',
        'ablation_id': args.ablation_id,
        'run_name': args.run_name or f"{args.model}_sr{args.sr_factor}",
        'command_line': sys.argv,
        'started_at_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'environment': {
            'python': sys.version,
            'torch': torch.__version__,
            'platform': platform.platform(),
        },
        'git_commit': _git_commit(),
        'dataset': {
            'name': args.dataset,
            'root_resolved': str(Path(dataset_root).resolve()),
            'split_seed': args.split_seed,
            'channel_seed': args.channel_seed,
            'random_channel_selection': args.random_channel_selection,
            'bad_channel_policy': getattr(args, 'bad_channel_policy', 'keep_all'),
            'ld_input_mode': args.ld_input_mode,
        },
        'subjects': subjects,
        'config': {k: v for k, v in vars(args).items() if k != 'device'},
    }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)


def _regularization_terms(model: nn.Module, coef: float = 2e-4):
    keys = ('logvar', 'means', 'cholesky', 'chol_log_diag', 'chol_off_diag',
            'centers', 'proj_w', 'proj_u', 'proj_v', 'log_sigma_space', 'log_sigma_time')
    return [coef * (p**2).mean() for n, p in model.named_parameters()
            if any(k in n for k in keys)]


def _ld_tensor_for_forward(ld_chunk, disable_ld: bool):
    return None if disable_ld else ld_chunk


def train_one_subject(args, subject: str, dataset_root: Path, checkpoint_dir: Path,
                      results_dir: Path) -> dict:
    device = args.device

    if args.constellation and args.dataset not in ('seed', 'seed_iv'):
        raise ValueError('--constellation is only supported for seed and seed_iv datasets')
    if args.channel_indices and args.constellation:
        raise ValueError('Use only one of --channel_indices or --constellation')

    train_ds, test_ds = _make_datasets(args, subject, dataset_root)

    batch0 = train_ds[0]
    if isinstance(batch0, dict):
        sample_ld, sample_hd = batch0['x_ld'], batch0['x_hd']
    else:
        sample_ld, sample_hd = batch0[0], batch0[1]
    n_ld, n_hd = sample_ld.shape[0], sample_hd.shape[0]
    T = sample_ld.shape[1]
    logger.info(f"[{subject}] SR={args.sr_factor}x: n_ld={n_ld}, n_hd={n_hd}, T={T}, "
                f"train={len(train_ds)}, test={len(test_ds)}")

    electrode_positions = _get_electrode_positions(args, subject, dataset_root)
    pos_hd_mm = np.asarray(electrode_positions, dtype=np.float64)
    logger.info(f"[{subject}] Loaded {len(electrode_positions)} electrode positions")

    n_ld_model = n_hd if args.ld_input_mode == 'interpolated' else n_ld

    display_name, cls = MODEL_REGISTRY[args.model]
    kwargs = {
        'n_electrodes': n_hd, 'n_grid': args.n_grid,
        'grid_resolution': args.grid_resolution,
        'n_ld_channels': n_ld_model,
        'sampling_rate': _get_sampling_rate(args), 'device': device,
        'electrode_positions': electrode_positions,
        'grid_type': args.grid_type,
        'grid_support': args.grid_support,
    }
    if args.model == 'emag':
        fo = args.forward_operator
        if fo == 'direct_electrode_mlp':
            raise ValueError("Use --model direct_electrode_mlp for direct mapping (Ablation 5.1).")
        kwargs.update(
            n_gaussians_per_point=args.n_gaussians_per_point,
            cov_ablation=args.cov_ablation,
            covariance_mode=args.covariance_mode,
            offdiag_mode=args.offdiag_mode,
            center_mode=args.center_mode,
            ld_conditioning='none' if args.disable_ld_conditioning else args.ld_conditioning,
            mixture_assignment=args.mixture_assignment,
            active_point_ratio=args.active_point_ratio,
            mixture_seed=_mixture_seed(args, subject),
            learned_projection_rank=args.learned_projection_rank,
            forward_operator=fo,
        )
        if fo == 'leadfield_matrix':
            if not args.leadfield_path:
                raise ValueError("--leadfield_path required for forward_operator=leadfield_matrix")
            if load_leadfield_matrix is None:
                raise ImportError("leadfield_io.load_leadfield_matrix is unavailable")
            n_tot = _mixture_n_total(args)
            L = load_leadfield_matrix(args.leadfield_path, n_hd, n_tot, device)
            kwargs['leadfield_matrix'] = L.cpu()

    model = cls(**kwargs)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"[{subject}] Model: {display_name}, Parameters: {n_params:,}")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    use_ld = not args.disable_ld_conditioning

    time_chunk = args.time_chunk
    if time_chunk == 0:
        time_chunk = _probe_max_time_chunk(model, n_ld_model, T, device, use_ld=use_ld)
        logger.info(f"[{subject}] Using auto time_chunk = {time_chunk}")

    batch_size = args.batch_size
    if batch_size == 0:
        batch_size = _probe_max_batch_size(model, n_ld_model, time_chunk, device, use_ld=use_ld)
        logger.info(f"[{subject}] Using auto batch_size = {batch_size}")

    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK') or len(os.sched_getaffinity(0)))
    n_workers = max(1, n_cpus - 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False,
                              num_workers=n_workers, persistent_workers=True, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, drop_last=False,
                              num_workers=n_workers, persistent_workers=True, pin_memory=True)
    logger.info(f"[{subject}] DataLoader: batch_size={batch_size}, num_workers={n_workers}")

    history = []
    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    start_epoch = 0

    run_name = args.run_name or f"{args.model}_sr{args.sr_factor}"
    resume_ckpt_path = checkpoint_dir / f'{run_name}_{subject}_resume.pt'
    final_ckpt_path = checkpoint_dir / f'{run_name}_{subject}.pt'
    if not resume_ckpt_path.exists() and final_ckpt_path.exists() and args.continue_training:
        logger.info(f"[{subject}] Final checkpoint found — continuing training from {final_ckpt_path}")
        ck = torch.load(final_ckpt_path, map_location='cpu', weights_only=False)
        model.load_state_dict({k: v.to(device) for k, v in ck['model_state_dict'].items()})
        history = ck.get('history', [])
        start_epoch = len(history)
        if history:
            val_losses = [h.get('val_loss', float('inf')) for h in history]
            val_losses = [v for v in val_losses if v == v]  # drop NaN
            best_val_loss = min(val_losses) if val_losses else float('inf')
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if start_epoch >= args.epochs:
            logger.warning(f"[{subject}] history length ({start_epoch}) >= --epochs ({args.epochs}); "
                           f"increase --epochs to actually train more.")
        else:
            logger.info(f"[{subject}] Continuing at epoch {start_epoch}/{args.epochs}, "
                        f"best_val_loss={best_val_loss:.6f} (fresh optimizer)")
    elif resume_ckpt_path.exists():
        logger.info(f"[{subject}] Resume checkpoint found — loading {resume_ckpt_path}")
        resume = torch.load(resume_ckpt_path, map_location='cpu', weights_only=False)
        live_sd = resume['model_state_dict']
        live_has_nan = any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in live_sd.values())
        best_sd = resume.get('best_state')
        best_has_nan = best_sd is None or any(torch.is_tensor(v) and not torch.isfinite(v).all() for v in best_sd.values())
        if live_has_nan and not best_has_nan:
            logger.warning(f"[{subject}] Resume model_state_dict has NaN — falling back to best_state and resetting optimizer")
            model.load_state_dict({k: v.to(device) for k, v in best_sd.items()})
            resume['optimizer_state_dict'] = {}  # force fresh optimizer
        else:
            model.load_state_dict({k: v.to(device) for k, v in live_sd.items()})
        try:
            optimizer.load_state_dict(resume['optimizer_state_dict'])
            for state in optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        except (ValueError, KeyError) as e:
            logger.info(f"[{subject}] Optimizer state not loaded ({type(e).__name__}), restarting fresh")
        start_epoch = resume['epoch']
        best_val_loss = resume['best_val_loss']
        patience_counter = resume['patience_counter']
        history = resume['history']
        best_state = resume['best_state']
        logger.info(f"[{subject}] Resumed at epoch {start_epoch}/{args.epochs}, "
                    f"best_val_loss={best_val_loss:.6f}")

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        n_samples = 0
        t0 = time.time()

        for batch in train_loader:
            ld_eeg, hd_eeg, meta = _unwrap_batch(batch)
            n_time = hd_eeg.shape[-1]
            n_chunks = max(1, (n_time + time_chunk - 1) // time_chunk)
            optimizer.zero_grad()
            batch_loss = 0.0
            B_cur = hd_eeg.shape[0]

            for t_start in range(0, n_time, time_chunk):
                t_end = min(t_start + time_chunk, n_time)
                hd_chunk = hd_eeg[:, :, t_start:t_end].to(device).float()
                hd_target = hd_chunk.transpose(1, 2)
                ld_chunk = ld_eeg[:, :, t_start:t_end]
                ld_model = _collate_ld_for_model(ld_chunk, meta, args, pos_hd_mm, device)
                t_input = torch.arange(t_start, t_end, dtype=torch.float32, device=device) \
                    .unsqueeze(0).expand(B_cur, -1)

                pred = model(t_input, x_ld=_ld_tensor_for_forward(ld_model, args.disable_ld_conditioning))
                n_out = pred.shape[-1]
                loss = criterion(pred, hd_target[:, :, :n_out])
                (loss / n_chunks).backward()
                batch_loss += loss.item()

            reg_terms = _regularization_terms(model, coef=1e-4 if args.continue_training else 2e-4)
            if reg_terms:
                sum(reg_terms).backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += batch_loss / n_chunks
            n_samples += 1

        avg_loss = epoch_loss / max(n_samples, 1)

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in test_loader:
                ld_eeg, hd_eeg, meta = _unwrap_batch(batch)
                n_time = hd_eeg.shape[-1]
                n_chunks_v = max(1, (n_time + time_chunk - 1) // time_chunk)
                B_cur = hd_eeg.shape[0]
                sample_loss = 0.0
                for t_start in range(0, n_time, time_chunk):
                    t_end = min(t_start + time_chunk, n_time)
                    hd_chunk = hd_eeg[:, :, t_start:t_end].to(device).float()
                    hd_target = hd_chunk.transpose(1, 2)
                    ld_chunk = ld_eeg[:, :, t_start:t_end]
                    ld_model = _collate_ld_for_model(ld_chunk, meta, args, pos_hd_mm, device)
                    t_input = torch.arange(t_start, t_end, dtype=torch.float32, device=device) \
                        .unsqueeze(0).expand(B_cur, -1)
                    pred = model(t_input, x_ld=_ld_tensor_for_forward(ld_model, args.disable_ld_conditioning))
                    n_out = pred.shape[-1]
                    sample_loss += criterion(pred, hd_target[:, :, :n_out]).item()
                val_loss += sample_loss / n_chunks_v
                val_n += 1
        avg_val_loss = val_loss / val_n if val_n > 0 else float('inf')

        elapsed = time.time() - t0
        logger.info(f"[{subject}] Epoch {epoch+1}/{args.epochs}: "
                    f"train_loss={avg_loss:.6f}, val_loss={avg_val_loss:.6f} ({elapsed:.1f}s)")
        history.append({'epoch': epoch+1, 'train_loss': avg_loss,
                        'val_loss': avg_val_loss, 'time': elapsed})

        if avg_val_loss < best_val_loss - args.min_delta:
            best_val_loss = avg_val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        _atomic_save({
            'epoch': epoch + 1,
            'model_state_dict': {k: v.cpu().clone() for k, v in model.state_dict().items()},
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'best_state': best_state,
            'history': history,
        }, resume_ckpt_path)

        if _PREEMPT_FLAG['requested']:
            logger.warning(f"[{subject}] Preemption signal {_PREEMPT_FLAG['sig']} received — "
                           f"resume.pt saved at epoch {epoch+1}, exiting cleanly for SLURM requeue")
            sys.exit(0)

        if args.patience > 0 and patience_counter >= args.patience:
            logger.info(f"[{subject}] Early stopping at epoch {epoch+1}")
            break

    if resume_ckpt_path.exists():
        resume_ckpt_path.unlink()
        logger.info(f"[{subject}] Removed resume checkpoint")

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        logger.info(f"[{subject}] Restored best model (val_loss={best_val_loss:.6f})")

    logger.info(f"[{subject}] Evaluating...")
    model.eval()
    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            ld_eeg, hd_eeg, meta = _unwrap_batch(batch)
            n_time = hd_eeg.shape[-1]
            B_cur = hd_eeg.shape[0]
            pred_chunks = []
            for t_start in range(0, n_time, time_chunk):
                t_end = min(t_start + time_chunk, n_time)
                ld_chunk = ld_eeg[:, :, t_start:t_end]
                ld_model = _collate_ld_for_model(ld_chunk, meta, args, pos_hd_mm, device)
                t_input = torch.arange(t_start, t_end, dtype=torch.float32, device=device) \
                    .unsqueeze(0).expand(B_cur, -1)
                pred = model(t_input, x_ld=_ld_tensor_for_forward(ld_model, args.disable_ld_conditioning))
                pred_chunks.append(pred.cpu())
            pred_full = torch.cat(pred_chunks, dim=1).transpose(1, 2)
            n_out = pred_full.shape[1]
            all_true.append(hd_eeg[:, :n_out, :].numpy())
            all_pred.append(pred_full.numpy())

    eeg_true = np.concatenate(all_true, axis=0)
    eeg_pred = np.concatenate(all_pred, axis=0)
    metrics = compute_metrics(eeg_true, eeg_pred)
    logger.info(f"[{subject}] Results: NMSE={metrics['NMSE']:.4f}, PCC={metrics['PCC']:.4f}, "
                f"SNR={metrics['SNR']:.2f}dB")

    subj_run_name = f"{run_name}_{subject}"
    ckpt_path = checkpoint_dir / f'{subj_run_name}.pt'
    full_config = vars(args).copy()
    full_config.update({
        'n_ld': n_ld, 'n_hd': n_hd, 'n_params': n_params,
        'subject': subject, 'model_variant': args.model,
        'cov_ablation': getattr(args, 'cov_ablation', 'full'),
        'ablation_group': args.ablation_group,
        'ablation_id': args.ablation_id,
        'mixture_seed_resolved': _mixture_seed(args, subject),
    })
    _atomic_save({
        'model_state_dict': model.state_dict(),
        'metrics': metrics,
        'history': history,
        'config': full_config,
    }, ckpt_path)
    logger.info(f"[{subject}] Saved checkpoint: {ckpt_path}")

    np.savez(
        results_dir / f'{subject}_predictions.npz',
        eeg_true=eeg_true[:3], eeg_pred=eeg_pred[:3],
    )

    return metrics


def main():
    args = parse_args()

    results_dir = args.results_dir
    if args.results_subdir:
        results_dir = results_dir / args.results_subdir
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_dir = args.checkpoint_dir
    if args.checkpoint_subdir:
        checkpoint_dir = checkpoint_dir / args.checkpoint_subdir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset_root = _get_dataset_root(args)

    if args.subject:
        subjects = [args.subject]
    else:
        subjects = _list_subjects(args, dataset_root)
        if not subjects:
            raise FileNotFoundError(f"No subjects found under {dataset_root}")
        logger.info(f"Found {len(subjects)} subjects: {subjects}")

    run_name = args.run_name or f"{args.model}_sr{args.sr_factor}"
    args.run_name = run_name

    try:
        manifest_path = results_dir / run_name / 'run_manifest.json'
        _write_run_manifest(manifest_path, args, subjects, dataset_root)
    except Exception as e:  # pragma: no cover
        logger.warning(f"Could not write run manifest: {e}")

    per_subject_metrics = {}
    for subject in subjects:
        logger.info(f"{'='*60}\nTraining {run_name} on {subject}\n{'='*60}")

        final_ckpt = checkpoint_dir / f'{run_name}_{subject}.pt'
        resume_ckpt = checkpoint_dir / f'{run_name}_{subject}_resume.pt'
        if final_ckpt.exists() and not resume_ckpt.exists() and not args.continue_training:
            ckpt = torch.load(final_ckpt, map_location='cpu', weights_only=False)
            if 'metrics' in ckpt and all(not np.isnan(v) for v in ckpt['metrics'].values()):
                logger.info(f"[{subject}] Final checkpoint exists — reusing saved metrics")
                per_subject_metrics[subject] = ckpt['metrics']
                continue

        metrics = train_one_subject(args, subject, dataset_root, checkpoint_dir, results_dir)
        per_subject_metrics[subject] = metrics

    metric_names = ['NMSE', 'PCC', 'SNR']
    aggregated = {}
    for m in metric_names:
        values = [per_subject_metrics[s][m] for s in subjects]
        aggregated[f'{m}_mean'] = float(np.mean(values))
        aggregated[f'{m}_std'] = float(np.std(values))
        aggregated[f'{m}_values'] = values

    logger.info(f"\n{'='*60}\nAggregated results across {len(subjects)} subjects:")
    for m in metric_names:
        logger.info(f"  {m}: {aggregated[f'{m}_mean']:.4f} ± {aggregated[f'{m}_std']:.4f}")
    logger.info(f"{'='*60}")

    results_json = results_dir / args.grid_search_json
    if results_json.exists():
        with open(results_json) as f:
            all_results = json.load(f)
    else:
        all_results = {}

    full_config = {k: v for k, v in vars(args).items() if k != 'device'}
    all_results[run_name] = {
        'aggregated_metrics': {m: {'mean': aggregated[f'{m}_mean'], 'std': aggregated[f'{m}_std']}
                               for m in metric_names},
        'per_subject_metrics': per_subject_metrics,
        'config': full_config,
        'n_subjects': len(subjects),
        'subjects': subjects,
        'ablation_id': args.ablation_id,
        'ablation_tags': {
            'ablation_group': args.ablation_group,
            'ablation_id': args.ablation_id,
            'cov_ablation': getattr(args, 'cov_ablation', None),
            'covariance_mode': args.covariance_mode,
            'offdiag_mode': args.offdiag_mode,
            'forward_operator': args.forward_operator,
            'ld_conditioning': args.ld_conditioning,
            'ld_input_mode': args.ld_input_mode,
            'mixture_assignment': args.mixture_assignment,
            'disable_ld_conditioning': getattr(args, 'disable_ld_conditioning', False),
            'constellation': getattr(args, 'constellation', None),
            'channel_indices': getattr(args, 'channel_indices', None),
        },
    }
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Updated {results_json}")


if __name__ == '__main__':
    main()

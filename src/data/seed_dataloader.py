"""
SEED EEG Spatial Super-Resolution Dataset.

Expects the **preprocessed** SEED dataset (same .mat format as SEED-IV),
NOT the raw SEED_Multimodal .cnt files.

Expected directory structure under `root`:
    eeg_raw_data/{session}/{subject_id}_{date}.mat
        Keys: {initials}_eeg1 ... {initials}_eeg15  (15 trials per session)
        Shape per trial: (62, T)  — 200 Hz, variable T

    channel_62_pos.locs   (optional)

Labels per session (emotion: 1=positive, 0=neutral, -1=negative remapped to 0,1,2):
    SESSION_LABELS[session] = list of 15 integers (0=neutral, 1=positive, 2=negative)

NOTE: If you have the raw SEED_Multimodal .cnt files, you need to obtain the
preprocessed version from http://bcmi.sjtu.edu.cn/~seed/ instead.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.io as sio
import torch
from torch.utils.data import Dataset

from .eeg_constellations import VALID_PRESETS, indices_for_preset


def _compute_n_ld(n_hd: int, sr_factor: int, mode: str = "floor") -> int:
    """Number of low-density input channels given total HD count and SR factor.

    mode: 'floor' (default; n_hd // sr_factor) — strictly >= sr_factor× upsampling.
          'ceil'  (n_hd / sr_factor rounded up) — closer to literal SR ratio when n_hd
                  is not divisible.
          'round' (round-half-to-even via int(round(...))).
    """
    if sr_factor <= 0:
        raise ValueError(f"sr_factor must be > 0, got {sr_factor}")
    if mode == "floor":
        n = n_hd // sr_factor
    elif mode == "ceil":
        n = (n_hd + sr_factor - 1) // sr_factor
    elif mode == "round":
        n = int(round(n_hd / sr_factor))
    else:
        raise ValueError(f"ld_count_rounding must be floor|ceil|round, got {mode!r}")
    return max(1, min(n, n_hd))

# Standard 10-20 channel order for SEED (same 62-channel setup as SEED-IV)
SEED_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2',
]

# Trial emotion labels per session.
# Original: 1=positive, 0=neutral, -1=negative
# Remapped to:  0=neutral, 1=positive, 2=negative  (non-negative class indices)
# Source: SEED ReadMe — same 15 clips × 3 sessions, repeated across sessions.
SESSION_LABELS = {
    1: [1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2],
    2: [1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2],
    3: [1, 0, 2, 0, 1, 2, 0, 1, 2, 1, 0, 2, 0, 1, 2],
}
N_CLASSES = 3  # neutral, positive, negative


@dataclass
class _TrialInfo:
    subject_id: str
    session: int
    trial_idx: int      # 1-based
    mat_path: Path
    key_prefix: str     # e.g. 'djc_eeg'
    n_channels: int
    n_samples: int
    label: int          # 0/1/2


def _find_data_root(root: Path) -> Path:
    """Resolve actual data root — handles session-subdir and flat layouts.

    Supported layouts
    -----------------
    1. Session subdirs (mirrors SEED-IV):
         root/eeg_raw_data/{1,2,3}/*.mat

    2. Flat preprocessed dir (direct SEED zip extraction):
         root/preprocessed_flat/*.mat   (or root/Preprocessed_EEG/*.mat)
         Files named {subject_id}_{YYYYMMDD}.mat; sessions inferred by date order.
    """
    # Layout 1: session subdirs
    if (root / 'eeg_raw_data').exists():
        return root
    for subdir in ['SEED', 'Preprocessed_EEG']:
        if (root / subdir / 'eeg_raw_data').exists():
            return root / subdir
    # Layout 2: flat mat directory
    for flat in ['preprocessed_flat', 'Preprocessed_EEG', 'SEED_EEG/Preprocessed_EEG']:
        flat_dir = root / flat
        if flat_dir.is_dir() and any(flat_dir.glob('*.mat')):
            return root  # signal to _discover_trials via _flat_dir attribute below
    return root


def _find_flat_dir(root: Path) -> Optional[Path]:
    """Return flat preprocessed dir if layout 2 is used, else None."""
    for flat in ['preprocessed_flat', 'Preprocessed_EEG', 'SEED_EEG/Preprocessed_EEG']:
        flat_dir = root / flat
        if flat_dir.is_dir() and any(flat_dir.glob('*.mat')):
            return flat_dir
    return None


def _get_electrode_positions_mm(root: Optional[Path] = None) -> np.ndarray:
    if root is not None:
        data_root = _find_data_root(root)
        locs_path = data_root / 'channel_62_pos.locs'
        if locs_path.exists():
            return _parse_locs_file(locs_path)
    try:
        import mne
        montage = mne.channels.make_standard_montage('standard_1005')
        pos_dict = montage.get_positions()['ch_pos']
        std_lower = {n.lower(): n for n in pos_dict}
        positions = np.zeros((62, 3), dtype=np.float64)
        for i, ch in enumerate(SEED_CHANNELS):
            key = ch.lower()
            if key in std_lower:
                positions[i] = pos_dict[std_lower[key]] * 1000.0
            elif ch == 'CB1':
                positions[i] = (pos_dict[std_lower['o1']] + pos_dict[std_lower['p7']]) / 2 * 1000.0
                positions[i, 2] -= 5.0
            elif ch == 'CB2':
                positions[i] = (pos_dict[std_lower['o2']] + pos_dict[std_lower['p8']]) / 2 * 1000.0
                positions[i, 2] -= 5.0
        return positions
    except ImportError:
        pass
    return _fallback_electrode_positions_mm()


def _parse_locs_file(locs_path: Path) -> np.ndarray:
    R = 85.0
    positions = []
    with open(locs_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            theta_deg = float(parts[1])
            radius = float(parts[2])
            theta_rad = np.radians(theta_deg)
            r_mm = radius * R / 0.5
            x = r_mm * np.sin(theta_rad)
            y = r_mm * np.cos(theta_rad)
            flat_r = np.sqrt(x**2 + y**2)
            z = np.sqrt(max(0, R**2 - flat_r**2))
            positions.append([x, y, z])
    return np.array(positions, dtype=np.float64)


def _fallback_electrode_positions_mm() -> np.ndarray:
    rng = np.random.default_rng(62)
    phi = rng.uniform(0, 2 * np.pi, 62)
    theta = rng.uniform(0, np.pi / 2, 62)
    r = 85.0
    return np.stack([
        r * np.sin(theta) * np.cos(phi),
        r * np.sin(theta) * np.sin(phi),
        r * np.cos(theta),
    ], axis=1)


class SEEDDataset(Dataset):
    """PyTorch dataset for SEED EEG spatial super-resolution.

    Mirrors the SEEDIVDataset interface exactly — can be used interchangeably.
    Returns (x_ld, x_hd) pairs where LD = random subset of 62 channels.

    Parameters
    ----------
    root : path to the SEED preprocessed dataset root
    sr_factor : super-resolution factor (2, 4, 8)
    split_mode : 'paper80_20' (80/20 trial split) or 'all'
    subset : 'train', 'test', or 'all'
    split_seed : seed for reproducible train/test split
    channel_seed : seed for consistent LD channel selection
    window_size : samples per window (800 = 4 s at 200 Hz, matches paper)
    """

    SAMPLING_RATE = 200  # Hz

    def __init__(
        self,
        root: str | Path,
        sr_factor: int = 4,
        split_mode: str = "all",
        subset: str = "all",
        split_seed: int = 0,
        random_channel_selection: bool = True,
        channel_seed: int = 0,
        constellation: Optional[str] = None,
        fixed_ld_channel_indices: Optional[Sequence[int]] = None,
        normalization: str = "none",
        return_metadata: bool = True,
        time_chunk_size: Optional[int] = None,
        subject: Optional[str] = None,
        session: Optional[int] = None,
        window_size: int = 800,
        window_stride: Optional[int] = None,
        preload: bool = False,         # cache full trial arrays in RAM (~500MB/subject)
        ld_count_rounding: str = "floor",  # 'floor' (default), 'ceil', or 'round'
    ) -> None:
        self.root = Path(root)
        self.sr_factor = sr_factor
        self.ld_count_rounding = ld_count_rounding
        self.split_mode = split_mode
        self.subset = subset
        self.split_seed = split_seed
        self.random_channel_selection = random_channel_selection
        self.channel_seed = channel_seed
        self.constellation = constellation
        self.fixed_ld_channel_indices = (
            None
            if fixed_ld_channel_indices is None
            else np.asarray(fixed_ld_channel_indices, dtype=np.int64)
        )
        self.normalization = normalization
        self.return_metadata = return_metadata
        self.time_chunk_size = time_chunk_size
        self.subject = subject
        self.session = session
        self.window_size = window_size
        self.window_stride = window_stride if window_stride is not None else window_size

        if self.constellation and self.fixed_ld_channel_indices is not None:
            raise ValueError("Use only one of constellation or fixed_ld_channel_indices")
        if self.constellation and self.constellation.lower() not in VALID_PRESETS:
            raise ValueError(
                f"constellation must be one of {VALID_PRESETS}, got {self.constellation!r}"
            )

        self._data_root = _find_data_root(self.root)
        self._flat_dir: Optional[Path] = _find_flat_dir(self.root)
        self._subject_ld_indices: Dict[str, np.ndarray] = {}

        self._trials: List[_TrialInfo] = self._discover_trials()
        if not self._trials:
            raise FileNotFoundError(
                f"No SEED EEG data found under: {self.root}\n"
                f"Tried session-subdir layout: {self._data_root}/eeg_raw_data/{{1,2,3}}/*.mat\n"
                f"Tried flat layout: {self.root}/preprocessed_flat/*.mat\n"
                f"NOTE: raw .cnt files (SEED_Multimodal) are NOT supported."
            )

        self._samples: List[Tuple[int, int]] = self._build_samples()
        self._apply_split()

        # Optional in-memory cache: trial_idx -> full (n_channels, n_samples) array
        self._trial_cache: Optional[List[np.ndarray]] = None
        if preload:
            self._trial_cache = []
            for trial in self._trials:
                key = f'{trial.key_prefix}{trial.trial_idx}'
                mat = sio.loadmat(str(trial.mat_path), variable_names=[key])
                self._trial_cache.append(mat[key].astype(np.float32))

    def _discover_trials(self) -> List[_TrialInfo]:
        if self._flat_dir is not None:
            return self._discover_trials_flat(self._flat_dir)
        raw_dir = self._data_root / 'eeg_raw_data'
        if not raw_dir.exists():
            return []
        return self._discover_trials_session_dirs(raw_dir)

    def _discover_trials_flat(self, flat_dir: Path) -> List[_TrialInfo]:
        """Handle flat layout: {subj}_{YYYYMMDD}.mat, sessions inferred by date order."""
        from collections import defaultdict
        # Group files by subject, sort by date → assign session 1/2/3
        subj_files: Dict[str, list] = defaultdict(list)
        for mat_path in flat_dir.glob('*.mat'):
            parts = mat_path.stem.split('_')
            if len(parts) >= 2 and parts[0].isdigit():
                subj_files[parts[0]].append(mat_path)
        for subj in subj_files:
            subj_files[subj].sort(key=lambda p: p.stem.split('_')[1])  # sort by YYYYMMDD

        trials: List[_TrialInfo] = []
        wanted_sessions = {self.session} if self.session else {1, 2, 3}
        for subj_id, paths in sorted(subj_files.items(), key=lambda x: int(x[0])):
            if self.subject and subj_id != str(self.subject):
                continue
            for sess_idx, mat_path in enumerate(paths, start=1):
                if sess_idx not in wanted_sessions:
                    continue
                try:
                    mat = sio.loadmat(str(mat_path))
                    all_keys = [k for k in mat.keys() if not k.startswith('_')]
                except Exception:
                    continue
                eeg_keys = [k for k in all_keys if 'eeg' in k.lower()]
                if not eeg_keys:
                    continue
                prefix = eeg_keys[0].rstrip('0123456789')
                n_channels = mat[eeg_keys[0]].shape[0]
                labels = SESSION_LABELS[sess_idx]
                for trial_num in range(1, 16):
                    key = f'{prefix}{trial_num}'
                    if key not in mat:
                        continue
                    trials.append(_TrialInfo(
                        subject_id=subj_id,
                        session=sess_idx,
                        trial_idx=trial_num,
                        mat_path=mat_path,
                        key_prefix=prefix,
                        n_channels=n_channels,
                        n_samples=mat[key].shape[1],
                        label=labels[trial_num - 1],
                    ))
        return trials

    def _discover_trials_session_dirs(self, raw_dir: Path) -> List[_TrialInfo]:
        sessions = [self.session] if self.session else [1, 2, 3]
        trials: List[_TrialInfo] = []

        for sess in sessions:
            sess_dir = raw_dir / str(sess)
            if not sess_dir.exists():
                continue

            for mat_path in sorted(sess_dir.glob('*.mat')):
                subj_id = mat_path.stem.split('_')[0]
                if self.subject and subj_id != str(self.subject):
                    continue

                try:
                    mat = sio.loadmat(str(mat_path))
                    all_keys = [k for k in mat.keys() if not k.startswith('_')]
                except Exception:
                    continue

                eeg_keys = [k for k in all_keys if 'eeg' in k.lower()]
                if not eeg_keys:
                    continue

                prefix = eeg_keys[0].rstrip('0123456789')
                n_channels = mat[eeg_keys[0]].shape[0]
                labels = SESSION_LABELS[sess]

                for trial_num in range(1, 16):  # SEED has 15 trials (vs 24 in SEED-IV)
                    key = f'{prefix}{trial_num}'
                    if key not in mat:
                        continue
                    n_samples = mat[key].shape[1]
                    trials.append(_TrialInfo(
                        subject_id=subj_id,
                        session=sess,
                        trial_idx=trial_num,
                        mat_path=mat_path,
                        key_prefix=prefix,
                        n_channels=n_channels,
                        n_samples=n_samples,
                        label=labels[trial_num - 1],
                    ))
        return trials

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples = []
        for trial_idx, trial in enumerate(self._trials):
            for win_start in range(0, trial.n_samples - self.window_size + 1, self.window_stride):
                samples.append((trial_idx, win_start))
        return samples

    def _apply_split(self) -> None:
        if self.split_mode == "all" or self.subset == "all":
            return
        n = len(self._samples)
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(n)
        train_n = int(0.8 * n)
        idx = perm[:train_n] if self.subset == "train" else perm[train_n:]
        self._samples = [self._samples[i] for i in idx]

    def _get_ld_indices(self, subject_id: str) -> np.ndarray:
        if subject_id in self._subject_ld_indices:
            return self._subject_ld_indices[subject_id]
        n_hd = 62
        n_ld = _compute_n_ld(n_hd, self.sr_factor, self.ld_count_rounding)
        hd_indices = np.arange(n_hd, dtype=np.int64)
        if self.fixed_ld_channel_indices is not None:
            fix = self.fixed_ld_channel_indices
            if fix.shape[0] != n_ld:
                raise ValueError(
                    f"fixed_ld_channel_indices length {fix.shape[0]} != n_ld={n_ld}"
                )
            if not np.isin(fix, hd_indices).all():
                raise ValueError("fixed_ld_channel_indices out of range for 62-channel HD")
            ld_indices = np.sort(fix)
        elif self.constellation:
            seed = self._stable_seed(subject_id, self.channel_seed)
            rng = np.random.default_rng(seed)
            ld_indices = indices_for_preset(
                self.constellation.lower(), SEED_CHANNELS, n_ld, rng
            )
        elif not self.random_channel_selection:
            ld_indices = hd_indices[:n_ld].copy()
        else:
            seed = self._stable_seed(subject_id, self.channel_seed)
            rng = np.random.default_rng(seed)
            ld_indices = np.sort(rng.choice(hd_indices, size=n_ld, replace=False))
        self._subject_ld_indices[subject_id] = ld_indices
        return ld_indices

    @staticmethod
    def _stable_seed(text: str, base_seed: int) -> int:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return (int(h, 16) + int(base_seed)) % (2**32)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        trial_idx, win_start = self._samples[index]
        trial = self._trials[trial_idx]

        if self._trial_cache is not None:
            data = self._trial_cache[trial_idx]
        else:
            key = f'{trial.key_prefix}{trial.trial_idx}'
            mat = sio.loadmat(str(trial.mat_path), variable_names=[key])
            data = mat[key]

        win_end = win_start + self.window_size
        epoch = data[:, win_start:win_end].astype(np.float32)

        ld_indices = self._get_ld_indices(trial.subject_id)
        hd = epoch
        ld = epoch[ld_indices]

        if self.normalization == "per_channel_zscore":
            hd = self._zscore_per_channel(hd)
            ld = self._zscore_per_channel(ld)

        hd_t = torch.from_numpy(hd)
        ld_t = torch.from_numpy(ld)

        if not self.return_metadata:
            return ld_t, hd_t

        meta = {
            "subject_id": trial.subject_id,
            "session": trial.session,
            "trial_idx": trial.trial_idx,
            "window_start": win_start,
            "label": trial.label,
            "sr_factor": self.sr_factor,
            "sampling_frequency": self.SAMPLING_RATE,
            "hd_indices": torch.arange(62),
            "ld_indices": torch.from_numpy(ld_indices.copy()),
        }
        return {"x_ld": ld_t, "x_hd": hd_t, "meta": meta}

    @staticmethod
    def _zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = x.mean(axis=1, keepdims=True)
        sigma = x.std(axis=1, keepdims=True)
        return (x - mu) / np.maximum(sigma, eps)

    @staticmethod
    def list_subjects(root: str | Path) -> List[str]:
        root = Path(root)
        subjects = set()
        flat_dir = _find_flat_dir(root)
        if flat_dir is not None:
            for mat_file in flat_dir.glob('*.mat'):
                parts = mat_file.stem.split('_')
                if len(parts) >= 2 and parts[0].isdigit():
                    subjects.add(parts[0])
        else:
            data_root = _find_data_root(root)
            raw_dir = data_root / 'eeg_raw_data'
            if raw_dir.exists():
                for sess_dir in raw_dir.iterdir():
                    if not sess_dir.is_dir():
                        continue
                    for mat_file in sess_dir.glob('*.mat'):
                        subj_id = mat_file.stem.split('_')[0]
                        subjects.add(subj_id)
        return sorted(subjects, key=lambda s: int(s) if s.isdigit() else s)

    @staticmethod
    def get_electrode_positions_mm(root: Optional[str | Path] = None) -> np.ndarray:
        r = Path(root) if root is not None else None
        return _get_electrode_positions_mm(r)

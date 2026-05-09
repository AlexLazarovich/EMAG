"""
SEED-IV EEG Spatial Super-Resolution Dataset.

Loads SEED-IV raw EEG data (62-channel, 200Hz) and provides
(x_ld, x_hd) pairs for spatial super-resolution, matching the
Localize-MI dataloader interface.

Dataset structure (official zip extracts to SEED_IV/ subdirectory):
    SEED_IV/eeg_raw_data/{session}/{subjectid_date}.mat
    - Keys: {initials}_eeg1 ... {initials}_eeg24  (24 trials, prefix varies per subject)
    - Each: (62, T) array, T varies by trial

Subjects: 1-15, Sessions: 1-3, 24 trials per session.
Sampling rate: 200 Hz (downsampled from 1000 Hz).
62 channels in standard 10-20 montage.
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


# Standard 10-20 channel order for SEED-IV (from Channel Order.xlsx)
SEED_IV_CHANNELS = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2',
]

# Session-level trial emotion labels from ReadMe.txt
# 0=neutral, 1=sad, 2=fear, 3=happy
SESSION_LABELS = {
    1: [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],
    2: [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],
    3: [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],
}


@dataclass
class _TrialInfo:
    """Metadata for one trial."""
    subject_id: str
    session: int
    trial_idx: int       # 1-based (matches {prefix}{trial_idx} key)
    mat_path: Path
    key_prefix: str      # e.g. 'cz_eeg', 'ha_eeg' — varies per subject
    n_channels: int
    n_samples: int
    label: int           # emotion label 0-3


def _get_electrode_positions_mm(root: Optional[Path] = None) -> np.ndarray:
    """
    Return (62, 3) array of electrode positions in mm.

    Parses channel_62_pos.locs from the dataset if available,
    otherwise falls back to MNE standard_1005 montage.
    """
    # Try to load from the dataset's own locs file
    if root is not None:
        locs_path = _find_data_root(root) / 'channel_62_pos.locs'
        if locs_path.exists():
            return _parse_locs_file(locs_path)

    # Fallback: MNE standard montage
    try:
        import mne
        montage = mne.channels.make_standard_montage('standard_1005')
        pos_dict = montage.get_positions()['ch_pos']
        std_lower = {n.lower(): n for n in pos_dict}

        positions = np.zeros((62, 3), dtype=np.float64)
        for i, ch in enumerate(SEED_IV_CHANNELS):
            key = ch.lower()
            if key in std_lower:
                positions[i] = pos_dict[std_lower[key]] * 1000.0
            elif ch == 'CB1':
                o1 = pos_dict[std_lower['o1']] * 1000.0
                p7 = pos_dict[std_lower['p7']] * 1000.0
                positions[i] = (o1 + p7) / 2
                positions[i, 2] -= 5.0
            elif ch == 'CB2':
                o2 = pos_dict[std_lower['o2']] * 1000.0
                p8 = pos_dict[std_lower['p8']] * 1000.0
                positions[i] = (o2 + p8) / 2
                positions[i, 2] -= 5.0
        return positions
    except ImportError:
        pass

    return _fallback_electrode_positions_mm()


def _parse_locs_file(locs_path: Path) -> np.ndarray:
    """Parse .locs file (polar coords) into (62, 3) Cartesian positions in mm."""
    R = 85.0  # head radius in mm
    positions = []
    with open(locs_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            theta_deg = float(parts[1])
            radius = float(parts[2])
            # Convert polar to 3D Cartesian (EEG convention)
            theta_rad = np.radians(theta_deg)
            r_mm = radius * R / 0.5  # 0.5 is the normalized head edge
            x = r_mm * np.sin(theta_rad)
            y = r_mm * np.cos(theta_rad)
            flat_r = np.sqrt(x**2 + y**2)
            z = np.sqrt(max(0, R**2 - flat_r**2))
            positions.append([x, y, z])
    return np.array(positions, dtype=np.float64)


def _find_data_root(root: Path) -> Path:
    """Resolve the actual data root, handling the SEED_IV/ subdirectory."""
    # The official zip extracts to a SEED_IV/ subdirectory
    if (root / 'SEED_IV' / 'eeg_raw_data').exists():
        return root / 'SEED_IV'
    if (root / 'eeg_raw_data').exists():
        return root
    # Check for SEED_IV subdir without eeg_raw_data (shouldn't happen but be safe)
    if (root / 'SEED_IV').exists():
        return root / 'SEED_IV'
    return root


def _fallback_electrode_positions_mm() -> np.ndarray:
    """Fallback positions if MNE is not available — unit sphere × 85mm radius."""
    rng = np.random.default_rng(62)
    # Place on upper hemisphere of a sphere
    phi = rng.uniform(0, 2 * np.pi, 62)
    theta = rng.uniform(0, np.pi / 2, 62)
    r = 85.0
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return np.stack([x, y, z], axis=1)


class SEEDIVDataset(Dataset):
    """PyTorch dataset for SEED-IV EEG spatial super-resolution.

    Matches the Localize-MI dataloader interface:
      - Returns (x_ld, x_hd) tensors for spatial super-resolution
      - LD = subset of HD channels, selected by sr_factor
      - Supports per-subject filtering, train/test split, time chunking

    Differences from Localize-MI:
      - 62 channels (scalp EEG) vs 256 (sEEG)
      - 200 Hz vs 8000 Hz
      - Variable-length trials segmented into fixed-size windows
      - 15 subjects × 3 sessions × 24 trials
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
        window_size: int = 800,        # 4 seconds at 200 Hz (matches paper)
        window_stride: Optional[int] = None,  # defaults to window_size (no overlap)
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

        self._validate_args()

        # Resolve actual data root (handles SEED_IV/ subdirectory)
        self._data_root = _find_data_root(self.root)

        self._trials: List[_TrialInfo] = self._discover_trials()
        if not self._trials:
            raise FileNotFoundError(
                f"No SEED-IV EEG data found under: {self.root}"
            )

        # Pre-compute LD channel indices (consistent per subject)
        self._subject_ld_indices: Dict[str, np.ndarray] = {}

        # Build (trial_idx, window_start) sample list
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

    def _validate_args(self) -> None:
        if self.sr_factor <= 0:
            raise ValueError("sr_factor must be > 0")
        valid_split_modes = {"all", "paper80_20"}
        if self.split_mode not in valid_split_modes:
            raise ValueError(f"split_mode '{self.split_mode}' not in {valid_split_modes}")
        valid_subset = {"all", "train", "test"}
        if self.subset not in valid_subset:
            raise ValueError(f"subset '{self.subset}' not in {valid_subset}")
        valid_norm = {"none", "per_channel_zscore"}
        if self.normalization not in valid_norm:
            raise ValueError(f"normalization '{self.normalization}' not in {valid_norm}")
        if self.session is not None and self.session not in (1, 2, 3):
            raise ValueError(f"session must be 1, 2, or 3, got {self.session}")
        if self.window_size <= 0:
            raise ValueError("window_size must be > 0")
        if self.constellation and self.fixed_ld_channel_indices is not None:
            raise ValueError("Use only one of constellation or fixed_ld_channel_indices")
        if self.constellation:
            if self.constellation.lower() not in VALID_PRESETS:
                raise ValueError(
                    f"constellation must be one of {VALID_PRESETS}, got {self.constellation!r}"
                )

    def _discover_trials(self) -> List[_TrialInfo]:
        """Scan eeg_raw_data and build trial metadata list."""
        raw_dir = self._data_root / 'eeg_raw_data'
        if not raw_dir.exists():
            return []

        sessions = [self.session] if self.session else [1, 2, 3]
        trials: List[_TrialInfo] = []

        for sess in sessions:
            sess_dir = raw_dir / str(sess)
            if not sess_dir.exists():
                continue

            mat_files = sorted(sess_dir.glob('*.mat'))
            for mat_path in mat_files:
                # Filename format: {subject_id}_{date}.mat
                subj_id = mat_path.stem.split('_')[0]

                if self.subject and subj_id != str(self.subject):
                    continue

                # Detect key prefix by loading variable names
                try:
                    mat = sio.loadmat(str(mat_path))
                    all_keys = [k for k in mat.keys() if not k.startswith('_')]
                except Exception:
                    continue

                # Find the key prefix (e.g. 'cz_eeg', 'ha_eeg')
                # Keys look like '{initials}_eeg1', '{initials}_eeg2', ...
                eeg_keys = [k for k in all_keys if 'eeg' in k.lower()]
                if not eeg_keys:
                    continue

                # Extract prefix from first key: 'cz_eeg1' -> 'cz_eeg'
                sample_key = eeg_keys[0]
                # Strip trailing digits to get prefix
                prefix = sample_key.rstrip('0123456789')
                n_channels = mat[sample_key].shape[0]

                labels = SESSION_LABELS[sess]
                for trial_num in range(1, 25):
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
        """Create (trial_index, window_start) pairs by windowing each trial."""
        samples: List[Tuple[int, int]] = []
        for trial_idx, trial in enumerate(self._trials):
            for win_start in range(0, trial.n_samples - self.window_size + 1,
                                   self.window_stride):
                samples.append((trial_idx, win_start))
        return samples

    def _apply_split(self) -> None:
        if self.split_mode == "all" or self.subset == "all":
            return

        n = len(self._samples)
        rng = np.random.default_rng(self.split_seed)
        perm = rng.permutation(n)
        train_n = int(0.8 * n)

        if self.subset == "train":
            idx = perm[:train_n]
        else:
            idx = perm[train_n:]

        self._samples = [self._samples[i] for i in idx]

    def _get_ld_indices(self, subject_id: str) -> np.ndarray:
        """Return LD channel indices for a subject (cached, consistent across sessions)."""
        if subject_id in self._subject_ld_indices:
            return self._subject_ld_indices[subject_id]

        n_hd = 62  # all channels always valid for SEED-IV
        from .seed_dataloader import _compute_n_ld
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
                self.constellation.lower(), SEED_IV_CHANNELS, n_ld, rng
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
            data = mat[key]  # (62, T_full)

        # Extract window
        win_end = win_start + self.window_size
        epoch = data[:, win_start:win_end].astype(np.float32)  # (62, W)

        # HD = all channels, LD = subset
        ld_indices = self._get_ld_indices(trial.subject_id)
        hd = epoch  # (62, W)
        ld = epoch[ld_indices]  # (n_ld, W)

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
        return {
            "x_ld": ld_t,
            "x_hd": hd_t,
            "meta": meta,
        }

    @staticmethod
    def _zscore_per_channel(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        mu = x.mean(axis=1, keepdims=True)
        sigma = x.std(axis=1, keepdims=True)
        return (x - mu) / np.maximum(sigma, eps)

    @staticmethod
    def list_subjects(root: str | Path) -> List[str]:
        """Return sorted list of subject IDs with raw EEG data."""
        root = Path(root)
        data_root = _find_data_root(root)
        raw_dir = data_root / 'eeg_raw_data'
        subjects = set()
        for sess_dir in raw_dir.iterdir():
            if not sess_dir.is_dir():
                continue
            for mat_file in sess_dir.glob('*.mat'):
                subj_id = mat_file.stem.split('_')[0]
                subjects.add(subj_id)
        return sorted(subjects, key=lambda s: int(s) if s.isdigit() else s)

    @staticmethod
    def get_electrode_positions_mm(root: Optional[str | Path] = None) -> np.ndarray:
        """Return (62, 3) electrode positions in mm."""
        r = Path(root) if root is not None else None
        return _get_electrode_positions_mm(r)


if __name__ == "__main__":
    import os, sys
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    elif "DATA_ROOT" in os.environ:
        dataset_root = os.environ["DATA_ROOT"]
    else:
        raise SystemExit("Usage: python seed_iv_dataloader.py <dataset_root>")
    subjects = SEEDIVDataset.list_subjects(dataset_root)
    print(f"Available subjects: {subjects}")

    for subj in subjects[:2]:  # Quick test on first 2
        ds = SEEDIVDataset(
            root=dataset_root,
            sr_factor=4,
            split_mode="paper80_20",
            subset="train",
            subject=subj,
            normalization="per_channel_zscore",
            return_metadata=False,
        )
        sample_ld, sample_hd = ds[0]
        print(f"Subject {subj}: len={len(ds)}, x_ld={tuple(sample_ld.shape)}, "
              f"x_hd={tuple(sample_hd.shape)}")

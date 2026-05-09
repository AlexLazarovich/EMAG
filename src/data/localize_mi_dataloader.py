from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class _RunInfo:
    run_id: str
    npy_path: Path
    channels_path: Path
    epochs_tsv_path: Path
    hd_indices: np.ndarray
    ld_indices: np.ndarray
    n_epochs: int
    sampling_frequency: float
    epoch_duration_s: float


class LocalizeMIDataset(Dataset):
    """PyTorch dataset for Localize-MI EEG spatial super-resolution.

    Paper-aligned defaults used here:
    - Keep Localize-MI epoch windows as-is (260 ms, 2081 samples at 8 kHz).
    - Support SR factors 2x/4x/8x/16x as parameters.
    - Support random LD channel selection while keeping one fixed selection per run.

    Notes:
    - `split_mode='all'` is useful for overfit-style experiments.
    - `split_mode='paper80_20'` reproduces a random 80/20 trial split behavior.
    - `time_chunk_size`: If set, returns data chunked along timestep dimension.
      Useful for memory-efficient processing of full epochs (2081 timesteps).
      When used with batch_size=1 in DataLoader, processes one epoch one time-chunk at a time.
    """

    def __init__(
        self,
        root: str | Path,
        sr_factor: int = 4,
        split_mode: str = "all",
        subset: str = "all",
        split_seed: int = 0,
        random_channel_selection: bool = True,
        consistent_channels_within_recording: bool = True,
        channel_seed: int = 0,
        fixed_ld_channel_indices: Optional[Sequence[int]] = None,
        bad_channel_policy: str = "keep_all",
        normalization: str = "none",
        return_metadata: bool = True,
        time_chunk_size: Optional[int] = None,
        subject: Optional[str] = None,
        ld_count_rounding: str = "floor",
    ) -> None:
        self.root = Path(root)
        self.sr_factor = sr_factor
        self.ld_count_rounding = ld_count_rounding
        self.split_mode = split_mode
        self.subset = subset
        self.split_seed = split_seed
        self.random_channel_selection = random_channel_selection
        self.consistent_channels_within_recording = consistent_channels_within_recording
        self.channel_seed = channel_seed
        self.fixed_ld_channel_indices = (
            None
            if fixed_ld_channel_indices is None
            else np.asarray(fixed_ld_channel_indices, dtype=np.int64)
        )
        self.bad_channel_policy = bad_channel_policy
        self.normalization = normalization
        self.return_metadata = return_metadata
        self.time_chunk_size = time_chunk_size
        self.subject = subject

        self._validate_args()

        self._runs: List[_RunInfo] = self._discover_runs()
        if not self._runs:
            raise FileNotFoundError(
                f"No Localize-MI epoch files found under: {self.root}"
            )

        # Don't load arrays into memory - store paths and load on-demand
        # This keeps memory footprint minimal and only loads what's needed

        self._samples: List[Tuple[int, int]] = self._build_samples()
        self._apply_split()

    def _validate_args(self) -> None:
        if self.sr_factor <= 0:
            raise ValueError("sr_factor must be > 0")

        valid_split_modes = {"all", "paper80_20"}
        if self.split_mode not in valid_split_modes:
            raise ValueError(
                f"Unsupported split_mode '{self.split_mode}'. "
                f"Choose one of {sorted(valid_split_modes)}"
            )

        valid_subset = {"all", "train", "test"}
        if self.subset not in valid_subset:
            raise ValueError(
                f"Unsupported subset '{self.subset}'. "
                f"Choose one of {sorted(valid_subset)}"
            )

        valid_bad_channel_policy = {"keep_all", "drop_bad"}
        if self.bad_channel_policy not in valid_bad_channel_policy:
            raise ValueError(
                f"Unsupported bad_channel_policy '{self.bad_channel_policy}'. "
                f"Choose one of {sorted(valid_bad_channel_policy)}"
            )

        valid_norm = {"none", "per_channel_zscore"}
        if self.normalization not in valid_norm:
            raise ValueError(
                f"Unsupported normalization '{self.normalization}'. "
                f"Choose one of {sorted(valid_norm)}"
            )

    def _discover_runs(self) -> List[_RunInfo]:
        # Filter to a specific subject if requested
        if self.subject:
            glob_pattern = f"{self.subject}/eeg/*_epochs.npy"
        else:
            glob_pattern = "sub-*/eeg/*_epochs.npy"
        run_files = sorted(self.root.glob(glob_pattern))
        runs: List[_RunInfo] = []

        # Extract subject ID for consistent channel selection across runs
        subject_id = self.subject  # may be None if not filtering

        for npy_path in run_files:
            prefix = str(npy_path).replace("_epochs.npy", "")
            channels_path = Path(prefix + "_channels.tsv")
            epochs_tsv_path = Path(prefix + "_epochs.tsv")

            if not channels_path.exists() or not epochs_tsv_path.exists():
                continue

            try:
                arr = np.load(npy_path, mmap_mode="r")
            except ValueError:
                # Skip corrupt files (e.g. truncated npy)
                continue
            if arr.ndim != 3:
                continue

            n_epochs, n_channels, _ = arr.shape
            channels_df = pd.read_csv(channels_path, sep="\t")
            if len(channels_df) != n_channels:
                continue

            hd_indices = self._select_hd_indices(channels_df)
            if len(hd_indices) == 0:
                continue

            # Use subject ID for channel seed when consistent_channels_within_recording
            # so ALL runs of the same subject share identical LD channels
            if self.consistent_channels_within_recording and subject_id:
                seed_key = subject_id
            else:
                seed_key = npy_path.stem
            from .seed_dataloader import _compute_n_ld
            n_ld_expect = _compute_n_ld(len(hd_indices), self.sr_factor, self.ld_count_rounding)
            if self.fixed_ld_channel_indices is not None:
                fix = self.fixed_ld_channel_indices
                if fix.shape[0] != n_ld_expect:
                    raise ValueError(
                        f"fixed_ld_channel_indices length {fix.shape[0]} != n_ld={n_ld_expect} "
                        f"(sr_factor={self.sr_factor}, n_hd={len(hd_indices)})"
                    )
                if not np.isin(fix, hd_indices).all():
                    raise ValueError(
                        "fixed_ld_channel_indices must be a subset of HD indices for this run"
                    )
                ld_indices = np.sort(fix).astype(np.int64)
            else:
                ld_indices = self._sample_ld_indices(run_id=seed_key, hd_indices=hd_indices)

            epoch_df = pd.read_csv(epochs_tsv_path, sep="\t")
            duration = float(epoch_df["duration"].iloc[0]) if "duration" in epoch_df else np.nan
            sfreq = (
                float(channels_df["sampling_frequency"].iloc[0])
                if "sampling_frequency" in channels_df
                else np.nan
            )

            runs.append(
                _RunInfo(
                    run_id=npy_path.stem,
                    npy_path=npy_path,
                    channels_path=channels_path,
                    epochs_tsv_path=epochs_tsv_path,
                    hd_indices=hd_indices,
                    ld_indices=ld_indices,
                    n_epochs=n_epochs,
                    sampling_frequency=sfreq,
                    epoch_duration_s=duration,
                )
            )

        return runs

    def _select_hd_indices(self, channels_df: pd.DataFrame) -> np.ndarray:
        all_indices = np.arange(len(channels_df), dtype=np.int64)
        if self.bad_channel_policy == "keep_all" or "status" not in channels_df.columns:
            return all_indices

        is_good = channels_df["status"].astype(str).str.lower().eq("good").to_numpy()
        return all_indices[is_good]

    def _sample_ld_indices(self, run_id: str, hd_indices: np.ndarray) -> np.ndarray:
        from .seed_dataloader import _compute_n_ld
        n_hd = len(hd_indices)
        n_ld = _compute_n_ld(n_hd, self.sr_factor, self.ld_count_rounding)
        if n_ld > n_hd:
            raise ValueError(f"Invalid SR factor {self.sr_factor} for n_hd={n_hd}")

        if not self.random_channel_selection:
            return hd_indices[:n_ld].copy()

        seed = self._stable_seed(run_id, self.channel_seed)
        rng = np.random.default_rng(seed)
        chosen = np.sort(rng.choice(hd_indices, size=n_ld, replace=False))
        return chosen.astype(np.int64)

    @staticmethod
    def _stable_seed(text: str, base_seed: int) -> int:
        h = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
        return (int(h, 16) + int(base_seed)) % (2**32)

    def _build_samples(self) -> List[Tuple[int, int]]:
        samples: List[Tuple[int, int]] = []
        for run_idx, run in enumerate(self._runs):
            for epoch_idx in range(run.n_epochs):
                samples.append((run_idx, epoch_idx))
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

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int):
        run_idx, epoch_idx = self._samples[index]
        run = self._runs[run_idx]
        
        # Load only this specific epoch on-demand using memmap (lazy loading)
        arr = np.load(run.npy_path, mmap_mode="r")
        epoch = arr[epoch_idx]  # [channels, time]
        hd = epoch[run.hd_indices].astype(np.float32, copy=False)

        if self.consistent_channels_within_recording:
            ld_idx = run.ld_indices
        else:
            ld_idx = self._sample_ld_indices(
                run_id=f"{run.run_id}#{epoch_idx}",
                hd_indices=run.hd_indices,
            )

        ld = epoch[ld_idx].astype(np.float32, copy=False)

        if self.normalization == "per_channel_zscore":
            hd = self._zscore_per_channel(hd)
            ld = self._zscore_per_channel(ld)

        # If time chunking is enabled, extract a chunk and return info about the full epoch
        if self.time_chunk_size is not None:
            # This is a simplified approach: return chunk indices so caller can slice
            # Return metadata to help caller manage chunking
            hd_t = torch.from_numpy(hd)
            ld_t = torch.from_numpy(ld)

            if not self.return_metadata:
                return ld_t, hd_t

            meta = {
                "run_id": run.run_id,
                "epoch_idx": epoch_idx,
                "sr_factor": self.sr_factor,
                "sampling_frequency": run.sampling_frequency,
                "epoch_duration_s": run.epoch_duration_s,
                "hd_indices": torch.from_numpy(run.hd_indices.copy()),
                "ld_indices": torch.from_numpy(ld_idx.copy()),
                "time_chunk_size": self.time_chunk_size,
                "n_time_steps": hd_t.shape[-1],
            }
            return {
                "x_ld": ld_t,
                "x_hd": hd_t,
                "meta": meta,
            }

        hd_t = torch.from_numpy(hd)
        ld_t = torch.from_numpy(ld)

        if not self.return_metadata:
            return ld_t, hd_t

        meta = {
            "run_id": run.run_id,
            "epoch_idx": epoch_idx,
            "sr_factor": self.sr_factor,
            "sampling_frequency": run.sampling_frequency,
            "epoch_duration_s": run.epoch_duration_s,
            "hd_indices": torch.from_numpy(run.hd_indices.copy()),
            "ld_indices": torch.from_numpy(ld_idx.copy()),
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
        """Return sorted list of subject IDs that have epoch data."""
        root = Path(root)
        subjects = set()
        for p in root.glob("sub-*/eeg/*_epochs.npy"):
            # Extract subject ID from path: root/sub-XX/eeg/...
            subjects.add(p.parent.parent.name)
        return sorted(subjects)

    @staticmethod
    def chunk_time_data(
        x_ld: torch.Tensor,
        x_hd: torch.Tensor,
        chunk_size: int,
        n_time_steps: int,
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Chunk time-domain data along the time dimension for memory-efficient processing.
        
        Args:
            x_ld: Low-density EEG, shape (n_ld_channels, n_time_steps)
            x_hd: High-density EEG, shape (n_hd_channels, n_time_steps)
            chunk_size: Number of timesteps per chunk
            n_time_steps: Total number of timesteps (for reference)
        
        Returns:
            List of (x_ld_chunk, x_hd_chunk) tuples, each chunk has shape (n_channels, chunk_size)
        """
        chunks = []
        for t_start in range(0, n_time_steps, chunk_size):
            t_end = min(t_start + chunk_size, n_time_steps)
            x_ld_chunk = x_ld[:, t_start:t_end]  # (n_ld_channels, actual_chunk_size)
            x_hd_chunk = x_hd[:, t_start:t_end]  # (n_hd_channels, actual_chunk_size)
            chunks.append((x_ld_chunk, x_hd_chunk))
        return chunks


if __name__ == "__main__":
    import os, sys
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    elif "DATA_ROOT" in os.environ:
        dataset_root = os.environ["DATA_ROOT"]
    else:
        raise SystemExit("Usage: python localize_mi_dataloader.py <dataset_root>")
    subjects = LocalizeMIDataset.list_subjects(dataset_root)
    print(f"Available subjects: {subjects}")

    for subj in subjects:
        ds = LocalizeMIDataset(
            root=dataset_root,
            sr_factor=4,
            split_mode="paper80_20",
            subset="train",
            subject=subj,
            random_channel_selection=True,
            consistent_channels_within_recording=True,
            bad_channel_policy="keep_all",
            normalization="none",
        )
        sample = ds[0]
        print(f"{subj}: len={len(ds)}, x_ld={tuple(sample['x_ld'].shape)}, "
              f"x_hd={tuple(sample['x_hd'].shape)}")

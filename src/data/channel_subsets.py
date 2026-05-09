"""Named LD-channel subsets for SEED / SEED-IV electrode-subset experiments.

Each subset ID maps to a list of channel names from the canonical 62-channel
10–20 order used by `SEED_CHANNELS` / `SEED_IV_CHANNELS` (which are identical).
Subset sizes are strictly `62 // sr_factor` (15 for SR=4, 7 for SR=8, 31 for SR=2).

Public API:
    - `SUBSETS`            : dict[subset_id -> list[str]]
    - `SHORT_NAMES`        : dict[subset_id -> short run-name suffix]
    - `resolve_subset(id)` : returns np.ndarray of sorted global LD indices
    - `expected_size(id)`  : returns int
    - `all_subset_ids()`   : iterable of known IDs
    - `subsets_for_sr(sr)` : list of IDs valid for a given SR factor (62 // sr)

Validation is performed at import time against `SEED_IV_CHANNELS`.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

# Allow `utils.channel_subsets` to work both when imported via `utils.` package
# and when the caller has added `utils/` to sys.path (training scripts do the
# latter via `sys.path.insert(0, str(root / 'utils'))`).
_UTILS_DIR = Path(__file__).resolve().parent
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

# Canonical 62-channel 10–20 order. MUST match `SEED_IV_CHANNELS` in
# `seed_iv_dataloader.py` and `SEED_CHANNELS` in `seed_dataloader.py`. Inlined
# here so this module is importable without the heavy scipy/mne dependencies of
# the dataloaders. Consistency is asserted at import time when the dataloaders
# are importable (otherwise the check is skipped silently).
SEED_CHANNEL_NAMES: List[str] = [
    'FP1', 'FPZ', 'FP2', 'AF3', 'AF4', 'F7', 'F5', 'F3', 'F1', 'FZ',
    'F2', 'F4', 'F6', 'F8', 'FT7', 'FC5', 'FC3', 'FC1', 'FCZ', 'FC2',
    'FC4', 'FC6', 'FT8', 'T7', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4',
    'C6', 'T8', 'TP7', 'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
    'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
    'PO7', 'PO5', 'PO3', 'POZ', 'PO4', 'PO6', 'PO8', 'CB1', 'O1', 'OZ',
    'O2', 'CB2',
]


def _channel_names_to_indices(names, channel_order, n_ld):
    """Tiny local copy of utils.eeg_constellations.channel_names_to_indices,
    avoids the circular import path `utils.eeg_constellations` which is fine
    but lets this module import with zero third-party deps."""
    order_map = {c.upper(): i for i, c in enumerate(channel_order)}
    out: List[int] = []
    for name in names:
        key = name.upper()
        if key in order_map and order_map[key] not in out:
            out.append(order_map[key])
        if len(out) >= n_ld:
            break
    if len(out) < n_ld:
        raise ValueError(
            f"Constellation could only resolve {len(out)} of {n_ld} channels "
            f"(check channel names vs dataset order)."
        )
    return np.array(out, dtype=np.int64)


# Subsets for SR=4 (K=15)
_SUBSETS_15: Dict[str, List[str]] = {
    # Circumference-approximation (Valderrama et al., minus PO8).
    "V15_approx": [
        "FP1", "FP2", "F7", "F8",
        "FT7", "FT8", "T7", "T8",
        "TP7", "TP8", "P7", "P8",
        "PO7", "O1", "O2",
    ],
    # Fronto-temporal (classical affective EEG regions).
    "FT15_approx": [
        "FP1", "FP2", "F3", "F4",
        "F7", "F8", "FC5", "FC6",
        "FT7", "FT8", "T7", "T8",
        "TP7", "TP8", "P7",
    ],
    # Interior / midline control (avoids the rim).
    "INT15": [
        "AF3", "AF4", "F1", "F2", "FZ",
        "FC1", "FC2",
        "C1", "C2", "CZ",
        "CP1", "CP2",
        "P1", "P2", "PZ",
    ],
}

# Subsets for SR=8 (K=7)
_SUBSETS_7: Dict[str, List[str]] = {
    "VL7_approx":  ["FP1", "F7", "FT7", "T7", "TP7", "P7", "O1"],
    "VR7_approx":  ["FP2", "F8", "FT8", "T8", "TP8", "P8", "O2"],
    "VU7_approx":  ["FP1", "FP2", "F7", "F8", "FT7", "FT8", "T7"],
    "VLw7_approx": ["TP7", "TP8", "P7", "P8", "PO7", "O1", "O2"],
    "INT7":        ["FZ", "FC1", "FC2", "CZ", "CP1", "CP2", "PZ"],
}

# Subsets for SR=2 (K=31) — left / right hemisphere + selected midline.
# Only uses channels that actually exist in the 62-ch cap (spec references
# AF7/AF8/FT9/FT10/POz which we map to available canonical names).
# Left-hemisphere channels (name ends in odd number, 5, or 7) + midline FZ/CZ/PZ/OZ.
_SUBSETS_31: Dict[str, List[str]] = {
    "HEMI_LEFT31": [
        # Left frontal/peri-frontal
        "FP1", "AF3", "F7", "F5", "F3", "F1",
        "FT7", "FC5", "FC3", "FC1",
        # Left central
        "T7", "C5", "C3", "C1",
        # Left centro-parietal
        "TP7", "CP5", "CP3", "CP1",
        # Left parietal
        "P7", "P5", "P3", "P1",
        # Left parieto-occipital / occipital / CB
        "PO7", "PO5", "PO3", "CB1", "O1",
        # Midline
        "FZ", "CZ", "PZ", "OZ",
    ],
    "HEMI_RIGHT31": [
        "FP2", "AF4", "F8", "F6", "F4", "F2",
        "FT8", "FC6", "FC4", "FC2",
        "T8", "C6", "C4", "C2",
        "TP8", "CP6", "CP4", "CP2",
        "P8", "P6", "P4", "P2",
        "PO8", "PO6", "PO4", "CB2", "O2",
        "FZ", "CZ", "PZ", "OZ",
    ],
}

SUBSETS: Dict[str, List[str]] = {**_SUBSETS_15, **_SUBSETS_7, **_SUBSETS_31}

# Short, unambiguous suffixes for run names.
SHORT_NAMES: Dict[str, str] = {
    "V15_approx":   "subV15",
    "FT15_approx":  "subFT15",
    "INT15":        "subINT15",
    "VL7_approx":   "subVL7",
    "VR7_approx":   "subVR7",
    "VU7_approx":   "subVU7",
    "VLw7_approx":  "subVLw7",
    "INT7":         "subINT7",
    "HEMI_LEFT31":  "subHEMIL",
    "HEMI_RIGHT31": "subHEMIR",
}

# Map SR factor -> expected LD size -> subset IDs.
_SR_TO_IDS: Dict[int, List[str]] = {
    2: list(_SUBSETS_31.keys()),
    4: list(_SUBSETS_15.keys()),
    8: list(_SUBSETS_7.keys()),
}


def all_subset_ids() -> List[str]:
    return list(SUBSETS.keys())


def subsets_for_sr(sr_factor: int) -> List[str]:
    return list(_SR_TO_IDS.get(int(sr_factor), []))


def expected_size(subset_id: str) -> int:
    names = SUBSETS[subset_id]
    return len(names)


def short_name(subset_id: str) -> str:
    return SHORT_NAMES[subset_id]


def resolve_subset(subset_id: str, channel_order: List[str] = SEED_CHANNEL_NAMES) -> np.ndarray:
    """Return sorted int64 global indices of `subset_id` in `channel_order`.

    Raises ValueError if any name is not present in `channel_order`.
    """
    if subset_id not in SUBSETS:
        raise ValueError(
            f"Unknown subset_id {subset_id!r}. Known: {list(SUBSETS)}"
        )
    names = SUBSETS[subset_id]
    idx = _channel_names_to_indices(names, channel_order, len(names))
    return np.sort(np.asarray(idx, dtype=np.int64))


# ── Import-time validation ───────────────────────────────────────────────────
def _validate() -> None:
    order = [c.upper() for c in SEED_CHANNEL_NAMES]
    if len(order) != 62:
        raise RuntimeError(f"SEED_CHANNEL_NAMES has {len(order)} entries, expected 62")
    order_set = set(order)
    for sid, names in SUBSETS.items():
        missing = [n for n in names if n.upper() not in order_set]
        if missing:
            raise RuntimeError(
                f"Subset {sid!r} references unknown channels: {missing}."
            )
        upper = [n.upper() for n in names]
        if len(set(upper)) != len(upper):
            dupes = [n for n in set(upper) if upper.count(n) > 1]
            raise RuntimeError(f"Subset {sid!r} has duplicate channels: {dupes}")
        size = len(names)
        if size not in (7, 15, 31):
            raise RuntimeError(
                f"Subset {sid!r} size {size} is not one of {{7, 15, 31}}"
            )
    if set(SHORT_NAMES) != set(SUBSETS):
        diff = set(SHORT_NAMES) ^ set(SUBSETS)
        raise RuntimeError(f"SHORT_NAMES vs SUBSETS mismatch: {diff}")

    # Opportunistic consistency check against the dataloaders. Silently skipped
    # if scipy / the dataloaders aren't importable (e.g. in minimal envs).
    try:
        from .seed_iv_dataloader import SEED_IV_CHANNELS  # type: ignore
        if [c.upper() for c in SEED_IV_CHANNELS] != order:
            raise RuntimeError(
                "SEED_CHANNEL_NAMES does not match SEED_IV_CHANNELS — update "
                "utils/channel_subsets.py::SEED_CHANNEL_NAMES."
            )
    except ImportError:
        pass
    try:
        from .seed_dataloader import SEED_CHANNELS  # type: ignore
        if [c.upper() for c in SEED_CHANNELS] != order:
            raise RuntimeError(
                "SEED_CHANNEL_NAMES does not match SEED_CHANNELS — update "
                "utils/channel_subsets.py::SEED_CHANNEL_NAMES."
            )
    except ImportError:
        pass


_validate()

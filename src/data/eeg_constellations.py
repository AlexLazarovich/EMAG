"""
Fixed LD channel constellations for SEED / SEED-IV (62-channel 10–20 order).

Used for reproducible electrode-subset ablations (paper: circumference, fronto-temporal,
interior, vs random). Names must match `SEED_IV_CHANNELS` / `SEED_CHANNELS` order indices.
"""
from __future__ import annotations

from typing import List, Sequence

# Circumference-heavy set (paper cites perimeter channels for emotion decoding).
CIRCUMFERENCE_NAMES: List[str] = [
    "FP1", "FP2", "F7", "F8", "T7", "T8", "P7", "P8", "O1", "O2",
    "AF3", "AF4", "FT7", "FT8", "TP7", "TP8",
]

# Fronto-temporal (affective regions, lateral)
FRONTO_TEMPORAL_NAMES: List[str] = [
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
]

# Interior / midline / centro-parietal (avoid rim)
INTERIOR_NAMES: List[str] = [
    "FZ", "FCZ", "CZ", "CPZ", "PZ", "C1", "C2", "FC1", "FC2",
    "CP1", "CP2", "P1", "P2", "FC3", "FC4", "CP3", "CP4",
]

VALID_PRESETS = ("circumference", "fronto_temporal", "interior", "random")


def channel_names_to_indices(
    names: Sequence[str],
    channel_order: Sequence[str],
    n_ld: int,
) -> "object":
    """
    Map channel names to global indices in `channel_order`, taking the first `n_ld`
    names that exist (order preserved).

    Returns:
        np.ndarray of shape (n_take,) with n_take <= n_ld if not enough names resolve.
    """
    import numpy as np

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


def indices_for_preset(
    preset: str,
    channel_order: Sequence[str],
    n_ld: int,
    rng,
) -> "object":
    """Return sorted LD channel indices for a named preset or random."""
    import numpy as np

    preset = preset.lower().replace("-", "_")
    if preset not in VALID_PRESETS:
        raise ValueError(f"preset must be one of {VALID_PRESETS}, got {preset!r}")

    if preset == "random":
        hd = np.arange(len(channel_order), dtype=np.int64)
        return np.sort(rng.choice(hd, size=n_ld, replace=False))

    if preset == "circumference":
        names = CIRCUMFERENCE_NAMES
    elif preset == "fronto_temporal":
        names = FRONTO_TEMPORAL_NAMES
    elif preset == "interior":
        names = INTERIOR_NAMES
    else:
        names = CIRCUMFERENCE_NAMES

    return np.sort(channel_names_to_indices(names, channel_order, n_ld))

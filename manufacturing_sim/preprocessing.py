from __future__ import annotations

from pathlib import Path

import numpy as np


def load_motion_file(motion_path: str | Path) -> np.ndarray:
    """Load SKEL .MOTION files that contain 12-value rigid transforms."""
    path = Path(motion_path)
    if not path.exists():
        raise FileNotFoundError(f"Motion file not found: {path}")

    rows: list[list[float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) != 12:
                continue
            try:
                rows.append([float(value) for value in parts])
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No numeric 12-column motion rows found in {path}")
    return np.asarray(rows, dtype=np.float32)


def normalize_translation(trajectory: np.ndarray, scale: float = 1000.0) -> np.ndarray:
    translation = trajectory[:, 9:12].astype(np.float32) / scale
    translation = translation - translation[0]
    max_abs = float(np.max(np.abs(translation)))
    if max_abs > 0.75:
        translation = translation * (0.75 / max_abs)
    return translation


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32)
    window = min(window, len(values))
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.vstack(
        [np.convolve(padded[:, idx], kernel, mode="valid") for idx in range(values.shape[1])]
    ).T
    return smoothed.astype(np.float32)


def robust_clip(values: np.ndarray, z_limit: float = 4.0) -> np.ndarray:
    center = np.median(values, axis=0)
    spread = np.median(np.abs(values - center), axis=0)
    spread[spread < 1e-6] = 1e-6
    low = center - z_limit * 1.4826 * spread
    high = center + z_limit * 1.4826 * spread
    return np.clip(values, low, high).astype(np.float32)


def resample(values: np.ndarray, frames: int) -> np.ndarray:
    if len(values) == frames:
        return values.astype(np.float32)
    x_old = np.linspace(0.0, 1.0, len(values))
    x_new = np.linspace(0.0, 1.0, frames)
    out = np.vstack([np.interp(x_new, x_old, values[:, idx]) for idx in range(values.shape[1])]).T
    return out.astype(np.float32)


def build_manufacturing_target(
    motion_path: str | Path,
    frames: int = 240,
    smooth_window: int = 9,
) -> np.ndarray:
    """Convert human hand/object motion into a compact tabletop assembly path."""
    raw = load_motion_file(motion_path)
    xyz = normalize_translation(raw)
    xyz = robust_clip(xyz)
    xyz = moving_average(xyz, smooth_window)
    xyz = resample(xyz, frames)

    # Rotate and shift into a small work-cell tabletop frame.
    target = np.zeros_like(xyz)
    target[:, 0] = 0.28 + 0.55 * xyz[:, 0]
    target[:, 1] = 0.00 + 0.50 * xyz[:, 2]
    target[:, 2] = 0.34 + 0.45 * xyz[:, 1]
    target[:, 2] = np.clip(target[:, 2], 0.14, 0.62)
    return target.astype(np.float32)


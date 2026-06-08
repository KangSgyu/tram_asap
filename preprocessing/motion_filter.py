import sys
from pathlib import Path

import numpy as np
from scipy.signal import medfilt
from scipy.signal import savgol_filter

# =========================
# Project Path Setting
# =========================

PROJECT_ROOT = Path.cwd().parent

sys.path.append(str(PROJECT_ROOT))

# =========================
# Import Modules
# =========================
from skel_loader import SKELLoader
from visualization.plot_trajectory import (plot_upper_body_trajectory)

# =========================
# Filtering Functions
# =========================

def median_filter_trajectory(trajectory, kernel_size=7):
    """
    Remove spike noise using median filter
    """
    if trajectory.ndim != 2:

        raise ValueError("trajectory must be 2D")

    filtered = np.zeros_like(
        trajectory,
        dtype=np.float32
    )

    for feature_idx in range(
        trajectory.shape[1]
    ):

        filtered[:, feature_idx] = (
            medfilt(
                trajectory[:, feature_idx],
                kernel_size=kernel_size
            )
        )

    return filtered


def savgol_smoothing(
    trajectory,
    window_length=21,
    polyorder=3
):

    """
    Smooth trajectory while preserving motion shape
    """

    if trajectory.ndim != 2:

        raise ValueError(
            "trajectory must be 2D"
        )

    filtered = np.zeros_like(
        trajectory,
        dtype=np.float32
    )

    for feature_idx in range(
        trajectory.shape[1]
    ):

        filtered[:, feature_idx] = (
            savgol_filter(
                trajectory[:, feature_idx],
                window_length=window_length,
                polyorder=polyorder
            )
        )

    return filtered

# =========================
# Data Loading
# =========================

DATA_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "SKEL"
)

loader = SKELLoader(
    str(DATA_ROOT)
)

# =========================
# Motion Processing
# =========================

result = loader.process_motion(
    "Rigid",
    "Model_Hand_R.MOTION"
)

trajectory = result["trajectory"]

# =========================
# Filtering Pipeline
# =========================

# 1. Spike Removal
trajectory = median_filter_trajectory(
    trajectory,
    kernel_size=7
)

# 2. Smooth Motion
trajectory = savgol_smoothing(
    trajectory,
    window_length=21,
    polyorder=3
)

# =========================
# Debug Information
# =========================

print("Environment:")
print(result["environment"])

print("\nTrajectory Shape:")
print(trajectory.shape)

print("\nTrajectory Range:")
print(
    trajectory.min(),
    trajectory.max()
)

# =========================
# Visualization
# =========================

plot_upper_body_trajectory(
    trajectory,
    title="Filtered Rigid Manipulation Trajectory"
)
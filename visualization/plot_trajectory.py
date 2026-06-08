import matplotlib.pyplot as plt
import numpy as np


def plot_upper_body_trajectory(trajectory, title="Upper-body Trajectory"):

    if trajectory.ndim != 2:
        raise ValueError("trajectory shape must be (T, F)")

    if trajectory.shape[1] != 12:
        raise ValueError("trajectory feature must be 12")

    # rotation / translation 분리
    rotation = trajectory[:, :9]

    translation = trajectory[:, 9:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # -------------------------
    # Rotation Matrix Plot
    # -------------------------

    for i in range(9):

        axes[0].plot(rotation[:, i], label=f"R{i}")

    axes[0].set_title("Rotation Matrix Trajectory")

    axes[0].set_xlabel("Frame")

    axes[0].set_ylabel("Rotation Value")

    axes[0].grid(True)

    axes[0].legend(loc="upper right", ncol=3)

    # -------------------------
    # Translation Plot
    # -------------------------

    translation_labels = ["X", "Y", "Z"]

    for i in range(3):

        axes[1].plot(
            translation[:, i],
            label=translation_labels[i])

    axes[1].set_title("Translation Trajectory")

    axes[1].set_xlabel("Frame")

    axes[1].set_ylabel("Position")

    axes[1].grid(True)

    axes[1].legend()

    fig.suptitle(title)

    plt.tight_layout()

    plt.show()
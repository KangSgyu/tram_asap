from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from manufacturing_sim.robot import UpperBodyRobot


def _draw_cell(ax, process_name: str) -> None:
    table_x = np.array([[-0.20, 0.82], [-0.20, 0.82]])
    table_y = np.array([[-0.46, -0.46], [0.46, 0.46]])
    table_z = np.zeros((2, 2)) + 0.08
    ax.plot_surface(table_x, table_y, table_z, color="#d7dee8", alpha=0.45, linewidth=0)
    ax.plot([0.44, 0.64], [-0.12, -0.12], [0.105, 0.105], color="#2f4858", lw=5)
    ax.plot([0.44, 0.64], [0.12, 0.12], [0.105, 0.105], color="#2f4858", lw=5)
    ax.scatter([0.16, 0.60], [-0.28, 0.25], [0.12, 0.12], s=[100, 130], c=["#ec6f43", "#27a6a6"])
    ax.text(0.36, -0.42, 0.16, process_name.replace("_", " "), fontsize=9)


def save_path_plot(
    target: np.ndarray,
    before: np.ndarray,
    after: np.ndarray,
    error_before: np.ndarray,
    error_after: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.plot(target[:, 0], target[:, 1], target[:, 2], color="#141414", lw=2, label="human target")
    ax1.plot(before[:, 0], before[:, 1], before[:, 2], color="#d65f5f", lw=1.8, label="before")
    ax1.plot(after[:, 0], after[:, 1], after[:, 2], color="#2f8f83", lw=1.8, label="after")
    _draw_cell(ax1, title)
    ax1.set_xlabel("X m")
    ax1.set_ylabel("Y m")
    ax1.set_zlabel("Z m")
    ax1.set_xlim(-0.22, 0.85)
    ax1.set_ylim(-0.50, 0.50)
    ax1.set_zlim(0.05, 0.98)
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(122)
    ax2.plot(error_before, color="#d65f5f", label="before")
    ax2.plot(error_after, color="#2f8f83", label="after")
    ax2.set_title("Tracking error")
    ax2.set_xlabel("Frame")
    ax2.set_ylabel("Error m")
    ax2.grid(True, alpha=0.35)
    ax2.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_robot_animation(
    target: np.ndarray,
    response: np.ndarray,
    path: str | Path,
    process_name: str,
    frames: int = 90,
) -> None:
    robot = UpperBodyRobot()
    indices = np.linspace(0, len(response) - 1, frames).astype(int)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int):
        idx = int(indices[frame_idx])
        ax.clear()
        _draw_cell(ax, process_name)
        ax.plot(target[:, 0], target[:, 1], target[:, 2], color="#141414", lw=1.0, alpha=0.4)
        ax.plot(response[: idx + 1, 0], response[: idx + 1, 1], response[: idx + 1, 2], color="#2f8f83", lw=2.0)
        pose = robot.full_pose(response[idx])
        for arm_name, color in [("right", "#335c81"), ("left", "#7f4f24")]:
            arm = pose[arm_name]
            ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], color=color, lw=7, solid_capstyle="round")
            ax.scatter(arm[:, 0], arm[:, 1], arm[:, 2], color=color, s=[45, 38, 50])
        ax.plot([0, 0], [0, 0], [0.30, pose["head"][2]], color="#434343", lw=8, solid_capstyle="round")
        ax.scatter([pose["head"][0]], [pose["head"][1]], [pose["head"][2]], color="#434343", s=140)
        ax.scatter([response[idx, 0]], [response[idx, 1]], [response[idx, 2]], color="#2f8f83", s=80)
        ax.set_xlim(-0.22, 0.85)
        ax.set_ylim(-0.50, 0.50)
        ax.set_zlim(0.05, 0.98)
        ax.set_xlabel("X m")
        ax.set_ylabel("Y m")
        ax.set_zlabel("Z m")
        ax.set_title(f"{process_name} upper-body assembly robot")
        ax.view_init(elev=24, azim=-58)
        return []

    animation = FuncAnimation(fig, update, frames=len(indices), interval=50, blit=False)
    animation.save(path, writer=PillowWriter(fps=18))
    plt.close(fig)


def save_rl_training_plot(
    history: list[dict[str, float]],
    path: str | Path,
    title: str = "Deep RL virtual robot training",
) -> None:
    generations = [item["generation"] for item in history]
    reward = [item["reward"] for item in history]
    mean_reward = [item["mean_reward"] for item in history]
    distance = [item["final_distance_m"] for item in history]
    success = [item["success_rate"] for item in history]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(generations, reward, color="#1f6f8b", lw=2, label="best")
    axes[0].plot(generations, mean_reward, color="#9a6f2f", lw=1.6, label="population mean")
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Generation")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend()

    axes[1].plot(generations, distance, color="#c84b31", lw=2)
    axes[1].set_title("Object-goal distance")
    axes[1].set_xlabel("Generation")
    axes[1].set_ylabel("m")
    axes[1].grid(True, alpha=0.35)

    axes[2].plot(generations, success, color="#2f8f83", lw=2)
    axes[2].set_title("Success rate")
    axes[2].set_xlabel("Generation")
    axes[2].set_ylabel("%")
    axes[2].set_ylim(-2, 102)
    axes[2].grid(True, alpha=0.35)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_virtual_robot_trace(
    trace: dict[str, np.ndarray],
    goal: np.ndarray,
    obstacles: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    ee = trace["ee"]
    obj = trace["object"]
    rewards = trace["reward"]

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(121, projection="3d")
    _draw_cell(ax1, title)
    ax1.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="#335c81", lw=2, label="end effector")
    ax1.plot(obj[:, 0], obj[:, 1], obj[:, 2], color="#ec6f43", lw=2, label="object")
    ax1.scatter([goal[0]], [goal[1]], [goal[2]], color="#2f8f83", s=120, marker="*", label="goal")
    for obstacle in obstacles:
        ax1.scatter([obstacle[0]], [obstacle[1]], [obstacle[2]], color="#4a4a4a", s=160, alpha=0.75)
    ax1.set_xlim(-0.05, 0.85)
    ax1.set_ylim(-0.50, 0.50)
    ax1.set_zlim(0.05, 0.80)
    ax1.set_xlabel("X m")
    ax1.set_ylabel("Y m")
    ax1.set_zlabel("Z m")
    ax1.legend(loc="upper left")
    ax1.view_init(elev=25, azim=-62)

    ax2 = fig.add_subplot(122)
    ax2.plot(rewards, color="#1f6f8b", lw=1.8)
    ax2.set_title("Step reward")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Reward")
    ax2.grid(True, alpha=0.35)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_virtual_robot_before_after(
    before_trace: dict[str, np.ndarray],
    after_trace: dict[str, np.ndarray],
    goal: np.ndarray,
    obstacles: np.ndarray,
    path: str | Path,
    title: str,
) -> None:
    fig = plt.figure(figsize=(14, 6))
    cases = [
        ("Before RL", before_trace, "#c84b31"),
        ("After RL", after_trace, "#2f8f83"),
    ]
    for plot_idx, (label, trace, color) in enumerate(cases, start=1):
        ax = fig.add_subplot(1, 2, plot_idx, projection="3d")
        _draw_cell(ax, label)
        ee = trace["ee"]
        obj = trace["object"]
        ax.plot(ee[:, 0], ee[:, 1], ee[:, 2], color="#335c81", lw=1.8, label="end effector")
        ax.plot(obj[:, 0], obj[:, 1], obj[:, 2], color=color, lw=2.2, label="object")
        ax.scatter([obj[0, 0]], [obj[0, 1]], [obj[0, 2]], color="#141414", s=55, label="start")
        ax.scatter([obj[-1, 0]], [obj[-1, 1]], [obj[-1, 2]], color=color, s=80, label="final")
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="#2f8f83", s=130, marker="*", label="goal")
        for obstacle in obstacles:
            ax.scatter([obstacle[0]], [obstacle[1]], [obstacle[2]], color="#4a4a4a", s=150, alpha=0.70)
        ax.set_xlim(-0.05, 0.85)
        ax.set_ylim(-0.50, 0.50)
        ax.set_zlim(0.05, 0.80)
        ax.set_xlabel("X m")
        ax.set_ylabel("Y m")
        ax.set_zlabel("Z m")
        ax.set_title(label)
        ax.legend(loc="upper left")
        ax.view_init(elev=25, azim=-62)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def save_virtual_robot_animation_from_trace(
    trace: dict[str, np.ndarray],
    goal: np.ndarray,
    obstacles: np.ndarray,
    path: str | Path,
    title: str,
    frames: int = 90,
) -> None:
    robot = UpperBodyRobot()
    ee = trace["ee"]
    obj = trace["object"]
    indices = np.linspace(0, len(ee) - 1, min(frames, len(ee))).astype(int)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame_idx: int):
        idx = int(indices[frame_idx])
        ax.clear()
        _draw_cell(ax, title)
        for obstacle in obstacles:
            ax.scatter([obstacle[0]], [obstacle[1]], [obstacle[2]], color="#4a4a4a", s=150, alpha=0.70)
        ax.scatter([goal[0]], [goal[1]], [goal[2]], color="#2f8f83", s=150, marker="*", label="goal")
        ax.plot(ee[: idx + 1, 0], ee[: idx + 1, 1], ee[: idx + 1, 2], color="#335c81", lw=2.0)
        ax.plot(obj[: idx + 1, 0], obj[: idx + 1, 1], obj[: idx + 1, 2], color="#ec6f43", lw=2.4)
        pose = robot.full_pose(ee[idx])
        for arm_name, color in [("right", "#335c81"), ("left", "#7f4f24")]:
            arm = pose[arm_name]
            ax.plot(arm[:, 0], arm[:, 1], arm[:, 2], color=color, lw=7, solid_capstyle="round")
            ax.scatter(arm[:, 0], arm[:, 1], arm[:, 2], color=color, s=[45, 38, 50])
        ax.scatter([obj[idx, 0]], [obj[idx, 1]], [obj[idx, 2]], color="#ec6f43", s=95, label="object")
        ax.set_xlim(-0.05, 0.85)
        ax.set_ylim(-0.50, 0.50)
        ax.set_zlim(0.05, 0.80)
        ax.set_xlabel("X m")
        ax.set_ylabel("Y m")
        ax.set_zlabel("Z m")
        ax.set_title(title)
        ax.view_init(elev=25, azim=-62)
        return []

    animation = FuncAnimation(fig, update, frames=len(indices), interval=55, blit=False)
    animation.save(path, writer=PillowWriter(fps=18))
    plt.close(fig)


def save_camera_snapshot(image: np.ndarray, path: str | Path, title: str = "Virtual camera sensor") -> None:
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(image, cmap="magma", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

from __future__ import annotations

import numpy as np

from .config import RobotSpec


class UpperBodyRobot:
    """Simple dual-arm upper-body robot model for manufacturing-cell animation."""

    def __init__(self, spec: RobotSpec | None = None) -> None:
        self.spec = spec or RobotSpec()
        self.torso = np.array([0.0, 0.0, 0.58], dtype=np.float32)
        half_width = self.spec.shoulder_width_m / 2.0
        self.right_shoulder = np.array([0.0, -half_width, 0.76], dtype=np.float32)
        self.left_shoulder = np.array([0.0, half_width, 0.76], dtype=np.float32)

    def clamp_workspace(self, wrist: np.ndarray, shoulder: np.ndarray) -> np.ndarray:
        delta = wrist - shoulder
        dist = float(np.linalg.norm(delta))
        max_reach = self.spec.upper_arm_m + self.spec.forearm_m
        if dist > max_reach:
            wrist = shoulder + delta / dist * max_reach
        wrist[2] = max(0.08, wrist[2])
        return wrist.astype(np.float32)

    def arm_joints(self, wrist: np.ndarray, side: str = "right") -> np.ndarray:
        shoulder = self.right_shoulder if side == "right" else self.left_shoulder
        wrist = self.clamp_workspace(wrist.copy(), shoulder)
        delta = wrist - shoulder
        dist = max(float(np.linalg.norm(delta)), 1e-6)
        direction = delta / dist
        side_sign = -1.0 if side == "right" else 1.0
        bend = np.array([0.02, side_sign * 0.10, -0.12], dtype=np.float32)
        elbow = shoulder + direction * min(self.spec.upper_arm_m, dist * 0.52) + bend
        return np.vstack([shoulder, elbow, wrist]).astype(np.float32)

    def full_pose(self, right_wrist: np.ndarray) -> dict[str, np.ndarray]:
        right = self.arm_joints(right_wrist, "right")
        helper_target = np.array([right_wrist[0] * 0.72, 0.24, right_wrist[2] + 0.02], dtype=np.float32)
        left = self.arm_joints(helper_target, "left")
        return {
            "torso": self.torso,
            "right": right,
            "left": left,
            "head": np.array([0.0, 0.0, 0.94], dtype=np.float32),
        }

    def estimate_torque(self, previous_wrist: np.ndarray, wrist: np.ndarray, dt: float) -> float:
        speed = float(np.linalg.norm(wrist - previous_wrist) / max(dt, 1e-6))
        reach = float(np.linalg.norm(wrist - self.right_shoulder))
        return 18.0 + 42.0 * speed + 12.0 * reach


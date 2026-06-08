from __future__ import annotations

import numpy as np

from .config import ProcessSpec, RobotSpec
from .robot import UpperBodyRobot


class ManufacturingCell:
    """Tabletop assembly cell with material-dependent dynamics."""

    def __init__(
        self,
        process: ProcessSpec,
        robot_spec: RobotSpec | None = None,
        seed: int = 7,
    ) -> None:
        self.process = process
        self.robot = UpperBodyRobot(robot_spec)
        self.rng = np.random.default_rng(seed)

    def simulate_episode(
        self,
        target: np.ndarray,
        correction: np.ndarray | None = None,
        dt: float = 1.0 / 30.0,
    ) -> dict[str, np.ndarray | float]:
        if correction is None:
            correction = np.zeros_like(target, dtype=np.float32)

        command = target + correction
        response = np.zeros_like(command, dtype=np.float32)
        response[0] = command[0]
        safety_events = 0

        phase = np.linspace(0.0, np.pi * 10.0, len(command), dtype=np.float32)
        vibration = self.process.vibration * np.column_stack(
            [np.sin(phase), np.sin(phase * 1.7 + 0.4), np.cos(phase * 1.3)]
        )
        noise = self.rng.normal(0.0, self.process.contact_noise, command.shape).astype(np.float32)

        for idx in range(1, len(command)):
            desired = command[idx] + vibration[idx] + noise[idx]
            material_pull = self.process.material_compliance * (response[idx - 1] - target[idx])
            desired = desired + material_pull
            step = (1.0 - self.process.lag) * desired + self.process.lag * response[idx - 1]
            delta = step - response[idx - 1]
            norm = float(np.linalg.norm(delta))
            if norm > self.robot.spec.max_step_m:
                step = response[idx - 1] + delta / norm * self.robot.spec.max_step_m
                safety_events += 1

            torque = self.robot.estimate_torque(response[idx - 1], step, dt)
            if torque > self.robot.spec.torque_limit_nm:
                blend = self.robot.spec.torque_limit_nm / torque
                step = response[idx - 1] + (step - response[idx - 1]) * blend
                safety_events += 1
            response[idx] = step

        error = np.linalg.norm(target - response, axis=1)
        defects = error > self.process.tolerance_m
        return {
            "target": target,
            "command": command,
            "response": response,
            "error": error.astype(np.float32),
            "defects": defects,
            "defect_rate": float(np.mean(defects) * 100.0),
            "mean_error": float(np.mean(error)),
            "max_error": float(np.max(error)),
            "safety_events": float(safety_events),
            "malfunction_rate": 0.0,
        }


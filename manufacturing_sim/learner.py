from __future__ import annotations

import numpy as np

from .environment import ManufacturingCell


class DynamicsAlignmentLearner:
    """Lightweight ASAP-like residual learner with ASCPO-style safety limits."""

    def __init__(
        self,
        cell: ManufacturingCell,
        learning_rate: float = 0.42,
        smooth_window: int = 13,
        max_correction_m: float = 0.10,
    ) -> None:
        self.cell = cell
        self.learning_rate = learning_rate
        self.smooth_window = smooth_window
        self.max_correction_m = max_correction_m

    def _smooth(self, values: np.ndarray) -> np.ndarray:
        if self.smooth_window <= 1:
            return values
        window = min(self.smooth_window, len(values))
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / window
        return np.vstack(
            [np.convolve(padded[:, idx], kernel, mode="valid") for idx in range(values.shape[1])]
        ).T.astype(np.float32)

    def train(self, target: np.ndarray, epochs: int = 12) -> dict[str, object]:
        correction = np.zeros_like(target, dtype=np.float32)
        history: list[dict[str, float]] = []
        first_episode = None
        best_episode = None
        best_correction = correction.copy()

        for epoch in range(epochs):
            episode = self.cell.simulate_episode(target, correction)
            if first_episode is None:
                first_episode = episode
                best_episode = episode
            if (
                float(episode["defect_rate"]),
                float(episode["mean_error"]),
            ) <= (
                float(best_episode["defect_rate"]),
                float(best_episode["mean_error"]),
            ):
                best_episode = episode
                best_correction = correction.copy()
            residual = target - episode["response"]
            error = np.linalg.norm(residual, axis=1)
            active = (error > self.cell.process.tolerance_m * 0.65).astype(np.float32)[:, None]
            correction += self.learning_rate * self._smooth(residual * active)
            correction = np.clip(correction, -self.max_correction_m, self.max_correction_m)
            history.append(
                {
                    "epoch": float(epoch + 1),
                    "defect_rate": float(episode["defect_rate"]),
                    "mean_error": float(episode["mean_error"]),
                    "max_error": float(episode["max_error"]),
                    "safety_events": float(episode["safety_events"]),
                    "malfunction_rate": float(episode["malfunction_rate"]),
                }
            )

        assert first_episode is not None
        assert best_episode is not None
        return {
            "history": history,
            "before": first_episode,
            "after": best_episode,
            "correction": best_correction,
        }

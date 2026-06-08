from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


def ensure_dir(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_history_csv(path: str | Path, history: list[dict[str, float]]) -> None:
    if not history:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def write_summary_json(path: str | Path, summary: dict[str, object]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)


def compact_episode(episode: dict[str, object]) -> dict[str, float]:
    return {
        "defect_rate": float(episode["defect_rate"]),
        "mean_error": float(episode["mean_error"]),
        "max_error": float(episode["max_error"]),
        "safety_events": float(episode["safety_events"]),
        "malfunction_rate": float(episode["malfunction_rate"]),
    }


def save_episode_npz(path: str | Path, episode: dict[str, object]) -> None:
    np.savez_compressed(
        path,
        target=episode["target"],
        command=episode["command"],
        response=episode["response"],
        error=episode["error"],
        defects=episode["defects"],
    )


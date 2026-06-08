from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from manufacturing_sim.config import OUTPUT_ROOT, PROCESS_SPECS, RobotSpec
from manufacturing_sim.physical_ai_env import DeepRLTrainer, RobotVirtualEnvironment
from manufacturing_sim.reporting import ensure_dir, write_history_csv, write_summary_json
from visualization.simulation_viewer import (
    save_camera_snapshot,
    save_rl_training_plot,
    save_virtual_robot_animation_from_trace,
    save_virtual_robot_before_after,
    save_virtual_robot_trace,
)


def run(process_name: str, generations: int, population: int, max_steps: int) -> dict[str, object]:
    process = PROCESS_SPECS[process_name]
    env = RobotVirtualEnvironment(process, RobotSpec(), max_steps=max_steps)
    trainer = DeepRLTrainer(env, population=population)
    baseline_rng = np.random.default_rng(101)
    baseline_params = baseline_rng.normal(0.0, 0.55, trainer.policy.param_dim).astype(np.float32)
    _, before_metrics, before_trace = trainer.evaluate(
        baseline_params,
        episodes=1,
        record=True,
        use_prior=False,
    )
    result = trainer.train(generations=generations)

    out_dir = ensure_dir(OUTPUT_ROOT / "physical_ai_rl" / process_name)
    history = result["history"]
    after_trace = result["trace"]
    after_metrics = result["best_metrics"]
    comparison = [
        {"stage": "before_rl", **before_metrics},
        {"stage": "after_rl", **after_metrics},
    ]

    write_history_csv(out_dir / "rl_training_history.csv", history)
    write_history_csv(out_dir / "rl_before_after_comparison.csv", comparison)
    np.savez_compressed(
        out_dir / "rl_episode_trace.npz",
        before_end_effector=before_trace["ee"],
        before_object=before_trace["object"],
        before_reward=before_trace["reward"],
        after_end_effector=after_trace["ee"],
        after_object=after_trace["object"],
        after_reward=after_trace["reward"],
        policy_params=result["policy_params"],
    )
    save_rl_training_plot(
        history,
        out_dir / "rl_training_curve.png",
        title=f"{process_name} pick-and-place deep RL training",
    )
    save_virtual_robot_trace(
        after_trace,
        env.goal_position,
        env.obstacles,
        out_dir / "virtual_robot_trace.png",
        title=f"{process_name} virtual physics environment",
    )
    save_virtual_robot_before_after(
        before_trace,
        after_trace,
        env.goal_position,
        env.obstacles,
        out_dir / "rl_before_after_trace.png",
        title=f"{process_name} before/after reinforcement learning",
    )
    save_virtual_robot_animation_from_trace(
        after_trace,
        env.goal_position,
        env.obstacles,
        out_dir / "learned_virtual_environment.gif",
        title=f"{process_name} learned virtual robot policy",
    )
    if len(after_trace["camera"]):
        save_camera_snapshot(after_trace["camera"][0], out_dir / "virtual_camera_snapshot.png")

    summary = {
        "process": process_name,
        "task": "pick-and-place object transfer in a tabletop manufacturing cell",
        "physics_engine": {
            "gravity_mps2": env.physics.gravity_mps2,
            "table_friction": env.physics.table_friction,
            "restitution": env.physics.restitution,
            "collision_bodies": int(len(env.obstacles) + 2),
            "material_compliance": process.material_compliance,
            "contact_noise": process.contact_noise,
        },
        "sensor_simulation": {
            "camera": f"{env.sensors.camera_size}x{env.sensors.camera_size} top-view occupancy image",
            "lidar_rays": env.sensors.lidar_rays,
            "imu_channels": 6,
        },
        "reinforcement_learning": {
            "policy": "two-layer neural policy",
            "optimizer": "ASCPO-like safety-constrained cross-entropy policy search",
            "before_after_definition": (
                "before_rl is an untrained neural policy without the pick-and-place prior; "
                "after_rl is the trained prior-residual policy."
            ),
            "efficiency_methods": [
                "reward shaping with distance progress",
                "potential-field pick-and-place prior with neural residual learning",
                "action smoothing",
                "elite sampling",
                "domain noise from material compliance/contact noise",
                "energy and collision penalties",
                "ASCPO-like constrained policy selection using explicit safety costs",
            ],
            "generations": generations,
            "population": population,
            "max_steps": max_steps,
        },
        "ascpo_like_safety_constraint": {
            "safety_cost": (
                "collision_cost + weighted_energy_cost + weighted_smoothness_cost "
                "+ timeout_cost + drop_cost"
            ),
            "safety_limit": env.safety.safety_limit,
            "selection_objective": "reward - adaptive_multiplier * max(0, safety_cost - safety_limit)",
            "adaptive_multiplier_initial": env.safety.multiplier_init,
        },
        "before_rl_metrics": before_metrics,
        "after_rl_metrics": after_metrics,
        "improvement": {
            "reward_delta": float(after_metrics["reward"] - before_metrics["reward"]),
            "final_distance_delta_m": float(before_metrics["final_distance_m"] - after_metrics["final_distance_m"]),
            "collision_delta": float(before_metrics["collisions"] - after_metrics["collisions"]),
            "safety_cost_delta": float(before_metrics["safety_cost"] - after_metrics["safety_cost"]),
            "step_delta": float(before_metrics["steps"] - after_metrics["steps"]),
        },
        "outputs": {
            "history_csv": str((out_dir / "rl_training_history.csv").resolve()),
            "before_after_csv": str((out_dir / "rl_before_after_comparison.csv").resolve()),
            "training_curve": str((out_dir / "rl_training_curve.png").resolve()),
            "trace_plot": str((out_dir / "virtual_robot_trace.png").resolve()),
            "before_after_trace": str((out_dir / "rl_before_after_trace.png").resolve()),
            "learned_animation": str((out_dir / "learned_virtual_environment.gif").resolve()),
            "camera_snapshot": str((out_dir / "virtual_camera_snapshot.png").resolve()),
        },
        "paper_conclusion": (
            "The upgraded simulation links a physics-based virtual cell, multimodal sensor observations, "
            "and reward-driven neural policy optimization. This provides an executable basis for discussing "
            "deep reinforcement learning in Physical AI manufacturing robots beyond trajectory tracking alone."
        ),
    }
    write_summary_json(out_dir / "rl_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run upgraded Physical AI robot RL simulation.")
    parser.add_argument("--process", choices=list(PROCESS_SPECS.keys()) + ["all"], default="all")
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=120)
    args = parser.parse_args()

    processes = list(PROCESS_SPECS.keys()) if args.process == "all" else [args.process]
    summaries = [run(name, args.generations, args.population, args.max_steps) for name in processes]
    for item in summaries:
        metrics = item["after_rl_metrics"]
        before = item["before_rl_metrics"]
        print(
            f"{item['process']}: reward {before['reward']:.2f} -> {metrics['reward']:.2f}, "
            f"success {before['success_rate']:.1f}% -> {metrics['success_rate']:.1f}%, "
            f"final distance {before['final_distance_m']:.3f} -> {metrics['final_distance_m']:.3f} m, "
            f"safety cost {before['safety_cost']:.2f} -> {metrics['safety_cost']:.2f}"
        )
    print(f"Outputs saved to: {Path(OUTPUT_ROOT / 'physical_ai_rl').resolve()}")


if __name__ == "__main__":
    main()

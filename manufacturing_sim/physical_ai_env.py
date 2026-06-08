from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ProcessSpec, RobotSpec
from .robot import UpperBodyRobot


@dataclass(frozen=True)
class PhysicsEngineConfig:
    """Compact rigid-body physics settings for tabletop robot learning."""

    dt: float = 1.0 / 30.0
    gravity_mps2: float = -9.81
    table_height_m: float = 0.18
    table_friction: float = 0.82
    object_radius_m: float = 0.035
    end_effector_radius_m: float = 0.045
    restitution: float = 0.12
    max_action_mps: float = 0.70
    contact_stiffness: float = 16.0
    action_smoothing: float = 0.30


@dataclass(frozen=True)
class SensorConfig:
    camera_size: int = 32
    camera_noise: float = 0.015
    lidar_rays: int = 24
    lidar_range_m: float = 1.00
    imu_noise: float = 0.018


@dataclass(frozen=True)
class RewardConfig:
    reach_scale: float = 1.00
    grasp_reward: float = 0.08
    place_reward: float = 10.00
    progress_scale: float = 5.00
    collision_penalty: float = 1.10
    drop_penalty: float = 2.00
    energy_penalty: float = 0.018
    smoothness_penalty: float = 0.010
    timeout_penalty: float = 0.80


@dataclass(frozen=True)
class SafetyConstraintConfig:
    """ASCPO-like safety-cost settings for constrained policy selection."""

    safety_limit: float = 1.20
    collision_cost: float = 1.00
    energy_cost: float = 0.020
    smoothness_cost: float = 0.060
    timeout_cost: float = 0.60
    drop_cost: float = 2.00
    multiplier_init: float = 1.20
    multiplier_step: float = 0.12
    multiplier_decay: float = 0.96


@dataclass
class VirtualRobotState:
    ee_position: np.ndarray
    ee_velocity: np.ndarray
    object_position: np.ndarray
    object_velocity: np.ndarray
    previous_action: np.ndarray
    grasped: bool
    step_index: int
    collisions: int


def _as_float3(values: list[float] | tuple[float, float, float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


class RobotVirtualEnvironment:
    """Physics, sensors, and reward loop for virtual robot policy learning.

    The model is intentionally lightweight so it can run without an external
    robotics simulator, but the API mirrors common deep-RL robot environments:
    reset(), observe(), and step(action).
    """

    def __init__(
        self,
        process: ProcessSpec,
        robot_spec: RobotSpec | None = None,
        physics: PhysicsEngineConfig | None = None,
        sensors: SensorConfig | None = None,
        rewards: RewardConfig | None = None,
        safety: SafetyConstraintConfig | None = None,
        seed: int = 11,
        max_steps: int = 120,
    ) -> None:
        self.process = process
        self.robot = UpperBodyRobot(robot_spec)
        self.physics = physics or PhysicsEngineConfig()
        self.sensors = sensors or SensorConfig()
        self.rewards = rewards or RewardConfig()
        self.safety = safety or SafetyConstraintConfig()
        self.max_steps = max_steps
        self.rng = np.random.default_rng(seed)
        self.goal_position = _as_float3([0.34, 0.02, self.physics.table_height_m + self.physics.object_radius_m])
        self.object_start = _as_float3([0.18, -0.22, self.physics.table_height_m + self.physics.object_radius_m])
        self.home_position = _as_float3([0.22, -0.18, 0.30])
        self.obstacles = np.asarray(
            [
                [0.34, -0.22, self.physics.table_height_m + 0.045, 0.040],
                [0.44, 0.14, self.physics.table_height_m + 0.045, 0.045],
            ],
            dtype=np.float32,
        )
        self.state = self.reset()

    @property
    def observation_dim(self) -> int:
        return int(self.observe().shape[0])

    @property
    def action_dim(self) -> int:
        return 3

    def reset(self) -> VirtualRobotState:
        object_jitter = self.rng.normal(0.0, [0.015, 0.015, 0.0]).astype(np.float32)
        ee_jitter = self.rng.normal(0.0, [0.015, 0.015, 0.010]).astype(np.float32)
        self.state = VirtualRobotState(
            ee_position=(self.home_position + ee_jitter).astype(np.float32),
            ee_velocity=np.zeros(3, dtype=np.float32),
            object_position=(self.object_start + object_jitter).astype(np.float32),
            object_velocity=np.zeros(3, dtype=np.float32),
            previous_action=np.zeros(3, dtype=np.float32),
            grasped=False,
            step_index=0,
            collisions=0,
        )
        return self.state

    def observe(self) -> np.ndarray:
        s = self.state
        lidar = self._simulate_lidar(s.ee_position)
        imu = self._simulate_imu(s.ee_velocity, s.previous_action)
        camera_features = self._simulate_camera_features()
        proprioception = np.concatenate(
            [
                s.ee_position,
                s.ee_velocity,
                s.object_position,
                s.object_velocity,
                self.goal_position,
                np.asarray([float(s.grasped), s.step_index / max(1, self.max_steps)], dtype=np.float32),
            ]
        )
        return np.concatenate([proprioception, lidar, imu, camera_features]).astype(np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        s = self.state
        prev_object_distance = float(np.linalg.norm(s.object_position - self.goal_position))
        prev_reach_distance = float(np.linalg.norm(s.ee_position - s.object_position))

        action = np.asarray(action, dtype=np.float32)
        action = np.nan_to_num(action, nan=0.0, posinf=1.0, neginf=-1.0)
        action = np.clip(action, -1.0, 1.0)
        previous_action = s.previous_action.copy()
        blended_action = (
            self.physics.action_smoothing * previous_action
            + (1.0 - self.physics.action_smoothing) * action
        )

        desired_velocity = blended_action * self.physics.max_action_mps
        new_ee_velocity = desired_velocity.astype(np.float32)
        new_ee_position = s.ee_position + new_ee_velocity * self.physics.dt
        new_ee_position = self.robot.clamp_workspace(new_ee_position, self.robot.right_shoulder)
        new_ee_position = self._clamp_to_cell(new_ee_position)

        object_force = np.zeros(3, dtype=np.float32)
        collision_count = 0
        collision_penalty = 0.0
        contact_distance = float(np.linalg.norm(new_ee_position - s.object_position))
        if contact_distance <= self.physics.end_effector_radius_m + self.physics.object_radius_m:
            object_force += self.physics.contact_stiffness * new_ee_velocity
            if not s.grasped and np.linalg.norm(new_ee_velocity) < 0.45:
                s.grasped = True

        for obstacle in self.obstacles:
            center = obstacle[:3]
            radius = float(obstacle[3])
            distance = float(np.linalg.norm(new_ee_position - center))
            if distance < radius + self.physics.end_effector_radius_m:
                collision_count += 1
                collision_penalty += self.rewards.collision_penalty
                direction = new_ee_position - center
                direction_norm = max(float(np.linalg.norm(direction)), 1e-6)
                new_ee_position = center + direction / direction_norm * (radius + self.physics.end_effector_radius_m)
                new_ee_velocity *= -self.physics.restitution

        if s.grasped:
            target_object_position = new_ee_position + np.asarray([0.0, 0.0, -0.035], dtype=np.float32)
            coupling = 1.0 / (1.0 + self.process.material_compliance * 2.8)
            new_object_position = s.object_position + (target_object_position - s.object_position) * coupling
            new_object_velocity = (new_object_position - s.object_position) / self.physics.dt
        else:
            material_drag = 1.0 + self.process.material_compliance * 2.4
            gravity = np.asarray([0.0, 0.0, self.physics.gravity_mps2], dtype=np.float32)
            acceleration = gravity + object_force / material_drag
            new_object_velocity = (s.object_velocity + acceleration * self.physics.dt).astype(np.float32)
            horizontal_friction = self.physics.table_friction + self.process.contact_noise * 8.0
            new_object_velocity[:2] *= max(0.0, 1.0 - horizontal_friction * self.physics.dt)
            new_object_position = s.object_position + new_object_velocity * self.physics.dt

        table_top = self.physics.table_height_m + self.physics.object_radius_m
        if new_object_position[2] < table_top:
            new_object_position[2] = table_top
            if new_object_velocity[2] < 0.0:
                new_object_velocity[2] = -new_object_velocity[2] * self.physics.restitution
            new_object_velocity[:2] *= max(0.0, 1.0 - self.physics.table_friction * self.physics.dt)

        if s.grasped and np.linalg.norm(new_ee_position - new_object_position) > 0.12:
            s.grasped = False

        s.ee_velocity = new_ee_velocity.astype(np.float32)
        s.ee_position = new_ee_position.astype(np.float32)
        s.object_velocity = new_object_velocity.astype(np.float32)
        s.object_position = self._clamp_object(new_object_position.astype(np.float32))
        s.previous_action = blended_action.astype(np.float32)
        s.step_index += 1
        s.collisions += collision_count

        object_distance = float(np.linalg.norm(s.object_position - self.goal_position))
        reach_distance = float(np.linalg.norm(s.ee_position - s.object_position))
        progress = (prev_object_distance - object_distance) + 0.5 * (prev_reach_distance - reach_distance)
        energy_cost = float(np.sum(np.square(action)))
        smoothness_cost = float(np.linalg.norm(action - previous_action))
        reward = (
            self.rewards.progress_scale * progress
            - self.rewards.reach_scale * reach_distance
            - self.rewards.energy_penalty * energy_cost
            - self.rewards.smoothness_penalty * smoothness_cost
            - collision_penalty
        )
        if s.grasped:
            reward += self.rewards.grasp_reward
        placed = bool(object_distance <= 0.075 and s.object_position[2] <= table_top + 0.025)
        if placed:
            reward += self.rewards.place_reward
        dropped = bool(s.object_position[2] < self.physics.table_height_m - 0.01)
        if dropped:
            reward -= self.rewards.drop_penalty
        done = placed or dropped or s.step_index >= self.max_steps
        timed_out = bool(done and not placed and not dropped)
        if timed_out:
            reward -= self.rewards.timeout_penalty

        safety_cost = (
            self.safety.collision_cost * float(collision_count)
            + self.safety.energy_cost * energy_cost
            + self.safety.smoothness_cost * smoothness_cost
            + self.safety.timeout_cost * float(timed_out)
            + self.safety.drop_cost * float(dropped)
        )

        info = {
            "object_distance": object_distance,
            "reach_distance": reach_distance,
            "grasped": float(s.grasped),
            "placed": float(placed),
            "collisions": float(s.collisions),
            "step_collisions": float(collision_count),
            "energy_cost": energy_cost,
            "smoothness_cost": smoothness_cost,
            "timeout": float(timed_out),
            "dropped": float(dropped),
            "safety_cost": float(safety_cost),
            "step": float(s.step_index),
            "reward": float(reward),
        }
        return self.observe(), float(reward), done, info

    def _clamp_to_cell(self, position: np.ndarray) -> np.ndarray:
        position = position.copy()
        position[0] = np.clip(position[0], -0.05, 0.82)
        position[1] = np.clip(position[1], -0.46, 0.46)
        position[2] = np.clip(position[2], self.physics.table_height_m + 0.05, 0.74)
        return position.astype(np.float32)

    def _clamp_object(self, position: np.ndarray) -> np.ndarray:
        position = position.copy()
        position[0] = np.clip(position[0], 0.02, 0.78)
        position[1] = np.clip(position[1], -0.42, 0.42)
        return position.astype(np.float32)

    def _simulate_lidar(self, origin: np.ndarray) -> np.ndarray:
        ranges = np.full(self.sensors.lidar_rays, self.sensors.lidar_range_m, dtype=np.float32)
        angles = np.linspace(-np.pi, np.pi, self.sensors.lidar_rays, endpoint=False)
        bodies = [(self.state.object_position, self.physics.object_radius_m)]
        bodies.extend((obstacle[:3], float(obstacle[3])) for obstacle in self.obstacles)
        for idx, angle in enumerate(angles):
            ray = np.asarray([np.cos(angle), np.sin(angle)], dtype=np.float32)
            for center, radius in bodies:
                delta = center[:2] - origin[:2]
                projection = float(np.dot(delta, ray))
                if projection <= 0.0:
                    continue
                closest = delta - projection * ray
                miss_distance = float(np.linalg.norm(closest))
                if miss_distance <= radius:
                    hit = max(0.0, projection - np.sqrt(max(radius**2 - miss_distance**2, 0.0)))
                    ranges[idx] = min(ranges[idx], hit)
        return np.clip(ranges / self.sensors.lidar_range_m, 0.0, 1.0).astype(np.float32)

    def _simulate_imu(self, velocity: np.ndarray, action: np.ndarray) -> np.ndarray:
        linear_accel = velocity / max(self.physics.dt, 1e-6)
        angular_proxy = np.asarray(
            [action[1] - action[2], action[2] - action[0], action[0] - action[1]],
            dtype=np.float32,
        )
        imu = np.concatenate([linear_accel, angular_proxy])
        noise = self.rng.normal(0.0, self.sensors.imu_noise, imu.shape).astype(np.float32)
        return (imu * 0.05 + noise).astype(np.float32)

    def _simulate_camera_features(self) -> np.ndarray:
        image = self.render_camera()
        yy, xx = np.mgrid[0 : image.shape[0], 0 : image.shape[1]]
        mass = float(np.sum(image)) + 1e-6
        centroid_x = float(np.sum(xx * image) / mass) / max(1, image.shape[1] - 1)
        centroid_y = float(np.sum(yy * image) / mass) / max(1, image.shape[0] - 1)
        return np.asarray(
            [
                float(np.mean(image)),
                float(np.max(image)),
                centroid_x,
                centroid_y,
                float(np.std(image)),
            ],
            dtype=np.float32,
        )

    def render_camera(self) -> np.ndarray:
        size = self.sensors.camera_size
        image = np.zeros((size, size), dtype=np.float32)

        def paint(position: np.ndarray, radius_px: int, value: float) -> None:
            x = int(np.interp(position[0], [0.0, 0.82], [0, size - 1]))
            y = int(np.interp(position[1], [-0.46, 0.46], [size - 1, 0]))
            yy, xx = np.ogrid[:size, :size]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius_px**2
            image[mask] = np.maximum(image[mask], value)

        paint(self.goal_position, 2, 0.35)
        paint(self.state.object_position, 2, 0.80)
        paint(self.state.ee_position, 2, 1.00)
        for obstacle in self.obstacles:
            paint(obstacle[:3], 2, 0.55)
        noise = self.rng.normal(0.0, self.sensors.camera_noise, image.shape).astype(np.float32)
        return np.clip(image + noise, 0.0, 1.0).astype(np.float32)


class NeuralPolicy:
    """Small MLP policy used by the sample-efficient CEM trainer."""

    def __init__(self, observation_dim: int, hidden_dim: int = 24, action_dim: int = 3) -> None:
        self.observation_dim = observation_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.param_dim = (
            observation_dim * hidden_dim
            + hidden_dim
            + hidden_dim * action_dim
            + action_dim
        )

    def unpack(self, params: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        offset = 0
        w1_size = self.observation_dim * self.hidden_dim
        w1 = params[offset : offset + w1_size].reshape(self.observation_dim, self.hidden_dim)
        offset += w1_size
        b1 = params[offset : offset + self.hidden_dim]
        offset += self.hidden_dim
        w2_size = self.hidden_dim * self.action_dim
        w2 = params[offset : offset + w2_size].reshape(self.hidden_dim, self.action_dim)
        offset += w2_size
        b2 = params[offset : offset + self.action_dim]
        return w1, b1, w2, b2

    def act(self, observation: np.ndarray, params: np.ndarray, use_prior: bool = True) -> np.ndarray:
        w1, b1, w2, b2 = self.unpack(params)
        hidden = np.tanh(observation @ w1 + b1)
        residual = 0.35 * np.tanh(hidden @ w2 + b2)
        if not use_prior:
            return residual.astype(np.float32)

        ee_position = observation[0:3]
        object_position = observation[6:9]
        goal_position = observation[12:15]
        grasped = observation[15] > 0.5
        reach_distance = float(np.linalg.norm(ee_position - object_position))
        if grasped:
            target = goal_position + np.asarray([0.0, 0.0, 0.04], dtype=np.float32)
        elif reach_distance > 0.075:
            target = object_position + np.asarray([0.0, 0.0, 0.055], dtype=np.float32)
        else:
            target = object_position + np.asarray([0.0, 0.0, 0.005], dtype=np.float32)
        prior = np.clip((target - ee_position) * 4.0, -1.0, 1.0)
        return np.tanh(prior + residual).astype(np.float32)


class DeepRLTrainer:
    """Cross-entropy policy optimizer with ASCPO-like safety constraints."""

    def __init__(
        self,
        env: RobotVirtualEnvironment,
        hidden_dim: int = 24,
        population: int = 18,
        elite_fraction: float = 0.25,
        seed: int = 23,
    ) -> None:
        self.env = env
        self.policy = NeuralPolicy(env.observation_dim, hidden_dim, env.action_dim)
        self.population = population
        self.elite_count = max(2, int(population * elite_fraction))
        self.rng = np.random.default_rng(seed)
        self.safety_multiplier = self.env.safety.multiplier_init

    def evaluate(
        self,
        params: np.ndarray,
        episodes: int = 2,
        record: bool = False,
        use_prior: bool = True,
    ) -> tuple[float, dict[str, float], dict[str, np.ndarray] | None]:
        rewards = []
        successes = []
        final_distances = []
        collisions = []
        steps = []
        safety_costs = []
        energy_costs = []
        smoothness_costs = []
        timeouts = []
        drops = []
        trace: dict[str, list[np.ndarray] | list[float]] = {
            "ee": [],
            "object": [],
            "reward": [],
            "camera": [],
        }
        for _ in range(episodes):
            self.env.reset()
            observation = self.env.observe()
            done = False
            total_reward = 0.0
            total_safety_cost = 0.0
            total_energy_cost = 0.0
            total_smoothness_cost = 0.0
            info: dict[str, float] = {}
            while not done:
                action = self.policy.act(observation, params, use_prior=use_prior)
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                total_safety_cost += info.get("safety_cost", 0.0)
                total_energy_cost += info.get("energy_cost", 0.0)
                total_smoothness_cost += info.get("smoothness_cost", 0.0)
                if record:
                    trace["ee"].append(self.env.state.ee_position.copy())
                    trace["object"].append(self.env.state.object_position.copy())
                    trace["reward"].append(float(reward))
                    if len(trace["camera"]) == 0:
                        trace["camera"].append(self.env.render_camera())
            rewards.append(total_reward)
            successes.append(info.get("placed", 0.0))
            final_distances.append(info.get("object_distance", 1.0))
            collisions.append(info.get("collisions", 0.0))
            steps.append(info.get("step", self.env.max_steps))
            safety_costs.append(total_safety_cost)
            energy_costs.append(total_energy_cost)
            smoothness_costs.append(total_smoothness_cost)
            timeouts.append(info.get("timeout", 0.0))
            drops.append(info.get("dropped", 0.0))
        mean_safety_cost = float(np.mean(safety_costs))
        safety_violation = max(0.0, mean_safety_cost - self.env.safety.safety_limit)
        metrics = {
            "reward": float(np.mean(rewards)),
            "success_rate": float(np.mean(successes) * 100.0),
            "final_distance_m": float(np.mean(final_distances)),
            "collisions": float(np.mean(collisions)),
            "steps": float(np.mean(steps)),
            "safety_cost": mean_safety_cost,
            "safety_limit": float(self.env.safety.safety_limit),
            "safety_violation": float(safety_violation),
            "energy_cost": float(np.mean(energy_costs)),
            "smoothness_cost": float(np.mean(smoothness_costs)),
            "timeout_rate": float(np.mean(timeouts) * 100.0),
            "drop_rate": float(np.mean(drops) * 100.0),
        }
        if not record:
            return metrics["reward"], metrics, None
        recorded = {
            "ee": np.asarray(trace["ee"], dtype=np.float32),
            "object": np.asarray(trace["object"], dtype=np.float32),
            "reward": np.asarray(trace["reward"], dtype=np.float32),
            "camera": np.asarray(trace["camera"], dtype=np.float32),
        }
        return metrics["reward"], metrics, recorded

    def train(self, generations: int = 12, eval_episodes: int = 2) -> dict[str, object]:
        mean = np.zeros(self.policy.param_dim, dtype=np.float32)
        std = np.full(self.policy.param_dim, 0.18, dtype=np.float32)
        history: list[dict[str, float]] = []
        best_params = mean.copy()
        best_objective = -1e9
        best_metrics = {
            "reward": -1e9,
            "success_rate": 0.0,
            "final_distance_m": 1.0,
            "collisions": 0.0,
            "steps": 0.0,
            "safety_cost": 1e9,
            "safety_violation": 1e9,
        }

        for generation in range(generations):
            candidates = self.rng.normal(mean, std, size=(self.population, self.policy.param_dim)).astype(np.float32)
            scored: list[tuple[float, float, np.ndarray, dict[str, float]]] = []
            for params in candidates:
                reward, metrics, _ = self.evaluate(params, episodes=eval_episodes)
                violation = float(metrics["safety_violation"])
                objective = reward - self.safety_multiplier * violation
                scored.append((objective, reward, params, metrics))
            scored.sort(key=lambda item: item[0], reverse=True)
            elites = np.asarray([item[2] for item in scored[: self.elite_count]], dtype=np.float32)
            mean = np.mean(elites, axis=0).astype(np.float32)
            std = np.maximum(np.std(elites, axis=0) * 0.92, 0.035).astype(np.float32)
            elite_safety = float(np.mean([item[3]["safety_cost"] for item in scored[: self.elite_count]]))
            if elite_safety > self.env.safety.safety_limit:
                self.safety_multiplier += self.env.safety.multiplier_step * (
                    elite_safety - self.env.safety.safety_limit
                )
            else:
                self.safety_multiplier = max(
                    self.env.safety.multiplier_init * 0.25,
                    self.safety_multiplier * self.env.safety.multiplier_decay,
                )
            if scored[0][0] > best_objective:
                best_objective = float(scored[0][0])
                best_params = scored[0][2].copy()
                best_metrics = scored[0][3].copy()
            history.append(
                {
                    "generation": float(generation + 1),
                    "objective": float(scored[0][0]),
                    "reward": float(scored[0][1]),
                    "mean_reward": float(np.mean([item[1] for item in scored])),
                    "success_rate": float(scored[0][3]["success_rate"]),
                    "final_distance_m": float(scored[0][3]["final_distance_m"]),
                    "collisions": float(scored[0][3]["collisions"]),
                    "safety_cost": float(scored[0][3]["safety_cost"]),
                    "mean_safety_cost": float(np.mean([item[3]["safety_cost"] for item in scored])),
                    "safety_violation": float(scored[0][3]["safety_violation"]),
                    "safety_multiplier": float(self.safety_multiplier),
                    "policy_std": float(np.mean(std)),
                }
            )

        _, final_metrics, trace = self.evaluate(best_params, episodes=1, record=True)
        return {
            "history": history,
            "best_metrics": final_metrics,
            "policy_params": best_params,
            "trace": trace,
        }

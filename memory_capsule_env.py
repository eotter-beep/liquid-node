"""Memory capsule training environment using Gymnasium.

This module introduces a custom Gymnasium environment that models a pool of
memory capsules. Each capsule represents the memory allocation for an
independent Docker workload. The environment emits observations describing how
much memory each capsule is currently consuming and the size of the next memory
request waiting to be scheduled. An accompanying tabular reinforcement learning
agent learns to place the incoming request into one of the capsules so that
utilization stays balanced without over-allocating the available memory.

Running this module as a script executes a small training session and prints the
training curve so that the behaviour of the agent can be inspected quickly.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:  # Gymnasium is an optional dependency during linting/analysis.
    import gymnasium as gym
    from gymnasium import spaces
except ModuleNotFoundError as exc:  # pragma: no cover - handled gracefully at runtime.
    raise ModuleNotFoundError(
        "memory_capsule_env requires the 'gymnasium' package. Install it with"
        " `pip install gymnasium`."
    ) from exc


class MemoryCapsuleEnv(gym.Env[np.ndarray, int]):
    """Simulates memory pressure across a fixed number of Docker workloads.

    The environment models a row-based view of *capsules*. Each capsule has a
    configurable capacity (default 512 MB). At every step a new request arrives
    that represents memory the Dockerfile would like to consume. The agent must
    route the request to one of the capsules or skip allocation altogether. If a
    capsule lacks sufficient free memory the allocation fails and the agent is
    penalised.

    Observations are a vector consisting of each capsule's utilisation and the
    normalised size of the pending request. Rewards encourage the agent to keep
    capsules near a comfortable operating point (75% full) while severely
    penalising failed allocations or skipped work.
    """

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        *,
        capsules: int = 10,
        capsule_capacity_mb: int = 512,
        episode_length: int = 100,
        request_min_mb: int = 32,
        request_max_mb: int = 256,
        comfort_target: float = 0.75,
        seed: int | None = None,
    ) -> None:
        super().__init__()
        self.capsules = capsules
        self.capacity = float(capsule_capacity_mb)
        self.episode_length = episode_length
        self.request_min = float(request_min_mb)
        self.request_max = float(request_max_mb)
        self.comfort_target = float(comfort_target)
        self._rng = np.random.default_rng(seed)

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.capsules + 1,),
            dtype=np.float32,
        )
        # One discrete action per capsule plus an action to skip allocation.
        self.action_space = spaces.Discrete(self.capsules + 1)

        self._usage = np.zeros(self.capsules, dtype=np.float32)
        self._pending_request: float = 0.0
        self._step_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Dict[str, float] | None = None,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._usage.fill(0.0)
        self._step_count = 0
        self._pending_request = self._sample_request()
        return self._get_observation(), {"request": self._pending_request}

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, float | bool]]:
        self._step_count += 1
        reward = -0.05  # discourage unnecessary steps
        terminated = False
        truncated = False
        info: Dict[str, float | bool] = {}

        request = self._pending_request
        capsule_selected = action < self.capsules

        if capsule_selected:
            capsule_idx = int(action)
            free_capacity = self.capacity - float(self._usage[capsule_idx])
            if request <= free_capacity:
                # Reward the agent for keeping the capsule near the comfort target.
                self._usage[capsule_idx] += request
                utilisation = float(self._usage[capsule_idx]) / self.capacity
                distance = abs(utilisation - self.comfort_target)
                reward += 1.0 - distance
                info.update({"allocated": True, "capsule": capsule_idx})
            else:
                reward -= 1.5  # heavy penalty for failed allocation
                info.update(
                    {
                        "allocated": False,
                        "capsule": capsule_idx,
                        "overflow": request - free_capacity,
                    }
                )
        else:
            reward -= 0.3  # penalty for skipping allocation entirely
            info["allocated"] = False

        if np.all(self._usage >= self.capacity * 0.99):
            terminated = True
            info["full"] = True

        if self._step_count >= self.episode_length:
            truncated = True

        self._pending_request = self._sample_request()
        info["request"] = self._pending_request
        return self._get_observation(), reward, terminated, truncated, info

    def render(self) -> None:
        header = "Capsule".ljust(10) + "Usage (MB)".ljust(15) + "Utilisation"
        print(header)
        print("-" * len(header))
        for index, usage in enumerate(self._usage):
            utilisation = usage / self.capacity
            print(
                f"{index:<10}{usage:>8.1f} MB     {utilisation:>6.2%}"
            )
        print(f"Pending request: {self._pending_request:.1f} MB")

    def close(self) -> None:  # pragma: no cover - for API completeness
        pass

    def _get_observation(self) -> np.ndarray:
        normalised_usage = self._usage / self.capacity
        request_ratio = self._pending_request / self.capacity
        observation = np.append(normalised_usage, request_ratio)
        return observation.astype(np.float32)

    def _sample_request(self) -> float:
        return float(self._rng.uniform(self.request_min, self.request_max))


@dataclass
class CapsuleTrainingConfig:
    episodes: int = 500
    alpha: float = 0.1
    gamma: float = 0.95
    epsilon: float = 0.2
    epsilon_min: float = 0.05
    epsilon_decay: float = 0.995
    discretisation_bins: int = 10


class TabularCapsuleAgent:
    """A simple epsilon-greedy agent trained with Q-learning."""

    def __init__(self, env: MemoryCapsuleEnv, config: CapsuleTrainingConfig) -> None:
        self.env = env
        self.config = config
        self.q_table: Dict[Tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(self.env.action_space.n, dtype=np.float32)
        )
        self._bins = config.discretisation_bins
        self._epsilon = config.epsilon

    def train(self) -> List[float]:
        rewards: List[float] = []
        for _ in range(self.config.episodes):
            observation, _ = self.env.reset()
            state = self._discretise(observation)
            episode_reward = 0.0
            terminated = False
            truncated = False

            while not (terminated or truncated):
                action = self._select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_state = self._discretise(next_obs)
                self._update(state, action, reward, next_state, terminated)
                state = next_state
                episode_reward += reward

            rewards.append(episode_reward)
            self._decay_epsilon()
        return rewards

    def evaluate(self, episodes: int = 10) -> float:
        total_reward = 0.0
        for _ in range(episodes):
            observation, _ = self.env.reset()
            state = self._discretise(observation)
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = int(np.argmax(self.q_table[state]))
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                state = self._discretise(next_obs)
                total_reward += reward
        return total_reward / max(1, episodes)

    def _select_action(self, state: Tuple[int, ...]) -> int:
        if np.random.random() < self._epsilon:
            return int(self.env.action_space.sample())
        return int(np.argmax(self.q_table[state]))

    def _update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        terminated: bool,
    ) -> None:
        best_next = float(np.max(self.q_table[next_state]))
        target = reward if terminated else reward + self.config.gamma * best_next
        td_error = target - self.q_table[state][action]
        self.q_table[state][action] += self.config.alpha * td_error

    def _decay_epsilon(self) -> None:
        self._epsilon = max(
            self.config.epsilon_min, self._epsilon * self.config.epsilon_decay
        )

    def _discretise(self, observation: np.ndarray) -> Tuple[int, ...]:
        clipped = np.clip(observation, 0.0, 1.0)
        scaled = np.floor(clipped * self._bins).astype(int)
        return tuple(int(val) for val in scaled)


def moving_average(values: Iterable[float], window: int) -> List[float]:
    buffer = list(values)
    if not buffer:
        return []
    window = max(1, window)
    averages = []
    cumulative = 0.0
    for index, value in enumerate(buffer):
        cumulative += value
        if index >= window:
            cumulative -= buffer[index - window]
        averages.append(cumulative / min(window, index + 1))
    return averages


def train_memory_capsules(
    *,
    episodes: int = 500,
    capsules: int = 10,
    capsule_capacity_mb: int = 512,
) -> Dict[str, List[float] | float]:
    """Train a tabular agent on the memory capsule environment.

    Returns a dictionary containing the raw episode rewards, the moving average
    of those rewards, and the average reward obtained during an evaluation run.
    """

    env = MemoryCapsuleEnv(capsules=capsules, capsule_capacity_mb=capsule_capacity_mb)
    config = CapsuleTrainingConfig(episodes=episodes)
    agent = TabularCapsuleAgent(env, config)
    reward_history = agent.train()
    trend = moving_average(reward_history, window=max(5, episodes // 10))
    evaluation_score = agent.evaluate()

    return {
        "rewards": reward_history,
        "trend": trend,
        "evaluation_score": evaluation_score,
    }


if __name__ == "__main__":
    results = train_memory_capsules(episodes=200)
    last_rewards = results["rewards"][-10:]
    print("Recent episode rewards:", [round(val, 3) for val in last_rewards])
    print("Moving average trend:", [round(val, 3) for val in results["trend"][-10:]])
    print("Evaluation reward:", round(float(results["evaluation_score"]), 3))

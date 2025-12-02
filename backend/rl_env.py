"""
GridWorld 強化學習環境與多策略 Agent 抽象層。

此模組包含：
1. GridWorldEnv：環境本體，負責狀態轉移與獎勵。
2. BaseAgent 及其子類：提供多種可互換的強化學習策略。
3. Agent 工廠方法：根據字串名稱組建對應策略，方便透過 API 切換。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, Optional, Tuple, Type

import numpy as np


# 定義離散動作對應的位移 (dx, dy)
ACTION_DELTAS: Dict[int, Tuple[int, int]] = {
    0: (0, -1),  # 向上（減少 y）
    1: (0, 1),   # 向下（增加 y）
    2: (-1, 0),  # 向左（減少 x）
    3: (1, 0),   # 向右（增加 x）
}


@dataclass
class TrainingConfig:
    """
    控制訓練流程與獎勵結構的設定。

    Attributes:
        max_episodes: 每次批次訓練允許的最大 episode 數。
        max_steps_per_episode: 單個 episode 可執行的最大步數。
        reward_step_penalty: 每走一步的成本。
        reward_hit_wall: 撞牆時的額外懲罰。
        reward_goal: 抵達目標的獎勵。
    """

    max_episodes: int = 100
    max_steps_per_episode: int = 100
    reward_step_penalty: float = -0.01
    reward_hit_wall: float = -0.05
    reward_goal: float = 1.0

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class StepResult:
    """封裝一步互動後的結果，方便在後端與前端之間傳遞。"""

    state: Dict[str, int]
    reward: float
    done: bool
    info: Dict[str, bool]


class GridWorldEnv:
    """
    簡化版 GridWorld 環境，介面類似 OpenAI Gym。
    """

    def __init__(
        self,
        width: int,
        height: int,
        walls: Optional[Iterable[Tuple[int, int]]] = None,
        start_pos: Tuple[int, int] = (0, 0),
        goal_pos: Tuple[int, int] = (0, 0),
        training_config: Optional[TrainingConfig] = None,
        wall_density: float = 0.2,
        seed: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.start_pos = tuple(start_pos)
        self.goal_pos = tuple(goal_pos)
        self.training_config = training_config or TrainingConfig()
        self.max_steps = self.training_config.max_steps_per_episode
        self.wall_density = wall_density
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.walls = set(walls or [])
        self.position = self.start_pos
        self.step_count = 0
        self.episode = 0
        self._generation_attempts = 100

        if walls is None:
            self.generate_random_map()

    def reset(self) -> Dict[str, int]:
        """
        重置環境並開始新的 episode。
        """
        self.position = self.start_pos
        self.step_count = 0
        self.episode += 1
        return self.get_state()

    def reset_training_counters(self) -> None:
        """
        在切換 Agent 或重新訓練時重置計數器（不自動啟動新 episode）。
        """
        self.position = self.start_pos
        self.step_count = 0
        self.episode = 0

    def set_size(self, size: int) -> None:
        """
        將網格大小切換為 size x size，並重新生成可達成的地圖。
        """
        self.width = size
        self.height = size
        self.generate_random_map()
        self.reset_training_counters()

    def set_training_config(self, config: TrainingConfig) -> None:
        """
        套用新的訓練設定並同步 max_steps。
        """
        self.training_config = config
        self.max_steps = config.max_steps_per_episode
        self.reset_training_counters()

    def get_state(self) -> Dict[str, int]:
        """回傳目前狀態字典，方便序列化為 JSON。"""
        return {
            "x": self.position[0],
            "y": self.position[1],
            "step_count": self.step_count,
            "episode": self.episode,
        }

    def step(self, action: int) -> StepResult:
        """
        執行一步動作並回傳結果。
        """
        if action not in ACTION_DELTAS:
            raise ValueError(f"未知的動作: {action}")

        dx, dy = ACTION_DELTAS[action]
        target = (self.position[0] + dx, self.position[1] + dy)
        hit_wall = False
        done = False
        reward = self.training_config.reward_step_penalty

        if not self._is_valid(target):
            hit_wall = True
            reward += self.training_config.reward_hit_wall
        else:
            self.position = target

        self.step_count += 1

        terminated_reason = ""
        if self.position == self.goal_pos:
            reward += self.training_config.reward_goal
            done = True
            terminated_reason = "goal_reached"
        elif self.step_count >= self.max_steps:
            done = True
            terminated_reason = "max_steps_exceeded"

        info = {
            "hit_wall": hit_wall,
            "goal_reached": self.position == self.goal_pos,
            "terminated_reason": terminated_reason,
        }
        return StepResult(self.get_state(), reward, done, info)

    def get_grid_layout(self) -> Dict[str, object]:
        """回傳網格佈局資訊，供前端渲染使用。"""
        return {
            "width": self.width,
            "height": self.height,
            "walls": [list(pos) for pos in sorted(self.walls)],
            "start_pos": list(self.start_pos),
            "goal_pos": list(self.goal_pos),
        }

    def _is_valid(self, pos: Tuple[int, int]) -> bool:
        """判斷目標座標是否可走。"""
        x, y = pos
        return (
            0 <= x < self.width
            and 0 <= y < self.height
            and pos not in self.walls
        )

    def generate_random_map(
        self,
        wall_density: Optional[float] = None,
        max_attempts: Optional[int] = None,
    ) -> None:
        """
        隨機生成一組 (start, goal, walls)，確保 start 能走到 goal。

        使用 BFS 驗證是否存在可行路徑；若失敗則重新嘗試，避免生成無法完成的任務。
        """
        density = wall_density if wall_density is not None else self.wall_density
        density = float(np.clip(density, 0.05, 0.4))
        attempts = max_attempts or self._generation_attempts

        for _ in range(attempts):
            start = (
                int(self.rng.integers(0, self.width)),
                int(self.rng.integers(0, self.height)),
            )
            goal = start
            while goal == start:
                goal = (
                    int(self.rng.integers(0, self.width)),
                    int(self.rng.integers(0, self.height)),
                )

            wall_target = int(self.width * self.height * density)
            walls: set[Tuple[int, int]] = set()
            while len(walls) < wall_target:
                cell = (
                    int(self.rng.integers(0, self.width)),
                    int(self.rng.integers(0, self.height)),
                )
                if cell == start or cell == goal:
                    continue
                walls.add(cell)

            if self._has_valid_path(start, goal, walls):
                self.start_pos = start
                self.goal_pos = goal
                self.walls = walls
                self.position = start
                self.step_count = 0
                return

        raise RuntimeError("在限定次數內未能生成可達成任務的地圖，請調整參數")

    def _has_valid_path(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        walls: Iterable[Tuple[int, int]],
    ) -> bool:
        """
        使用 BFS 檢查從 start 到 goal 是否存在可用路徑。

        若搜尋過程中被牆阻擋或超出邊界則忽略，直到找到目標或耗盡搜尋空間。
        """
        walls_set = set(walls)
        visited = set([start])
        queue: deque[Tuple[int, int]] = deque([start])

        while queue:
            x, y = queue.popleft()
            if (x, y) == goal:
                return True
            for dx, dy in ACTION_DELTAS.values():
                nx, ny = x + dx, y + dy
                if (
                    0 <= nx < self.width
                    and 0 <= ny < self.height
                    and (nx, ny) not in walls_set
                    and (nx, ny) not in visited
                ):
                    visited.add((nx, ny))
                    queue.append((nx, ny))
        return False


# ---------------------------------------------------------------------------
# Agent 抽象層
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """
    所有策略的基類，提供共用的超參數與介面。
    """

    algorithm_name = "base"

    def __init__(
        self,
        width: int,
        height: int,
        num_actions: int = 4,
        alpha: float = 0.2,
        gamma: float = 0.95,
        epsilon: float = 0.1,
        seed: Optional[int] = None,
        **_: Dict[str, object],
    ) -> None:
        self.width = width
        self.height = height
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def select_action(self, state: Dict[str, int]) -> int:
        """給定狀態選擇動作。"""

    @abstractmethod
    def update(
        self,
        state: Dict[str, int],
        action: int,
        reward: float,
        next_state: Dict[str, int],
        done: bool,
    ) -> None:
        """使用一筆轉移 (s, a, r, s') 更新策略。"""

    def reset_episode(self) -> None:
        """重置 episode 級別的暫存變數（預設無需處理）。"""

    def get_hyperparams(self) -> Dict[str, float]:
        """回傳可序列化的超參數，方便 API 回傳。"""
        return {
            "alpha": float(self.alpha),
            "gamma": float(self.gamma),
            "epsilon": float(self.epsilon),
            "num_actions": int(self.num_actions),
        }


class TabularAgent(BaseAgent):
    """
    以表格 (width x height x num_actions) 儲存 Q 值的 Agent 基底類。
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.q_table = np.zeros(
            (self.width, self.height, self.num_actions), dtype=np.float32
        )

    def _clip_state(self, state: Dict[str, int]) -> Tuple[int, int]:
        x = int(np.clip(state["x"], 0, self.width - 1))
        y = int(np.clip(state["y"], 0, self.height - 1))
        return x, y

    def _epsilon_greedy(self, state: Dict[str, int]) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        x, y = self._clip_state(state)
        return int(np.argmax(self.q_table[x, y]))


class QLearningAgent(TabularAgent):
    """
    表格 Q-Learning：離線評估下一狀態的最大 Q 值來更新。
    """

    algorithm_name = "q_learning"

    def select_action(self, state: Dict[str, int]) -> int:
        return self._epsilon_greedy(state)

    def update(
        self,
        state: Dict[str, int],
        action: int,
        reward: float,
        next_state: Dict[str, int],
        done: bool,
    ) -> None:
        x, y = self._clip_state(state)
        nx, ny = self._clip_state(next_state)
        current_q = self.q_table[x, y, action]
        next_q = 0.0 if done else float(np.max(self.q_table[nx, ny]))
        target = reward + self.gamma * next_q
        self.q_table[x, y, action] = current_q + self.alpha * (target - current_q)


class SARSAgent(TabularAgent):
    """
    表格 SARSA (採用 Expected SARSA 形式)：以策略期望值更新 Q(s, a)。
    """

    algorithm_name = "sarsa"

    def select_action(self, state: Dict[str, int]) -> int:
        return self._epsilon_greedy(state)

    def update(
        self,
        state: Dict[str, int],
        action: int,
        reward: float,
        next_state: Dict[str, int],
        done: bool,
    ) -> None:
        x, y = self._clip_state(state)
        current_q = self.q_table[x, y, action]
        expected_next = 0.0
        if not done:
            nx, ny = self._clip_state(next_state)
            next_values = self.q_table[nx, ny]
            greedy_value = float(next_values[np.argmax(next_values)])
            expected_random = float(np.sum(next_values)) / self.num_actions
            expected_next = (
                (1 - self.epsilon) * greedy_value
                + self.epsilon * expected_random
            )

        target = reward + self.gamma * expected_next
        self.q_table[x, y, action] = current_q + self.alpha * (target - current_q)


class RandomAgent(BaseAgent):
    """
    完全隨機策略，不會更新任何權重，方便測試渲染管線。
    """

    algorithm_name = "random"

    def select_action(self, state: Dict[str, int]) -> int:  # noqa: ARG002
        return int(self.rng.integers(0, self.num_actions))

    def update(  # noqa: D401
        self,
        state: Dict[str, int],  # noqa: ARG002
        action: int,  # noqa: ARG002
        reward: float,  # noqa: ARG002
        next_state: Dict[str, int],  # noqa: ARG002
        done: bool,  # noqa: ARG002
    ) -> None:
        """隨機策略不需要更新。"""

    def get_hyperparams(self) -> Dict[str, float]:
        return {"strategy": "random", "num_actions": int(self.num_actions)}


# ---------------------------------------------------------------------------
# Agent 工廠與註冊表
# ---------------------------------------------------------------------------

AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    QLearningAgent.algorithm_name: QLearningAgent,
    SARSAgent.algorithm_name: SARSAgent,
    RandomAgent.algorithm_name: RandomAgent,
}

DEFAULT_AGENT_NAME = QLearningAgent.algorithm_name


def create_agent(
    algorithm: str,
    width: int,
    height: int,
    **params: object,
) -> BaseAgent:
    """
    根據字串名稱產生對應 Agent。

    Args:
        algorithm: 例如 "q_learning"、"sarsa"、"random"。
        width/height: 用於建立 tabular Q-table。
        params: 任意自訂超參數 (alpha/gamma/epsilon...)。
    """
    key = algorithm.lower()
    agent_cls = AGENT_REGISTRY.get(key)
    if agent_cls is None:
        available = ", ".join(sorted(AGENT_REGISTRY))
        raise ValueError(f"未知的算法 {algorithm}，可選：{available}")

    # 預設動作數量 = ACTION_DELTAS 長度，允許被覆寫
    params.setdefault("num_actions", len(ACTION_DELTAS))
    return agent_cls(width=width, height=height, **params)


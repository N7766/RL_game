"""
FastAPI 入口：提供 GridWorld 強化學習實驗室的 HTTP API。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .models import (
    ResetResponse,
    StepRequest,
    StepResponse,
    TrajectoryStep,
    TrainEpisodeResponse,
)
from .rl_env import (
    DEFAULT_AGENT_NAME,
    ACTION_DELTAS,
    BaseAgent,
    GridWorldEnv,
    TrainingConfig,
    create_agent,
)


SUPPORTED_MAP_SIZES = (5, 10, 15, 20, 25)
DEFAULT_MAP_SIZE = 10
DEFAULT_WALL_DENSITY = 0.2


def _default_env() -> GridWorldEnv:
    """建立預設環境，可根據 API 重新配置大小。"""
    return GridWorldEnv(
        width=DEFAULT_MAP_SIZE,
        height=DEFAULT_MAP_SIZE,
        training_config=TrainingConfig(
            max_episodes=100,
            max_steps_per_episode=100,
            reward_step_penalty=-0.02,
            reward_hit_wall=-0.1,
            reward_goal=1.0,
        ),
        wall_density=DEFAULT_WALL_DENSITY,
    )


class AgentConfigRequest(BaseModel):
    """POST /config/agent 的請求模型。"""

    algorithm: str = Field(..., description="例如 q_learning / sarsa / random")
    params: Dict[str, float] = Field(
        default_factory=dict, description="超參數，可省略"
    )


class AgentConfigResponse(BaseModel):
    """POST /config/agent 的回應模型。"""

    ok: bool
    current_algorithm: str
    used_params: Dict[str, float]


class MapConfigRequest(BaseModel):
    """POST /config/map 的請求模型。"""

    size: int = Field(..., description="限定為 5/10/15/20/25 其中之一，單位為格數")


class MapConfigResponse(BaseModel):
    """POST /config/map 的回應模型。"""

    ok: bool
    size: int


class TrainingConfigRequest(BaseModel):
    """POST /config/training 的請求模型。"""

    max_episodes: Optional[int] = Field(None, gt=0)
    max_steps_per_episode: Optional[int] = Field(None, gt=0)
    reward_step_penalty: Optional[float] = None
    reward_hit_wall: Optional[float] = None
    reward_goal: Optional[float] = None


class TrainingConfigResponse(BaseModel):
    """POST /config/training 的回應模型。"""

    ok: bool
    config: Dict[str, float]


class TrainRunRequest(BaseModel):
    """POST /train/run 的請求模型。"""

    episodes: Optional[int] = Field(None, gt=0)
    record_last: int = Field(1, ge=0, description="記錄最後幾個 episode 的軌跡")


class TrainRunResponse(BaseModel):
    """POST /train/run 的回應。"""

    trained_episodes: int
    avg_total_reward: float
    best_total_reward: float
    algorithm: str
    last_episode_total_reward: Optional[float]
    last_episode_success: Optional[bool]
    last_episode_terminated_reason: Optional[str]
    last_episode_steps: Optional[int]
    last_episode_trajectory: List[TrajectoryStep] = Field(default_factory=list)


env = _default_env()
current_agent_name = DEFAULT_AGENT_NAME
current_agent_params: Dict[str, float] = {}
agent: BaseAgent = create_agent(current_agent_name, env.width, env.height)


def _set_agent(
    algorithm: str,
    params: Optional[Dict[str, float]] = None,
) -> BaseAgent:
    """
    建立並套用新的 Agent，同時重置環境計數器。
    """
    global agent, current_agent_name, current_agent_params
    effective_params = dict(params or current_agent_params)
    new_agent = create_agent(
        algorithm,
        env.width,
        env.height,
        **effective_params,
    )
    agent = new_agent
    current_agent_name = algorithm.lower()
    current_agent_params = effective_params
    env.reset_training_counters()
    agent.reset_episode()
    return new_agent


app = FastAPI(
    title="GridWorld RL Lab",
    description="一個簡化版的 GridWorld 強化學習環境與可視化後端。",
    version="0.2.0",
)

# 允許本機靜態前端存取（可視需求調整來源清單）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/env/layout")
def get_layout() -> dict:
    """
    GET /env/layout

    回傳網格佈局資訊，包含寬、高、牆壁、起點與終點。
    前端初始化 Three.js 場景時會用到這些資料。
    """
    return env.get_grid_layout()


@app.post("/env/reset", response_model=ResetResponse)
def reset_environment() -> ResetResponse:
    """
    POST /env/reset

    重置環境並開始新的 episode，返回初始狀態。
    """
    agent.reset_episode()
    state = env.reset()
    return ResetResponse(state=state)


@app.post("/env/step", response_model=StepResponse)
def step_environment(request: Optional[StepRequest] = None) -> StepResponse:
    """
    POST /env/step

    執行單一步驟：
        - 若 body 為空，將由當前 Agent 自動挑動作。
        - 若 body 指定 action，則採用客製動作（仍會觸發 Agent 更新）。
    """
    current_state = env.get_state()
    if request is None:
        action = agent.select_action(current_state)
    else:
        if request.action not in ACTION_DELTAS:
            raise HTTPException(status_code=400, detail="action 必須介於 0~3")
        action = request.action

    result = env.step(action)
    agent.update(current_state, action, result.reward, result.state, result.done)
    if result.done:
        agent.reset_episode()
    terminated_reason = result.info.get("terminated_reason", "") if result.info else ""
    success = terminated_reason == "goal_reached"
    return StepResponse(
        state=result.state,
        reward=result.reward,
        done=result.done,
        terminated_reason=terminated_reason,
        success=success,
        algorithm=current_agent_name,
    )


@app.post("/train/one_episode", response_model=TrainEpisodeResponse)
def train_one_episode() -> TrainEpisodeResponse:
    """
    POST /train/one_episode

    讓後端自動完成一個 episode 的訓練，並回傳該次軌跡，方便前端回放。
    """
    agent.reset_episode()
    state = env.reset()
    trajectory: List[TrajectoryStep] = []
    total_reward = 0.0
    done = False

    final_reason = ""
    while not done:
        action = agent.select_action(state)
        result = env.step(action)
        agent.update(state, action, result.reward, result.state, result.done)

        trajectory.append(
            TrajectoryStep(
                x=result.state["x"],
                y=result.state["y"],
                action=action,
                reward=result.reward,
                step=result.state["step_count"],
            )
        )
        total_reward += result.reward
        state = result.state
        done = result.done
        final_reason = result.info.get("terminated_reason", "") if result.info else ""

    agent.reset_episode()
    success = final_reason == "goal_reached"
    steps_used = state["step_count"]
    return TrainEpisodeResponse(
        trajectory=trajectory,
        total_reward=total_reward,
        success=success,
        terminated_reason=final_reason or "",
        steps_used=steps_used,
        algorithm=current_agent_name,
    )


@app.post("/config/agent", response_model=AgentConfigResponse)
def configure_agent(request: AgentConfigRequest) -> AgentConfigResponse:
    """
    POST /config/agent

    切換當前使用的強化學習策略，並可以同時調整超參數。
    """
    try:
        new_agent = _set_agent(request.algorithm, request.params)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return AgentConfigResponse(
        ok=True,
        current_algorithm=current_agent_name,
        used_params=new_agent.get_hyperparams(),
    )


@app.post("/config/map", response_model=MapConfigResponse)
def configure_map(request: MapConfigRequest) -> MapConfigResponse:
    """
    POST /config/map

    以指定尺寸重新隨機生成可達成任務的地圖，並保留目前策略設定。
    """
    size = request.size
    if size not in SUPPORTED_MAP_SIZES:
        allowed = ", ".join(map(str, SUPPORTED_MAP_SIZES))
        raise HTTPException(status_code=400, detail=f"size 必須為 {allowed} 之一")

    try:
        env.set_size(size)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # 重新建立一次 agent，確保 Q-table 維度符合新的地圖大小
    _set_agent(current_agent_name, current_agent_params)

    return MapConfigResponse(ok=True, size=size)


@app.post("/config/training", response_model=TrainingConfigResponse)
def configure_training(request: TrainingConfigRequest) -> TrainingConfigResponse:
    """
    POST /config/training

    調整訓練流程（最大 episode / 步數）與獎勵設定。
    """
    incoming = request.dict(exclude_unset=True)
    current = env.training_config.to_dict()
    current.update(incoming)
    new_config = TrainingConfig(**current)
    env.set_training_config(new_config)
    return TrainingConfigResponse(ok=True, config=new_config.to_dict())


@app.post("/train/run", response_model=TrainRunResponse)
def train_run(request: TrainRunRequest) -> TrainRunResponse:
    """
    POST /train/run

    依據指定期數批量訓練，並回傳統計與最後 episode 的軌跡。
    """
    target_episodes = request.episodes or env.training_config.max_episodes
    episodes_to_run = max(1, min(target_episodes, env.training_config.max_episodes))
    record_last = max(0, request.record_last)

    rewards: List[float] = []
    best_reward = float("-inf")
    last_episode_reward: Optional[float] = None
    last_episode_success: Optional[bool] = None
    last_episode_reason: Optional[str] = None
    last_episode_steps: Optional[int] = None
    last_episode_trajectory: List[TrajectoryStep] = []

    for episode_idx in range(episodes_to_run):
        agent.reset_episode()
        state = env.reset()
        episode_reward = 0.0
        trajectory: List[TrajectoryStep] = []
        done = False
        final_reason = ""

        while not done:
            action = agent.select_action(state)
            result = env.step(action)
            agent.update(state, action, result.reward, result.state, result.done)

            trajectory.append(
                TrajectoryStep(
                    x=result.state["x"],
                    y=result.state["y"],
                    action=action,
                    reward=result.reward,
                    step=result.state["step_count"],
                )
            )

            episode_reward += result.reward
            state = result.state
            done = result.done
            final_reason = (
                result.info.get("terminated_reason", "") if result.info else ""
            )

        agent.reset_episode()
        rewards.append(episode_reward)
        best_reward = max(best_reward, episode_reward)

        if record_last > 0 and episode_idx >= episodes_to_run - record_last:
            last_episode_reward = episode_reward
            last_episode_trajectory = trajectory
            last_episode_success = final_reason == "goal_reached"
            last_episode_reason = final_reason or ""
            last_episode_steps = state["step_count"]

    avg_reward = float(sum(rewards) / len(rewards)) if rewards else 0.0

    return TrainRunResponse(
        trained_episodes=episodes_to_run,
        avg_total_reward=avg_reward,
        best_total_reward=float(best_reward) if rewards else 0.0,
        algorithm=current_agent_name,
        last_episode_total_reward=last_episode_reward,
        last_episode_success=last_episode_success,
        last_episode_terminated_reason=last_episode_reason,
        last_episode_steps=last_episode_steps,
        last_episode_trajectory=last_episode_trajectory,
    )


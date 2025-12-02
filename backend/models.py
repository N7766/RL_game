"""
Pydantic 資料模型，定義後端 API 的請求與回應結構。
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field, conint


class StateModel(BaseModel):
    """描述環境狀態 (x, y, 當前步數, episode 編號)。"""

    x: conint(ge=0)
    y: conint(ge=0)
    step_count: conint(ge=0) = Field(..., description="當前 episode 的步數計數器")
    episode: conint(ge=0) = Field(..., description="從 1 開始的 episode 編號")


class StepRequest(BaseModel):
    """POST /env/step 的請求，action 代表 4 個方向之一。"""

    action: conint(ge=0, le=3)


class ResetResponse(BaseModel):
    """POST /env/reset 的回應。"""

    state: StateModel


class StepResponse(BaseModel):
    """POST /env/step 的回應。"""

    state: StateModel
    reward: float
    done: bool
    terminated_reason: str | None = Field(
        default="",
        description="episode 終止原因，例如 goal_reached、max_steps_exceeded",
    )
    success: bool = Field(
        default=False, description="當次 episode 是否成功到達目標（done == True）"
    )
    algorithm: str = Field(default="q_learning", description="當前使用的算法名稱")


class TrajectoryStep(BaseModel):
    """單一步驟的軌跡資料。"""

    x: int
    y: int
    action: int
    reward: float
    step: int


class TrainEpisodeResponse(BaseModel):
    """POST /train/one_episode 的回應，包含整段軌跡與總回饋。"""

    trajectory: List[TrajectoryStep]
    total_reward: float
    success: bool
    terminated_reason: str
    steps_used: int
    algorithm: str


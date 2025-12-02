# Three.js + Python RL 网格实验室

一个用于演示 GridWorld 强化学习训练与可视化的最简实验项目：后端以 FastAPI 提供环境与训练 API，前端使用 Three.js 渲染 3D 网格，并实时展示小机器人在格子中的移动轨迹。

## 项目结构

```
.
├── backend/
│   ├── __init__.py        # 让 backend 目录成为 Python 包
│   ├── main.py            # FastAPI 入口，提供 REST API
│   ├── models.py          # Pydantic 数据结构
│   └── rl_env.py          # GridWorld 环境 + Q-Learning 逻辑
├── frontend/
│   ├── index.html         # 页面骨架，挂载 Three.js 与控制面板
│   ├── style.css          # 控制面板与画布布局
│   └── main.js            # Three.js 场景 + 与后端交互
└── README.md
```

## 后端运行

1. 安装依赖（建议放在虚拟环境内）：

   ```bash
   cd /Users/nn/GAAAME/RL_game
   python3 -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   pip install fastapi uvicorn[standard] numpy
   ```

2. 启动 FastAPI 服务：

   ```bash
   uvicorn backend.main:app --reload --port 8000
   ```

   服务启动后可通过 `http://127.0.0.1:8000/docs` 预览自动生成的 Swagger 文档。

## 前端使用

前端仅使用原生 HTML/CSS/JS 与 CDN 版 Three.js，无需构建工具。推荐使用任意静态服务器（如 VSCode Live Server、`python -m http.server` 等）托管 `frontend/` 目录：

```bash
cd /Users/nn/GAAAME/RL_game/frontend
python -m http.server 5500
```

然后在浏览器打开 `http://127.0.0.1:5500` 即可看到 3D 网格实验室。若你直接从文件系统打开 `index.html` 也能运行，但某些浏览器会阻止 `fetch`，因此推荐使用本地静态服务器。

## API 说明

所有接口前缀为 `http://127.0.0.1:8000`：

| 方法 | 路径                | 说明                                                                 |
|------|---------------------|----------------------------------------------------------------------|
| GET  | `/env/layout`       | 返回网格宽高、墙壁、起点、终点，用于前端初始化场景。                   |
| POST | `/env/reset`        | 重置环境，返回初始 `state = {x, y, step_count, episode}`。           |
| POST | `/env/step`         | 执行单步动作 `{action: 0~3}`，返回 `state、reward、done`。            |
| POST | `/train/one_episode`| 后端完整跑一回合 Q-learning，返回轨迹 `trajectory[]` 与 `total_reward`。 |

详见 `backend/main.py` 与 `backend/models.py` 中的注释说明。

## 交互流程

1. 打开前端页面后点击「初始化场景」，前端会：
   - 读取 `/env/layout` 创建 3D 网格；
   - 调用 `/env/reset` 将机器人放回起点。
2. 「单步 Step」会随机挑一个动作调用 `/env/step`，并在 Three.js 中平滑移动机器人。
3. 「播放一次训练」会请求 `/train/one_episode`，后端运行一整个 episode 并返回轨迹；前端以动画方式依次播放每一步，同时显示奖励与步数。

你可以在 `backend/rl_env.py` 中修改地图、奖励结构或替换 Q-learning 逻辑，在 `frontend/main.js` 中接入更复杂的策略或 UI 控制。整个项目维持最小依赖，方便快速迭代成自己的 RL 可视化实验室。欢迎扩展！


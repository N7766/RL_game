/**
 * Three.js 前端主程式：負責拉取後端資料並可視化 GridWorld。
 */
const API_BASE = "http://127.0.0.1:8000";
const MOVE_DURATION_MS = 250;
const MAP_SIZES = [5, 10, 15, 20, 25];
const ALGORITHM_LABELS = {
  q_learning: "Q-Learning",
  sarsa: "SARSA",
  random: "Random Policy",
};
let currentAlgorithmKey = "q_learning";

// DOM 參考
const elements = {
  canvasContainer: document.getElementById("canvas-container"),
  initButton: document.getElementById("init-button"),
  resetButton: document.getElementById("reset-button"),
  stepButton: document.getElementById("step-button"),
  playEpisodeButton: document.getElementById("play-episode-button"),
  episodeValue: document.getElementById("episode-value"),
  stepValue: document.getElementById("step-value"),
  rewardValue: document.getElementById("reward-value"),
  statusValue: document.getElementById("status-value"),
  mapSizeSelect: document.getElementById("map-size-select"),
  trainEpisodesInput: document.getElementById("train-episodes-input"),
  maxEpisodesInput: document.getElementById("max-episodes-input"),
  maxStepsInput: document.getElementById("max-steps-input"),
  stepPenaltyInput: document.getElementById("step-penalty-input"),
  wallPenaltyInput: document.getElementById("wall-penalty-input"),
  goalRewardInput: document.getElementById("goal-reward-input"),
  applyTrainingButton: document.getElementById("apply-training-button"),
  trainRunButton: document.getElementById("train-run-button"),
  avgRewardValue: document.getElementById("avg-reward-value"),
  bestRewardValue: document.getElementById("best-reward-value"),
  lastRewardValue: document.getElementById("last-reward-value"),
  resultModal: document.getElementById("episode-result-modal"),
  resultTitle: document.getElementById("result-title"),
  resultMessage: document.getElementById("result-message"),
  resultReward: document.getElementById("result-reward"),
  resultSteps: document.getElementById("result-steps"),
  resultAlgo: document.getElementById("result-algo"),
  modalCloseButton: document.getElementById("modal-close-button"),
  modalReplayButton: document.getElementById("modal-replay-button"),
};

// Three.js 相關變數
let scene;
let camera;
let renderer;
let robotMesh;
let goalMarkerMesh;
let goalMarkerBaseColor;
let layoutCache = null;
let animationState = null;
let resizeHandler = null;
let renderLoopStarted = false;
let celebrationHandle = null;

let isInitialized = false;
let cumulativeReward = 0;

elements.resetButton.disabled = true;
elements.stepButton.disabled = true;
elements.playEpisodeButton.disabled = true;

elements.initButton.addEventListener("click", handleInit);
elements.resetButton.addEventListener("click", handleReset);
elements.stepButton.addEventListener("click", handleSingleStep);
elements.playEpisodeButton.addEventListener("click", handlePlayEpisode);
elements.mapSizeSelect?.addEventListener("change", handleMapSizeChange);
elements.applyTrainingButton?.addEventListener("click", handleApplyTrainingConfig);
elements.trainRunButton?.addEventListener("click", handleBatchTraining);
elements.modalCloseButton?.addEventListener("click", () => {
  hideEpisodeResultModal();
});
elements.modalReplayButton?.addEventListener("click", async () => {
  hideEpisodeResultModal();
  if (isInitialized) {
    await handlePlayEpisode();
  } else {
    setStatus("請先初始化場景再開始訓練。");
  }
});

/**
 * 初始化流程：抓 layout -> 建場景 -> 重置環境。
 */
async function handleInit() {
  setStatus("正在載入環境...");
  toggleControls(true);
  try {
    await loadLayoutAndReset(false);
    isInitialized = true;
    setStatus("初始化完成，可以開始互動！");
    cumulativeReward = 0;
    updateReward(0, true);
  } catch (error) {
    console.error(error);
    setStatus(`初始化失敗：${error.message}`);
  } finally {
    toggleControls(false);
  }
}

/**
 * 重新開局：呼叫後端 reset。
 */
async function handleReset() {
  if (!isInitialized) return;
  toggleControls(true);
  hideEpisodeResultModal();
  setStatus("重置環境中...");
  try {
    const data = await fetchJSON("/env/reset", { method: "POST" });
    cumulativeReward = 0;
    updateReward(0, true);
    applyState(data.state, true);
    await setRobotPosition(data.state.x, data.state.y, false);
    setStatus("環境已重置。");
  } catch (error) {
    setStatus(`重置失敗：${error.message}`);
  } finally {
    toggleControls(false);
  }
}

/**
 * 單步互動：目前先隨機挑動作。
 */
async function handleSingleStep() {
  if (!isInitialized) return;
  toggleControls(true);
  hideEpisodeResultModal();
  setStatus("執行單步...");
  try {
    const action = Math.floor(Math.random() * 4);
    const data = await fetchJSON("/env/step", {
      method: "POST",
      body: JSON.stringify({ action }),
    });
    applyState(data.state, false);
    updateReward(data.reward);
    await setRobotPosition(data.state.x, data.state.y, true);
    if (data.done) {
      currentAlgorithmKey = data.algorithm || currentAlgorithmKey;
      showEpisodeResultModal({
        success: data.success,
        terminatedReason: data.terminated_reason || "",
        totalReward: cumulativeReward,
        stepsUsed: data.state.step_count,
        algorithm: getAlgorithmLabel(currentAlgorithmKey),
      });
      setStatus(
        data.success
          ? "任務完成，恭喜！"
          : "Episode 結束，請調整參數或重試。"
      );
    } else {
      setStatus(`動作 ${action} 完成，reward=${data.reward.toFixed(2)}`);
    }
  } catch (error) {
    setStatus(`單步執行失敗：${error.message}`);
  } finally {
    toggleControls(false);
  }
}

/**
 * 後端自動訓練一個 episode 並回放。
 */
async function handlePlayEpisode() {
  if (!isInitialized) return;
  toggleControls(true);
  hideEpisodeResultModal();
  setStatus("後端訓練中...");
  try {
    const data = await fetchJSON("/train/one_episode", { method: "POST" });
    setStatus("回放訓練軌跡中...");
    await playbackTrajectory({
      trajectory: data.trajectory,
      totalReward: data.total_reward,
      success: data.success,
      terminatedReason: data.terminated_reason,
      stepsUsed: data.steps_used,
      algorithm: data.algorithm,
    });
    setStatus(`回放完成，總獎勵 ${data.total_reward.toFixed(2)}`);
  } catch (error) {
    setStatus(`播放失敗：${error.message}`);
  } finally {
    toggleControls(false);
  }
}

async function handleMapSizeChange(event) {
  const size = Number(event.target.value);
  if (!MAP_SIZES.includes(size)) {
    setStatus("不支援的地圖尺寸");
    return;
  }
  toggleControls(true);
  setStatus(`切換至 ${size}x${size} 地圖並重新生成...`);
  try {
    await fetchJSON("/config/map", {
      method: "POST",
      body: JSON.stringify({ size }),
    });
    await loadLayoutAndReset(false);
    isInitialized = true;
    setStatus(`地圖已更新為 ${size}x${size}`);
  } catch (error) {
    setStatus(`切換地圖失敗：${error.message}`);
  } finally {
    toggleControls(false);
  }
}

async function handleApplyTrainingConfig() {
  toggleControls(true);
  setStatus("更新訓練配置...");
  try {
    const payload = collectTrainingConfig();
    const response = await fetchJSON("/config/training", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    updateTrainingInputs(response.config);
    setStatus("訓練配置已更新");
  } catch (error) {
    setStatus(`訓練配置更新失敗：${error.message}`);
  } finally {
    toggleControls(false);
  }
}

async function handleBatchTraining() {
  if (!isInitialized) {
    setStatus("請先初始化場景與地圖");
    return;
  }
  toggleControls(true);
  hideEpisodeResultModal();
  setStatus("批量訓練中...");
  try {
    const episodes = Number(elements.trainEpisodesInput?.value);
    const payload = { record_last: 1 };
    if (Number.isFinite(episodes) && episodes > 0) {
      payload.episodes = episodes;
    }
    const result = await fetchJSON("/train/run", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    updateTrainingStats(result);
    setStatus("播放最新訓練軌跡...");
    await playbackTrajectory({
      trajectory: result.last_episode_trajectory || [],
      totalReward: result.last_episode_total_reward ?? 0,
      success: result.last_episode_success ?? false,
      terminatedReason: result.last_episode_terminated_reason ?? "",
      stepsUsed: result.last_episode_steps ?? 0,
      algorithm: result.algorithm,
    });
    setStatus("批量訓練與回放完成");
  } catch (error) {
    setStatus(`批量訓練失敗：${error.message}`);
  } finally {
    toggleControls(false);
  }
}

async function loadLayoutAndReset(animateRobot = false) {
  hideEpisodeResultModal();
  layoutCache = await fetchJSON("/env/layout");
  initThreeScene(layoutCache);
  const resetData = await fetchJSON("/env/reset", { method: "POST" });
  applyState(resetData.state, true);
  await setRobotPosition(resetData.state.x, resetData.state.y, animateRobot);
  updateReward(0, true);
  updateMapSelector(layoutCache.width);
}

/**
 * 建立 Three.js 場景與基本物件。
 */
function initThreeScene(layout) {
  if (resizeHandler) {
    window.removeEventListener("resize", resizeHandler);
    resizeHandler = null;
  }
  stopGoalCelebration();
  if (renderer) {
    renderer.dispose();
    renderer = null;
  }
  scene = undefined;
  camera = undefined;
  robotMesh = undefined;
  goalMarkerMesh = undefined;
  goalMarkerBaseColor = undefined;
  elements.canvasContainer.innerHTML = "";

  const { clientWidth, clientHeight } = elements.canvasContainer;
  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070d);

  camera = new THREE.PerspectiveCamera(
    55,
    clientWidth / clientHeight,
    0.1,
    1000
  );
  const maxSpan = Math.max(layout.width, layout.height);
  camera.position.set(maxSpan, maxSpan, maxSpan);
  camera.lookAt(0, 0, 0);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(clientWidth, clientHeight);
  elements.canvasContainer.appendChild(renderer.domElement);

  const ambient = new THREE.AmbientLight(0xffffff, 0.7);
  const directional = new THREE.DirectionalLight(0xffffff, 0.6);
  directional.position.set(10, 20, 10);
  scene.add(ambient, directional);

  buildGrid(layout);
  buildMarkers(layout);

  resizeHandler = () => {
    const { clientWidth, clientHeight } = elements.canvasContainer;
    camera.aspect = clientWidth / clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(clientWidth, clientHeight);
  };
  window.addEventListener("resize", resizeHandler);

  if (!renderLoopStarted) {
    renderLoopStarted = true;
    requestAnimationFrame(renderLoop);
  }
}

/**
 * 建立地板、格線與牆壁。
 */
function buildGrid(layout) {
  const floorGeometry = new THREE.PlaneGeometry(layout.width, layout.height);
  const floorMaterial = new THREE.MeshStandardMaterial({
    color: 0x0b0f1c,
    side: THREE.DoubleSide,
  });
  const floor = new THREE.Mesh(floorGeometry, floorMaterial);
  floor.rotation.x = -Math.PI / 2;
  scene.add(floor);

  const gridSize = Math.max(layout.width, layout.height);
  const gridHelper = new THREE.GridHelper(
    gridSize,
    gridSize,
    0x4f7df5,
    0x25324f
  );
  gridHelper.position.y = 0.01;
  gridHelper.scale.set(layout.width / gridSize, 1, layout.height / gridSize);
  scene.add(gridHelper);

  const wallGeometry = new THREE.BoxGeometry(0.9, 0.9, 0.9);
  const wallMaterial = new THREE.MeshStandardMaterial({ color: 0x7f8c8d });
  layout.walls.forEach(([x, y]) => {
    const cube = new THREE.Mesh(wallGeometry, wallMaterial);
    cube.position.copy(gridToWorld(x, y));
    cube.position.y = 0.45;
    scene.add(cube);
  });
}

/**
 * 建立起點、終點與機器人 Mesh。
 */
function buildMarkers(layout) {
  const startGeometry = new THREE.BoxGeometry(0.8, 0.3, 0.8);
  const startMaterial = new THREE.MeshStandardMaterial({ color: 0x3498db });
  const startMesh = new THREE.Mesh(startGeometry, startMaterial);
  startMesh.position.copy(gridToWorld(layout.start_pos[0], layout.start_pos[1]));
  startMesh.position.y = 0.15;
  scene.add(startMesh);

  const goalGeometry = new THREE.BoxGeometry(0.8, 0.3, 0.8);
  const goalMaterial = new THREE.MeshStandardMaterial({ color: 0xf1c40f });
  const goalMesh = new THREE.Mesh(goalGeometry, goalMaterial);
  goalMesh.position.copy(gridToWorld(layout.goal_pos[0], layout.goal_pos[1]));
  goalMesh.position.y = 0.15;
  scene.add(goalMesh);
  goalMarkerMesh = goalMesh;
  goalMarkerBaseColor = goalMesh.material.color.clone();

  const robotGeometry = new THREE.BoxGeometry(0.7, 0.7, 0.7);
  const robotMaterial = new THREE.MeshStandardMaterial({
    color: 0xff6b6b,
    emissive: 0x2b0b0b,
    metalness: 0.2,
    roughness: 0.4,
  });
  robotMesh = new THREE.Mesh(robotGeometry, robotMaterial);
  robotMesh.position.copy(gridToWorld(layout.start_pos[0], layout.start_pos[1]));
  robotMesh.position.y = 0.35;
  scene.add(robotMesh);
}

/**
 * 渲染主迴圈，處理動畫插值。
 */
function renderLoop(timestamp) {
  if (animationState && robotMesh) {
    const progress = Math.min(
      1,
      (timestamp - animationState.start) / animationState.duration
    );
    robotMesh.position.lerpVectors(
      animationState.from,
      animationState.to,
      progress
    );
    if (progress >= 1) {
      const resolver = animationState.resolver;
      animationState = null;
      resolver?.();
    }
  }
  if (renderer && scene && camera) {
    renderer.render(scene, camera);
  }
  requestAnimationFrame(renderLoop);
}

/**
 * 平滑移動機器人到指定格子。
 */
function setRobotPosition(x, y, animated = true) {
  if (!robotMesh || !layoutCache) return Promise.resolve();
  const target = gridToWorld(x, y);
  target.y = robotMesh.position.y;
  if (!animated) {
    robotMesh.position.copy(target);
    return Promise.resolve();
  }
  return new Promise((resolve) => {
    animationState = {
      from: robotMesh.position.clone(),
      to: target,
      start: performance.now(),
      duration: MOVE_DURATION_MS,
      resolver: resolve,
    };
  });
}

/**
 * 將網格座標轉為 Three.js 世界座標。
 */
function gridToWorld(x, y) {
  if (!layoutCache) {
    throw new Error("尚未載入網格佈局");
  }
  const offsetX = (layoutCache.width - 1) / 2;
  const offsetY = (layoutCache.height - 1) / 2;
  const worldX = (x - offsetX);
  const worldZ = (y - offsetY);
  return new THREE.Vector3(worldX, 0, worldZ);
}

/**
 * 更新 UI 狀態顯示。
 */
function applyState(state, resetStep) {
  if (state.episode !== undefined) {
    elements.episodeValue.textContent = state.episode;
  }
  if (resetStep) {
    elements.stepValue.textContent = 0;
  } else if (state.step_count !== undefined) {
    elements.stepValue.textContent = state.step_count;
  }
}

/**
 * 更新獎勵累計。
 */
function updateReward(delta, reset = false) {
  if (reset) {
    cumulativeReward = 0;
  } else {
    cumulativeReward += delta;
  }
  elements.rewardValue.textContent = cumulativeReward.toFixed(2);
}

function setStatus(text) {
  elements.statusValue.textContent = text;
}

function toggleControls(disabled) {
  elements.initButton.disabled = disabled || isInitialized;
  elements.resetButton.disabled = disabled || !isInitialized;
  elements.stepButton.disabled = disabled || !isInitialized;
  elements.playEpisodeButton.disabled = disabled || !isInitialized;
  if (elements.mapSizeSelect) {
    elements.mapSizeSelect.disabled = disabled;
  }
  if (elements.applyTrainingButton) {
    elements.applyTrainingButton.disabled = disabled;
  }
  if (elements.trainRunButton) {
    elements.trainRunButton.disabled = disabled || !isInitialized;
  }
}

/**
 * 小工具：包裝 fetch，處理 JSON 與錯誤。
 */
async function fetchJSON(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || response.statusText);
  }
  return response.json();
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function updateMapSelector(size) {
  if (elements.mapSizeSelect) {
    elements.mapSizeSelect.value = String(size);
  }
}

function collectTrainingConfig() {
  const payload = {};
  const maxEpisodes = Number(elements.maxEpisodesInput?.value);
  const maxSteps = Number(elements.maxStepsInput?.value);
  const stepPenalty = Number(elements.stepPenaltyInput?.value);
  const wallPenalty = Number(elements.wallPenaltyInput?.value);
  const goalReward = Number(elements.goalRewardInput?.value);

  if (Number.isFinite(maxEpisodes) && maxEpisodes > 0) {
    payload.max_episodes = maxEpisodes;
  }
  if (Number.isFinite(maxSteps) && maxSteps > 0) {
    payload.max_steps_per_episode = maxSteps;
  }
  if (Number.isFinite(stepPenalty)) {
    payload.reward_step_penalty = stepPenalty;
  }
  if (Number.isFinite(wallPenalty)) {
    payload.reward_hit_wall = wallPenalty;
  }
  if (Number.isFinite(goalReward)) {
    payload.reward_goal = goalReward;
  }
  return payload;
}

function updateTrainingInputs(config) {
  if (!config) return;
  if (elements.maxEpisodesInput && config.max_episodes !== undefined) {
    elements.maxEpisodesInput.value = config.max_episodes;
  }
  if (elements.maxStepsInput && config.max_steps_per_episode !== undefined) {
    elements.maxStepsInput.value = config.max_steps_per_episode;
  }
  if (elements.stepPenaltyInput && config.reward_step_penalty !== undefined) {
    elements.stepPenaltyInput.value = config.reward_step_penalty;
  }
  if (elements.wallPenaltyInput && config.reward_hit_wall !== undefined) {
    elements.wallPenaltyInput.value = config.reward_hit_wall;
  }
  if (elements.goalRewardInput && config.reward_goal !== undefined) {
    elements.goalRewardInput.value = config.reward_goal;
  }
}

function updateTrainingStats(stats) {
  if (!stats) return;
  if (stats.algorithm) {
    currentAlgorithmKey = stats.algorithm;
  }
  if (
    elements.avgRewardValue &&
    typeof stats.avg_total_reward === "number" &&
    Number.isFinite(stats.avg_total_reward)
  ) {
    elements.avgRewardValue.textContent = stats.avg_total_reward.toFixed(2);
  }
  if (
    elements.bestRewardValue &&
    typeof stats.best_total_reward === "number" &&
    Number.isFinite(stats.best_total_reward)
  ) {
    elements.bestRewardValue.textContent = stats.best_total_reward.toFixed(2);
  }
  if (
    elements.lastRewardValue &&
    typeof stats.last_episode_total_reward === "number" &&
    Number.isFinite(stats.last_episode_total_reward)
  ) {
    elements.lastRewardValue.textContent =
      stats.last_episode_total_reward.toFixed(2);
  }
}

function getAlgorithmLabel(key) {
  return ALGORITHM_LABELS[key] || key || "Unknown";
}

function showEpisodeResultModal({
  success = false,
  terminatedReason = "",
  totalReward = 0,
  stepsUsed = 0,
  algorithm = getAlgorithmLabel(currentAlgorithmKey),
} = {}) {
  if (!elements.resultModal) return;
  const isSuccess = success || terminatedReason === "goal_reached";
  const reasonText = {
    goal_reached: "小機器人成功到達目標！",
    max_steps_exceeded: "已達最大步數，建議調整策略或參數後重試。",
  }[terminatedReason] || "本輪未成功，可以調整參數或算法後再試。";

  elements.resultTitle.textContent = isSuccess ? "任務完成！" : "任務未成功";
  elements.resultMessage.textContent = reasonText;
  elements.resultAlgo.textContent = algorithm;
  elements.resultSteps.textContent = Number.isFinite(stepsUsed)
    ? stepsUsed
    : "-";
  elements.resultReward.textContent = Number.isFinite(totalReward)
    ? totalReward.toFixed(2)
    : "-";
  elements.resultModal.classList.remove("hidden");
  elements.resultModal.classList.add("visible");

  if (isSuccess) {
    triggerGoalCelebration();
  } else {
    stopGoalCelebration();
  }
}

function hideEpisodeResultModal() {
  if (!elements.resultModal) return;
  elements.resultModal.classList.remove("visible");
  elements.resultModal.classList.add("hidden");
  stopGoalCelebration();
}

function triggerGoalCelebration() {
  if (!goalMarkerMesh) return;
  stopGoalCelebration();
  const startTime = performance.now();
  const baseColor = goalMarkerBaseColor?.clone() ?? new THREE.Color(0xf1c40f);
  const celebrationColor = new THREE.Color(0x00ff88);

  const animate = (time) => {
    const elapsed = time - startTime;
    const pulse = 1 + 0.2 * Math.sin(elapsed / 150);
    goalMarkerMesh.scale.set(pulse, 1, pulse);
    goalMarkerMesh.material.color.lerpColors(
      baseColor,
      celebrationColor,
      0.5 + 0.5 * Math.sin(elapsed / 200)
    );
    if (elapsed < 2000) {
      celebrationHandle = requestAnimationFrame(animate);
    } else {
      stopGoalCelebration();
    }
  };

  celebrationHandle = requestAnimationFrame(animate);
}

function stopGoalCelebration() {
  if (celebrationHandle) {
    cancelAnimationFrame(celebrationHandle);
    celebrationHandle = null;
  }
  if (goalMarkerMesh) {
    goalMarkerMesh.scale.set(1, 1, 1);
    if (goalMarkerBaseColor) {
      goalMarkerMesh.material.color.copy(goalMarkerBaseColor);
    }
  }
}

async function playbackTrajectory({
  trajectory = [],
  totalReward = 0,
  success = false,
  terminatedReason = "",
  stepsUsed = 0,
  algorithm,
} = {}) {
  if (algorithm) {
    currentAlgorithmKey = algorithm;
  }

  const hasPath =
    layoutCache && Array.isArray(trajectory) && trajectory.length > 0;

  if (hasPath) {
    await setRobotPosition(
      layoutCache.start_pos[0],
      layoutCache.start_pos[1],
      false
    );
    updateReward(0, true);
    for (const step of trajectory) {
      await setRobotPosition(step.x, step.y, true);
      applyState(
        { x: step.x, y: step.y, step_count: step.step, episode: "-" },
        false
      );
      updateReward(step.reward);
      await wait(MOVE_DURATION_MS * 0.6);
    }
  } else if (Number.isFinite(totalReward)) {
    elements.rewardValue.textContent = totalReward.toFixed(2);
  }

  const fallbackSteps =
    stepsUsed ||
    (Array.isArray(trajectory) && trajectory.length
      ? trajectory[trajectory.length - 1].step
      : 0);

  showEpisodeResultModal({
    success,
    terminatedReason,
    totalReward,
    stepsUsed: fallbackSteps,
    algorithm: getAlgorithmLabel(currentAlgorithmKey),
  });

  if (elements.lastRewardValue && Number.isFinite(totalReward)) {
    elements.lastRewardValue.textContent = totalReward.toFixed(2);
  }
}


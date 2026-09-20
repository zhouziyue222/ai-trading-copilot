const state = {
  sessionId: null,
  epoch: 0,
  pollBusy: false,
  pollFailures: 0,
  runtime: { status: "idle", active_runs: [] },
  activeView: "analysis",
  subscriptions: [],
  selectedSymbols: new Set(),
  currentRunId: null,
  pollTimer: null,
  progressTimer: null,
  activeReportKey: null,
  activeTraceSpanId: null,
  traceRunId: null,
  currentTrace: null,
  lastRunPayload: null,
  activeNodeName: null,
  expandedActivityDetails: new Set(),
  recentRuns: [],
  cancelPending: false,
  ragStatus: null,
  observability: null,
  portfolio: null,
  riskPosition: null,
  riskDirty: false,
  riskSaving: false,
  memories: [],
  memoryCounts: {},
};

const $ = (id) => document.getElementById(id);
const lastRunStorageKey = "aiTradingCopilot.lastRunId";

const primaryWorkflowNodes = [
  "Load Persona Markdown",
  "Load Subscription Symbols",
  "Retrieve Memories",
  "Technical Position",
  "News Sentiment",
  "Fundamental Analysis",
  "Trader",
  "Risk Check",
  "Portfolio Manager",
  "Explain Run",
  "Persist Trace",
];


const subscriptionStatusLabels = {
  observing: "观察中",
  near_opportunity: "接近机会",
  actionable: "可执行",
  risk_elevated: "风险升高",
  not_compatible: "不兼容",
};

const nodeStatusLabels = {
  pending: "待运行",
  running: "运行中",
  succeeded: "成功",
  failed: "失败",
  skipped: "跳过",
};

const runStatusLabels = {
  idle: "空闲",
  interrupted: "已中断",
  pending: "待执行",
  running: "运行中",
  succeeded: "成功",
  failed: "失败",
  skipped: "跳过",
  cancelled: "已取消",
  degraded: "降级完成",
};

const directionLabels = {
  buy: "买入",
  hold: "持有",
  reduce: "减仓",
  sell: "卖出",
  short: "做空",
  cover: "平空",
  watch: "观察",
};

const executionStatusLabels = {
  alert_only: "仅提醒",
  portfolio_decided: "组合决策",
  confirmation_required: "需要确认",
};
const preferredReportOrder = [
  "run_explanation",
  "risk_check",
  "portfolio_manager",
  "simulated_broker_orders",
  "trader",
  "news_sentiment",
  "fundamental_analysis",
  "technical_position",
  "futu_portfolio",
  "run_audit",
];

async function api(path, options = {}) {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });
  if (!response.ok) {
    let message = response.statusText;
    try {
      const payload = await response.json();
      message = typeof payload.detail === "string" ? payload.detail : "请检查输入参数";
      if (payload.request_id) message += `（故障编号 ${payload.request_id}）`;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (payload && typeof payload === "object") payload._session_id = response.headers.get("X-Session-ID");
  return payload;
}

function switchView(view) {
  state.activeView = view;
  document.querySelectorAll(".nav-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.view === view);
  });
  document.querySelectorAll(".view").forEach((section) => {
    section.classList.toggle("active", section.id === `view-${view}`);
  });
  if (view === "rag") {
    loadRagStatus().catch(showError);
  }
  if (view === "portfolio" && !state.portfolio) {
    loadPortfolio().catch(showError);
  }
  if (view === "risk") {
    loadRiskPosition().catch(showRiskError);
  }
  if (view === "memory") {
    loadMemories().catch(showMemoryError);
  }
}

function showMemoryError(error) {
  const target = $("memoryFeedback");
  if (target) target.textContent = error ? String(error.message || error) : "";
}

async function loadMemories() {
  showMemoryError("");
  const status = $("memoryStatusFilter")?.value || "";
  const query = status ? `?status=${encodeURIComponent(status)}` : "";
  const payload = await api(`/api/memories${query}`);
  state.memories = payload.items || [];
  state.memoryCounts = payload.counts || {};
  renderMemoryCounts();
  renderMemoryRows();
}

function renderMemoryCounts() {
  const counts = state.memoryCounts || {};
  const cards = [
    { label: "Candidate", value: counts.candidate || 0, tone: "warning" },
    { label: "Shadow", value: counts.shadow || 0, tone: "neutral" },
    { label: "Approved", value: counts.approved || 0, tone: "success" },
    { label: "Deprecated / Rejected", value: (counts.deprecated || 0) + (counts.rejected || 0), tone: "danger" },
  ];
  $("memoryCounts").innerHTML = cards.map(summaryCard).join("");
}

function renderMemoryRows() {
  const target = $("memoryRows");
  if (!target) return;
  if (!state.memories.length) {
    target.innerHTML = '<tr><td colspan="8" class="empty-cell">当前筛选下没有记忆。</td></tr>';
    return;
  }
  target.innerHTML = state.memories
    .map((item) => `
      <tr>
        <td><span class="inline-status ${memoryTone(item.status)}">${escapeHtml(item.status)}</span></td>
        <td>${escapeHtml(`${item.memory_kind || "-"} / ${item.scope || "-"}`)}</td>
        <td>${escapeHtml((item.symbols || []).join(", ") || "-")}</td>
        <td class="memory-lesson">
          <strong>${escapeHtml(item.lesson)}</strong>
          <small>${escapeHtml(item.memory_id || "-")}</small>
        </td>
        <td>${escapeHtml(item.sample_count || (item.evidence_run_ids || []).length || 0)}</td>
        <td>${item.confidence == null ? "-" : escapeHtml(Number(item.confidence).toFixed(2))}</td>
        <td>${escapeHtml(item.version || 1)}</td>
        <td><div class="memory-actions">${memoryActionButtons(item)}</div></td>
      </tr>
    `)
    .join("");
}

function memoryTone(status) {
  if (status === "approved") return "status-success";
  if (status === "candidate") return "status-warning";
  if (["deprecated", "rejected"].includes(status)) return "status-danger";
  return "status-neutral";
}

function memoryActionButtons(item) {
  const id = escapeHtml(item.memory_id || "");
  if (item.status === "candidate") {
    return `
      <button class="ghost-button" type="button" data-memory-action="shadow" data-memory-id="${id}">进入 Shadow</button>
      <button class="ghost-button danger-text" type="button" data-memory-action="rejected" data-memory-id="${id}">拒绝</button>`;
  }
  if (item.status === "shadow") {
    return `
      <button class="ghost-button" type="button" data-memory-action="evaluate" data-memory-id="${id}">评测</button>
      <button type="button" data-memory-action="approved" data-memory-id="${id}">门禁晋升</button>
      <button class="ghost-button danger-text" type="button" data-memory-action="rejected" data-memory-id="${id}">拒绝</button>`;
  }
  if (item.status === "approved") {
    return `<button class="ghost-button danger-text" type="button" data-memory-action="deprecated" data-memory-id="${id}">停用</button>`;
  }
  return `<button class="ghost-button" type="button" data-memory-action="candidate" data-memory-id="${id}">重新候选</button>`;
}

async function handleMemoryAction(action, memoryId) {
  showMemoryError("");
  if (action === "evaluate") {
    const payload = await api(`/api/memories/${encodeURIComponent(memoryId)}/evaluate`, { method: "POST" });
    const evaluation = payload.evaluation || {};
    showMemoryError(
      evaluation.eligible
        ? "评测通过：已满足晋升门禁。"
        : `评测未通过：${(evaluation.reasons || []).join("；") || "证据不足"}`,
    );
    return;
  }
  await api(`/api/memories/${encodeURIComponent(memoryId)}/transition`, {
    method: "POST",
    body: JSON.stringify({
      status: action,
      actor: "local_ui_user",
      reason: action === "approved" ? "shadow_evaluation_passed" : "manual_memory_review",
    }),
  });
  await loadMemories();
}

async function recordMemoryOutcome(event) {
  event.preventDefault();
  showMemoryError("");
  const optionalNumber = (id) => {
    const value = $(id).value.trim();
    return value === "" ? null : Number(value);
  };
  await api("/api/memory-outcomes", {
    method: "POST",
    body: JSON.stringify({
      run_id: $("outcomeRunId").value.trim(),
      symbol: $("outcomeSymbol").value.trim(),
      horizon_days: Number($("outcomeHorizon").value || 5),
      realized_return: optionalNumber("outcomeReturn"),
      benchmark_return: optionalNumber("outcomeBenchmark"),
      max_drawdown: optionalNumber("outcomeDrawdown"),
      source: "manual_ui",
    }),
  });
  showMemoryError("延迟市场结果已保存，可重新运行 Shadow 评测。");
}

function showError(error) {
  const errors = $("errors");
  if (errors) {
    errors.textContent = error ? String(error.message || error) : "";
  }
}

function showRiskError(error) {
  const target = $("riskFeedback");
  if (!target) return;
  target.className = error ? "risk-banner status-danger" : "risk-banner status-neutral";
  target.innerHTML = error
    ? `<strong>加载或保存失败</strong><span>${escapeHtml(String(error.message || error))}</span>`
    : `<strong>风控与仓位</strong><span>配置未加载。</span>`;
}

async function loadRiskPosition() {
  showRiskError("");
  const payload = await api("/api/risk-position");
  state.riskPosition = payload;
  state.riskDirty = false;
  const background = [];
  if (!state.portfolio) background.push(loadPortfolio().catch(showRiskError));
  if (!state.subscriptions.length) background.push(loadSubscriptions().catch(showRiskError));
  await Promise.all(background);
  renderRiskPosition();
}

function renderRiskPosition() {
  const config = state.riskPosition;
  if (!config) return;
  const persona = config.persona || {};
  setPercentInput("riskMaxDrawdown", "riskMaxDrawdownRange", persona.max_portfolio_drawdown);
  setPercentInput("riskMaxPosition", "riskMaxPositionRange", persona.max_single_position_weight);
  setPercentInput("riskGrossExposure", "riskGrossExposureRange", persona.max_gross_exposure);
  setPercentInput("riskMinWeight", "riskMinWeightRange", persona.default_position_weight_min);
  setPercentInput("riskMaxWeight", "riskMaxWeightRange", persona.default_position_weight_max);
  setPercentInput("riskTargetVolatility", "riskTargetVolatilityRange", persona.target_annual_volatility);
  setPercentInput("riskAccountRisk", "riskAccountRiskRange", persona.account_risk_per_trade);
  $("riskAtrStopMultiple").value = Number(persona.atr_stop_multiple ?? 2).toFixed(1);
  $("riskAtrStopMultipleRange").value = persona.atr_stop_multiple ?? 2;
  $("riskMinRewardRisk").value = Number(persona.minimum_reward_risk ?? 2).toFixed(1);
  $("riskMinRewardRiskRange").value = persona.minimum_reward_risk ?? 2;
  setPercentInput("riskMaxDistance", "riskMaxDistanceRange", persona.max_distance_to_support_pct);
  $("riskVolLookback").value = persona.volatility_lookback_days ?? 60;
  $("riskAtrWindow").value = persona.atr_window ?? 14;
  renderRiskCurrentPositions();
  renderRiskTargetRows();
  updateRiskFeedback();
}

function setPercentInput(numberId, rangeId, value) {
  const percent = Math.round((Number(value) || 0) * 1000) / 10;
  $(numberId).value = percent;
  $(rangeId).value = percent;
}

function syncRiskInputs(changed) {
  const pairs = [
    ["riskMaxDrawdown", "riskMaxDrawdownRange"],
    ["riskMaxPosition", "riskMaxPositionRange"],
    ["riskGrossExposure", "riskGrossExposureRange"],
    ["riskMinWeight", "riskMinWeightRange"],
    ["riskMaxWeight", "riskMaxWeightRange"],
    ["riskTargetVolatility", "riskTargetVolatilityRange"],
    ["riskAccountRisk", "riskAccountRiskRange"],
    ["riskAtrStopMultiple", "riskAtrStopMultipleRange"],
    ["riskMinRewardRisk", "riskMinRewardRiskRange"],
    ["riskMaxDistance", "riskMaxDistanceRange"],
  ];
  for (const [numberId, rangeId] of pairs) {
    if (changed.id !== numberId && changed.id !== rangeId) continue;
    const value = Number(changed.value);
    if (!Number.isFinite(value)) continue;
    $(numberId).value = value;
    $(rangeId).value = value;
  }
}

function renderRiskCurrentPositions() {
  const container = $("riskCurrentPositions");
  if (!container) return;
  const entries = Object.entries(state.portfolio?.position_weights || {}).sort(
    (a, b) => Math.abs(b[1]) - Math.abs(a[1]),
  );
  container.innerHTML = entries.length
    ? entries
        .map(
          ([symbol, weight]) => `
            <article class="position-row">
              <div>
                <strong>${escapeHtml(symbol)}</strong>
                <span>${formatPercent(weight)}</span>
              </div>
              <div class="weight-track"><span style="width: ${Math.min(Math.abs(Number(weight) || 0), 1) * 100}%"></span></div>
            </article>
          `,
        )
        .join("")
    : '<div class="empty-state compact"><strong>暂无当前持仓</strong><span>连接 Futu 模拟账户后显示。</span></div>';
}

function renderRiskTargetRows() {
  const container = $("riskTargetRows");
  if (!container) return;
  const targets = state.riskPosition?.target_weights || {};
  const symbols = riskAvailableSymbols();
  if (!symbols.length) {
    container.innerHTML = '<div class="empty-state compact"><strong>暂无可用标的</strong><span>请先在“我的自选股”添加关注。</span></div>';
    renderRiskTargetAddOptions();
    return;
  }
  container.innerHTML = symbols
    .map((symbol) => {
      const target = targets[symbol];
      const hasTarget = Object.prototype.hasOwnProperty.call(targets, symbol);
      const current = state.portfolio?.position_weights?.[symbol] || 0;
      const targetPercent = hasTarget ? Math.round((Number(target) || 0) * 1000) / 10 : 0;
      return `
        <article class="risk-target-row">
          <div>
            <strong>${escapeHtml(symbol)}</strong>
            <span class="current-weight">当前 ${formatPercent(current)}</span>
            <button class="ghost-button ${hasTarget ? "danger-text" : ""}" type="button" data-${hasTarget ? "remove" : "add"}-risk-target="${escapeHtml(symbol)}">${hasTarget ? "移除" : "设置"}</button>
          </div>
          <div class="target-inputs">
            <input type="number" min="-100" max="100" step="0.1" value="${targetPercent}" data-risk-target-number="${escapeHtml(symbol)}" ${hasTarget ? "" : "disabled"} />
            <input type="range" min="-100" max="100" step="0.1" value="${targetPercent}" data-risk-target-range="${escapeHtml(symbol)}" ${hasTarget ? "" : "disabled"} />
          </div>
        </article>
      `;
    })
    .join("");
  renderRiskTargetAddOptions();
}

function riskAvailableSymbols() {
  const symbols = new Set();
  for (const item of state.subscriptions || []) symbols.add(item.symbol);
  for (const symbol of Object.keys(state.portfolio?.position_weights || {})) symbols.add(symbol);
  for (const symbol of Object.keys(state.riskPosition?.target_weights || {})) symbols.add(symbol);
  return Array.from(symbols).sort();
}

function renderRiskTargetAddOptions() {
  const select = $("riskTargetAddSelect");
  if (!select) return;
  const existing = new Set(Object.keys(state.riskPosition?.target_weights || {}));
  const options = riskAvailableSymbols()
    .filter((symbol) => !existing.has(symbol))
    .map((symbol) => `<option value="${escapeHtml(symbol)}">${escapeHtml(symbol)}</option>`)
    .join("");
  select.innerHTML = options || '<option value="">暂无可添加标的</option>';
  $("riskTargetAdd").disabled = !options;
}

function updateRiskFeedback() {
  const banner = $("riskFeedback");
  const save = $("saveRiskPosition");
  if (!banner || !save) return;
  const config = state.riskPosition;
  const errors = validateRiskPosition();
  const source = config?.source === "user" ? "用户配置" : "默认配置";
  const updated = config?.updated_at ? ` · 保存于 ${formatDateTime(config.updated_at)}` : "";
  banner.className = errors.length
    ? "risk-banner status-warning"
    : state.riskDirty
      ? "risk-banner status-warning"
      : "risk-banner status-success";
  banner.innerHTML = errors.length
    ? `<strong>请修正输入</strong><span>${escapeHtml(errors.join("；"))}</span>`
    : state.riskDirty
      ? `<strong>有未保存修改</strong><span>${source}${updated}</span>`
      : `<strong>配置已加载</strong><span>${source}${updated}</span>`;
  save.disabled = !state.riskDirty || errors.length > 0 || state.riskSaving;
  updateRiskTargetGross();
}

function validateRiskPosition() {
  const errors = [];
  const values = collectRiskPersona();
  if (!(values.default_position_weight_min <= values.default_position_weight_max)) {
    errors.push("默认仓位区间最小值不能大于最大值");
  }
  if (values.default_position_weight_max > values.max_single_position_weight) {
    errors.push("默认仓位上限不能超过单标的上限");
  }
  if (values.target_annual_volatility <= 0) {
    errors.push("目标年化波动率必须大于 0");
  }
  if (values.account_risk_per_trade <= 0) {
    errors.push("单笔风险预算必须大于 0");
  }
  if (values.volatility_lookback_days < 5 || values.volatility_lookback_days > 252) {
    errors.push("波动率回看窗口必须在 5-252 之间");
  }
  if (values.atr_window < 2 || values.atr_window > 100) {
    errors.push("ATR 窗口必须在 2-100 之间");
  }
  if (values.minimum_reward_risk < 1 || values.minimum_reward_risk > 5) {
    errors.push("最低收益风险比建议在 1.0-5.0 之间");
  }
  if (values.max_distance_to_support_pct < 0 || values.max_distance_to_support_pct > 0.10) {
    errors.push("距支撑上限建议在 0%-10% 之间");
  }
  const targets = collectRiskTargetWeights();
  for (const [symbol, weight] of Object.entries(targets)) {
    if (!Number.isFinite(weight) || weight < -1 || weight > 1) {
      errors.push(`${symbol} 目标仓位必须在 -100% 到 100% 之间`);
    }
  }
  return errors;
}

function collectRiskPersona() {
  const persona = state.riskPosition?.persona || {};
  return {
    ...persona,
    max_portfolio_drawdown: percentValue("riskMaxDrawdown"),
    max_single_position_weight: percentValue("riskMaxPosition"),
    max_gross_exposure: percentValue("riskGrossExposure"),
    default_position_weight_min: percentValue("riskMinWeight"),
    default_position_weight_max: percentValue("riskMaxWeight"),
    target_annual_volatility: percentValue("riskTargetVolatility"),
    account_risk_per_trade: percentValue("riskAccountRisk"),
    atr_stop_multiple: floatValue("riskAtrStopMultiple"),
    minimum_reward_risk: floatValue("riskMinRewardRisk"),
    max_distance_to_support_pct: percentValue("riskMaxDistance"),
    volatility_lookback_days: intValue("riskVolLookback"),
    atr_window: intValue("riskAtrWindow"),
  };
}

function collectRiskTargetWeights() {
  const output = {};
  document.querySelectorAll("[data-risk-target-number]").forEach((input) => {
    if (input.disabled) return;
    const symbol = input.dataset.riskTargetNumber;
    output[symbol] = Number(input.value) / 100;
  });
  return output;
}

function percentValue(id) {
  const value = Number($(id)?.value);
  return Number.isFinite(value) ? value / 100 : 0;
}

function floatValue(id) {
  const value = Number($(id)?.value);
  return Number.isFinite(value) ? value : 0;
}

function intValue(id) {
  const value = parseInt($(id)?.value || "0", 10);
  return Number.isFinite(value) ? value : 0;
}

function updateRiskTargetGross() {
  const target = $("riskTargetGross");
  if (!target) return;
  const weights = Object.values(collectRiskTargetWeights());
  const gross = weights.reduce((sum, weight) => sum + Math.abs(weight || 0), 0);
  target.textContent = `目标总敞口 ${formatPercent(gross)}`;
}

async function saveRiskPosition() {
  const errors = validateRiskPosition();
  if (errors.length) {
    updateRiskFeedback();
    return;
  }
  showRiskError("");
  state.riskSaving = true;
  updateRiskFeedback();
  try {
    const payload = await api("/api/risk-position", {
      method: "PUT",
      body: JSON.stringify({
        persona: collectRiskPersona(),
        target_weights: collectRiskTargetWeights(),
      }),
    });
    state.riskPosition = payload;
    state.riskDirty = false;
    renderRiskPosition();
    showRiskSuccess("设置已保存，将在下一次分析中生效。");
  } finally {
    state.riskSaving = false;
    updateRiskFeedback();
  }
}

async function resetRiskPosition() {
  showRiskError("");
  const payload = await api("/api/risk-position/reset", { method: "POST" });
  state.riskPosition = payload;
  state.riskDirty = false;
  renderRiskPosition();
  showRiskSuccess("已恢复默认配置。");
}

function showRiskSuccess(message) {
  const banner = $("riskFeedback");
  if (!banner) return;
  banner.className = "risk-banner status-success";
  banner.innerHTML = `<strong>已保存</strong><span>${escapeHtml(message)}</span>`;
}

async function loadSubscriptions() {
  showError("");
  const payload = await api("/api/subscriptions");
  state.subscriptions = payload.items || [];
  const availableSymbols = new Set(state.subscriptions.map((item) => item.symbol));
  state.selectedSymbols = new Set(
    Array.from(state.selectedSymbols).filter((symbol) => availableSymbols.has(symbol)),
  );
  renderSubscriptions();
  renderSelectedCount();
}

async function loadRagStatus(probe = false) {
  const payload = await api(`/api/rag/status?probe=${probe}`);
  state.ragStatus = payload;
  renderRagStatus(payload);
}

async function loadObservabilityStatus() {
  const payload = await api("/api/observability/status");
  state.observability = payload;
  renderObservabilityStatus(payload, state.currentTrace);
}

function renderObservabilityStatus(payload = {}, trace = null) {
  const container = $("observabilityStatus");
  if (!container) return;
  const localEnabled = payload.local_trace_enabled !== false;
  const otelConfigured = payload.otel_configured === true;
  const exported = trace?.otel_export_enabled === true || trace?.otel?.otel_export_enabled === true;
  const statusText = otelConfigured
    ? exported
      ? "OTel exported"
      : "OTel configured"
    : "Local trace only";
  const tone = otelConfigured ? "status-success" : "status-neutral";
  const jaegerUrl = payload.jaeger_ui_url || "http://127.0.0.1:16686";
  container.className = `observability-status ${tone}`;
  container.innerHTML = `
    <div>
      <strong>${escapeHtml(statusText)}</strong>
      <span>local trace ${localEnabled ? "on" : "off"} · service ${escapeHtml(payload.service_name || "-")}</span>
    </div>
    <div>
      <span>${escapeHtml(payload.otlp_endpoint || "OTLP endpoint not set")}</span>
      <a href="${escapeHtml(jaegerUrl)}" target="_blank" rel="noreferrer">Jaeger UI</a>
    </div>
  `;
}

function renderRagStatus(payload = {}) {
  const status = $("ragStatus");
  const summary = $("ragStatusSummary");
  const diagnostics = $("ragDiagnostics");
  if (!status || !summary || !diagnostics) return;

  const available = payload.available === true;
  const unavailable = payload.available === false;
  const checking = !available && !unavailable;
  const count = payload.document_count ?? "-";
  const reason = payload.error || (unavailable ? "未返回具体错误。" : "");
  const tone = available ? "status-success" : checking ? "status-neutral" : "status-warning";

  status.textContent = available
    ? `知识库可用 · ${count} docs`
    : checking
      ? "尚未检查"
      : "Fallback 检索";
  status.className = `${tone} inline-status`;
  summary.className = `rag-status-summary ${tone}`;
  summary.innerHTML = available
    ? `<strong>向量知识库可用</strong><span>当前集合可读取，基本面分析师可以使用 RAG 工具。</span>`
    : checking
      ? `<strong>正在检查 RAG 状态</strong><span>正在打开 Chroma 并检查 embedding 配置。</span>`
      : `<strong>Chroma/Embedding 不可用</strong><span>${escapeHtml(humanizeRagError(reason))}</span>`;

  diagnostics.innerHTML = [
    diagnosticCard("后端", payload.backend || "-"),
    diagnosticCard("文档数", String(count)),
    diagnosticCard("Chroma 路径", payload.path || "-"),
    diagnosticCard("Embedding", payload.embedding_model || "-"),
    diagnosticCard("Probe", payload.probe_status || "-"),
    diagnosticCard("范围", payload.scope || "-"),
  ].join("");
}

function humanizeRagError(error) {
  if (!error) return "未返回具体错误。";
  if (error.includes("OPENAI_API_KEY") || error.includes("DASHSCOPE_API_KEY")) {
    return "未配置 OPENAI_API_KEY 或 DASHSCOPE_API_KEY。配置后刷新状态，或重新导入资料。";
  }
  if (error.includes("chromadb") || error.includes("Chroma")) {
    return `Chroma 初始化失败：${error}`;
  }
  return error;
}
function diagnosticCard(label, value) {
  return `
    <article class="diagnostic-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(value)}</strong>
    </article>
  `;
}

async function ingestDefaults() {
  showError("");
  const button = $("ingestDefaults");
  button.disabled = true;
  try {
    const result = await api("/api/rag/ingest-defaults", { method: "POST" });
    showError(result.errors?.length ? result.errors.join("\n") : "");
    await loadRagStatus();
  } finally {
    button.disabled = false;
  }
}

async function ingestOnlineResearch() {
  showError("");
  const button = $("ingestOnline");
  button.disabled = true;
  try {
    const result = await api("/api/rag/ingest-online", {
      method: "POST",
      body: JSON.stringify({
        symbols: Array.from(state.selectedSymbols),
        manual_symbols: $("ragSymbols").value,
        look_back_days: Number($("ragLookBackDays").value || 7),
        trade_date: $("tradeDate").value || null,
      }),
    });
    showError(result.errors?.length ? result.errors.join("\n") : "");
    await loadRagStatus();
  } finally {
    button.disabled = false;
  }
}

async function ingestTextKnowledge() {
  showError("");
  const text = $("ragText").value;
  if (!text.trim()) {
    showError(new Error("请先填写新资料。"));
    return;
  }
  const button = $("ingestText");
  button.disabled = true;
  try {
    const result = await api("/api/rag/ingest-text", {
      method: "POST",
      body: JSON.stringify({
        title: $("ragTitle").value || "Manual fundamental note",
        text,
        symbols: Array.from(state.selectedSymbols),
        tags: splitCsv($("ragTags").value || "manual,fundamentals,fundamental_research_note"),
      }),
    });
    showError(result.errors?.length ? result.errors.join("\n") : "");
    if (!result.errors?.length) {
      $("ragText").value = "";
    }
    await loadRagStatus();
  } finally {
    button.disabled = false;
  }
}

function renderSubscriptions() {
  const list = $("subscriptionList");
  if (!list) return;
  list.innerHTML = "";
  if (!state.subscriptions.length) {
    list.innerHTML = `
      <div class="empty-state">
        <strong>暂无自选股</strong>
        <span>先添加一只股票或 ETF，再到股票分析页启动机会扫描。</span>
      </div>
    `;
    return;
  }

  for (const item of state.subscriptions) {
    const selected = state.selectedSymbols.has(item.symbol);
    const row = document.createElement("article");
    row.className = "subscription-item";
    row.innerHTML = `
      <div class="subscription-head">
        <label class="check symbol-check">
          <input type="checkbox" data-symbol="${escapeHtml(item.symbol)}" ${selected ? "checked" : ""} />
          <span>
            <strong>${escapeHtml(item.symbol)}</strong>
            <small>${escapeHtml(marketLabel(item.market_type))}</small>
          </span>
        </label>
        <span class="status-chip ${toneClass(item.status)}">${statusLabel(item.status)}</span>
      </div>
      <label>
        状态
        <span class="intent-field">
          <select data-field="status" data-symbol="${escapeHtml(item.symbol)}">
            ${statusOption("observing", "观察中", item.status)}
            ${statusOption("near_opportunity", "接近机会", item.status)}
            ${statusOption("actionable", "可执行", item.status)}
            ${statusOption("risk_elevated", "风险升高", item.status)}
            ${statusOption("not_compatible", "不兼容", item.status)}
          </select>
          ${intentLight(item)}
        </span>
      </label>
      <input data-field="reason" data-symbol="${escapeHtml(item.symbol)}" value="${escapeHtml(item.reason || "")}" placeholder="关注理由" />
      <div class="subscription-actions">
        <input data-field="target_action" data-symbol="${escapeHtml(item.symbol)}" value="${escapeHtml(item.target_action || "")}" placeholder="目标动作" />
        <button class="ghost-button danger-text" data-delete="${escapeHtml(item.symbol)}" type="button">删除</button>
      </div>
    `;
    list.appendChild(row);
  }
}

function latestDecisionAction(symbol) {
  const row = latestDecisionRow(symbol);
  return row?.action || null;
}

function latestDecisionRow(symbol) {
  const runStatus = state.lastRunPayload?.status || {};
  if (!["succeeded", "degraded"].includes(runStatus.status)) return null;
  return (runStatus.decision_summary?.symbols || []).find(
    (item) => item.symbol === symbol,
  );
}

function intentLight(item) {
  const action = latestDecisionAction(item.symbol);
  const systemBuy = action === "buy";
  const row = latestDecisionRow(item.symbol);
  const positiveUser = ["actionable", "near_opportunity"].includes(item.status);
  const negativeUser = ["risk_elevated", "not_compatible"].includes(item.status);
  const userText = [item.reason, item.target_action]
    .filter((value) => String(value || "").trim())
    .map((value) => String(value).trim().slice(0, 80))
    .join(" · ");
  if (action === null) {
    return `<span class="intent-light intent-muted" title="${escapeHtml("尚无最近一次成功运行的结论，运行后会在此对比你和系统的判断。" + (userText ? ` 你的理由/目标：${userText}` : ""))}" role="img">未评估</span>`;
  }
  if (item.status === "observing") {
    return `<span class="intent-light intent-muted" title="${escapeHtml("你尚未表态；系统仅按自身分析输出结论。" + (userText ? ` 你的理由/目标：${userText}` : ""))}" role="img">未表态</span>`;
  }
  const conflict = (positiveUser && !systemBuy) || (negativeUser && systemBuy);
  const baseTitle = conflict
    ? "你的判断与系统最新运行结论冲突，请查看运行解释中的反对原因。"
    : "你的判断与系统最新运行结论一致。";
  const systemText = row?.execution_message || row?.suggested_action || "";
  const title = [
    baseTitle,
    userText ? `你的理由/目标：${userText}` : "",
    systemText ? `系统：${String(systemText).slice(0, 120)}` : "",
  ]
    .filter(Boolean)
    .join(" ");
  if (conflict) {
    return `<span class="intent-light intent-conflict" title="${escapeHtml(title)}" role="img">与系统冲突</span>`;
  }
  return `<span class="intent-light intent-ok" title="${escapeHtml(title)}" role="img">与系统一致</span>`;
}

function renderSelectedCount() {
  const counter = $("selectedCount");
  if (counter) {
    counter.textContent = `已选 ${state.selectedSymbols.size}`;
  }
}

function statusOption(value, label, selectedValue) {
  return `<option value="${value}" ${value === selectedValue ? "selected" : ""}>${label}</option>`;
}

async function updateSubscription(symbol) {
  const reason = document.querySelector(`[data-field="reason"][data-symbol="${symbol}"]`).value;
  const targetAction = document.querySelector(`[data-field="target_action"][data-symbol="${symbol}"]`).value;
  const status = document.querySelector(`[data-field="status"][data-symbol="${symbol}"]`).value;
  await api(`/api/subscriptions/${encodeURIComponent(symbol)}`, {
    method: "PATCH",
    body: JSON.stringify({ status, reason, target_action: targetAction }),
  });
}

function selectedAnalysts() {
  return Array.from(document.querySelectorAll('input[name="analyst"]:checked')).map(
    (input) => input.value,
  );
}

async function loadRuntimeConfig() {
  const payload = await api("/api/runtime-config");
  $("longTermMemoryEnabled").checked = payload.long_term_memory_enabled !== false;
}

async function startRun() {
  showError("");
  const startButton = $("startRun");
  startButton.disabled = true;
  startButton.textContent = "启动中";
  try {
    const symbols = Array.from(state.selectedSymbols);
    const payload = await api("/api/runs", {
      method: "POST",
      body: JSON.stringify({
        symbols,
        manual_symbols: $("manualSymbols").value,
        selected_analysts: selectedAnalysts(),
        trade_date: $("tradeDate").value || null,
        look_back_days: Number($("lookBackDays").value || 90),
        portfolio_mode: $("portfolioMode").value,
        long_term_memory_enabled: $("longTermMemoryEnabled").checked,
      }),
    });
    state.epoch += 1;
    state.sessionId = payload._session_id || state.sessionId;
    state.currentRunId = payload.run_id;
    localStorage.setItem(lastRunStorageKey, payload.run_id);
    state.activeReportKey = null;
    state.activeTraceSpanId = null;
    state.traceRunId = null;
    state.currentTrace = null;
    state.activeNodeName = null;
    state.expandedActivityDetails.clear();
    $("refreshRun").disabled = false;
    $("reportContent").textContent = "运行已启动，报告生成后可在下方切换查看。";
    renderRun(payload);
    startPolling();
    loadRuns().catch(showError);
  } finally {
    startButton.disabled = state.lastRunPayload?.status?.status === "running";
    startButton.textContent = "启动分析";
  }
}

async function loadRuns() {
  const payload = await api("/api/runs");
  state.recentRuns = payload.items || [];
  renderRunHistory();
  return state.recentRuns;
}

function resetRunView() {
  stopPolling();
  state.epoch += 1;
  state.currentRunId = null;
  state.lastRunPayload = null;
  state.activeReportKey = null;
  state.cancelPending = false;
  state.pollFailures = 0;
  $("runState").textContent = "空闲";
  $("runState").className = "run-state status-neutral";
  $("reportContent").textContent = "选择标的后开始分析。";
  $("reportTabs").replaceChildren();
  $("errors").replaceChildren();
  $("connectionStatus").textContent = "";
  $("startRun").disabled = false;
  $("refreshRun").hidden = true;
  updateRunControls();
  renderSummaryCards();
  renderProgress(null, {});
  renderDecisionRows(null, {});
}

async function refreshRuntime() {
  const runtime = await api("/api/runtime");
  if (state.sessionId && state.sessionId !== runtime.session_id) {
    resetRunView();
    localStorage.removeItem(lastRunStorageKey);
  }
  state.sessionId = runtime.session_id;
  state.runtime = runtime;
  $("runState").textContent = runtime.status === "running" ? "运行中" : "空闲";
  $("runState").className = `run-state ${runtime.status === "running" ? "status-warning" : "status-neutral"}`;
  return runtime;
}

async function restoreLastRun() {
  const stored = localStorage.getItem(lastRunStorageKey);
  const runtime = await refreshRuntime();
  await loadRuns();
  await loadCheckpoints();
  const active = runtime.active_runs || [];
  const target = active.find(run => run.run_id === stored) || [...active].sort((a, b) => a.started_at.localeCompare(b.started_at)).at(-1);
  if (target) await connectRun(target.run_id);
}

async function loadCheckpoints() {
  const payload = await api("/api/checkpoints");
  $("checkpoints").replaceChildren();
  for (const item of payload.items || []) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "ghost-button";
    button.textContent = `手动恢复 ${(item.symbols || []).join("、")}`;
    button.addEventListener("click", async () => {
      button.disabled = true;
      try {
        const run = await api(`/api/checkpoints/${item.checkpoint_id}/resume`, { method: "POST" });
        await connectRun(run.run_id);
        await loadCheckpoints();
      } catch (error) { showError(error); button.disabled = false; }
    });
    $("checkpoints").appendChild(button);
  }
}

async function connectRun(runId) {
  const epoch = ++state.epoch;
  showError("");
  stopPolling();
  state.currentRunId = runId;
  state.activeReportKey = null;
  state.activeTraceSpanId = null;
  state.traceRunId = null;
  state.currentTrace = null;
  state.expandedActivityDetails.clear();
  localStorage.setItem(lastRunStorageKey, runId);
  renderRunHistory();
  $("refreshRun").disabled = false;
  const runtime = await refreshRuntime();
  if (epoch !== state.epoch) return;
  const payload = await api(`/api/runs/${encodeURIComponent(runId)}`);
  if (epoch !== state.epoch || (state.sessionId && payload._session_id !== state.sessionId)) return;
  renderRun(payload);
  if (payload.status?.status === "running" || runtime.active_runs?.length) {
    startPolling();
  }
}

function startPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
  }
  state.pollTimer = setInterval(refreshRun, 2000);
  startProgressTicker();
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
  if (state.progressTimer) {
    clearInterval(state.progressTimer);
    state.progressTimer = null;
  }
}

function startProgressTicker() {
  if (state.progressTimer) return;
  state.progressTimer = setInterval(() => {
    if (!state.lastRunPayload?.status || state.lastRunPayload.status.status !== "running") {
      clearInterval(state.progressTimer);
      state.progressTimer = null;
      return;
    }
    renderSummaryCards(state.lastRunPayload.status);
    renderProgress(state.lastRunPayload.run_id, state.lastRunPayload.status);
  }, 1000);
}

async function refreshRun() {
  if (!state.currentRunId || state.pollBusy) return;
  state.pollBusy = true;
  const runId = state.currentRunId, epoch = state.epoch;
  try {
    await refreshRuntime();
    if (epoch !== state.epoch) return;
    const selectedStatus = state.lastRunPayload?.status?.status;
    if (isTerminalRunStatus(selectedStatus) && !(state.runtime.active_runs || []).length) {
      stopPolling();
      return;
    }
    if (isTerminalRunStatus(selectedStatus)) {
      return;
    }
    const payload = await api(`/api/runs/${encodeURIComponent(runId)}`);
    if (epoch !== state.epoch || runId !== state.currentRunId || payload._session_id !== state.sessionId) return;
    state.pollFailures = 0;
    $("connectionStatus").textContent = "";
    $("refreshRun").hidden = true;
    renderRun(payload);
    if (isTerminalRunStatus(payload.status?.status)) {
      await loadRuns();
      await loadCheckpoints();
      if (!(state.runtime.active_runs || []).length) stopPolling();
    }
  } catch (error) {
    if (epoch !== state.epoch) return;
    if (++state.pollFailures >= 3) {
      $("connectionStatus").textContent = "连接中断，保留最后已知状态；正在尝试重新连接。";
      $("refreshRun").hidden = false;
    }
  } finally {
    state.pollBusy = false;
  }
}

function renderRun(payload) {
  state.lastRunPayload = payload;
  const runStatus = payload.status || {};
  const runState = $("runState");
  const status = runStatus.status || "idle";
  // The header describes live service work, not the selected historical report.
  runState.textContent = state.runtime.status === "running" || status === "running" ? "运行中" : "空闲";
  runState.className = `run-state ${runState.textContent === "运行中" ? "status-warning" : "status-neutral"}`;
  updateRunControls(runStatus);

  renderSummaryCards(runStatus);
  renderProgress(payload.run_id, runStatus);

  renderDecisionRows(payload.run_id, runStatus);

  const errors = payload.errors || runStatus.errors || [];
  $("errors").textContent = errors.length ? "本次分析出现问题，请查看失败阶段。" : "";
  if (status === "failed") {
    const copy = document.createElement("button");
    copy.type = "button";
    copy.textContent = "复制故障编号";
    copy.addEventListener("click", () => navigator.clipboard.writeText(`Request ID: ${runStatus.origin_request_id || "未知"}\nRun ID: ${payload.run_id}\nTrace ID: ${runStatus.trace_id || "未知"}`).catch(showError));
    $("errors").appendChild(copy);
  }
  $("startRun").disabled = status === "running";
  renderReportTabs(payload.run_id, payload.reports || {});
  if (["succeeded", "degraded"].includes(status) && $("subscriptionList")) {
    renderSubscriptions();
  }
}

function updateRunControls(runStatus = {}) {
  const cancelButton = $("cancelRun");
  if (!cancelButton) return;
  const running = runStatus.status === "running";
  const cancelRequested = running && runStatus.cancel_requested === true;
  cancelButton.hidden = !running;
  cancelButton.disabled = !state.currentRunId || !running || cancelRequested || state.cancelPending;
  cancelButton.textContent = state.cancelPending || cancelRequested ? "取消中" : "取消并保存";
}

async function cancelRun() {
  if (!state.currentRunId || state.cancelPending) return;
  showError("");
  state.cancelPending = true;
  updateRunControls(state.lastRunPayload?.status || {});
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(state.currentRunId)}/cancel`, {
      method: "POST",
    });
    renderRun(payload);
    startPolling();
    await loadRuns();
  } finally {
    state.cancelPending = false;
    updateRunControls(state.lastRunPayload?.status || {});
  }
}

function renderRunHistory() {
  const container = $("runHistory");
  if (!container) return;
  const runs = state.recentRuns || [];
  if (!runs.length) {
    container.innerHTML = "";
    return;
  }
  container.innerHTML = `
    <div class="section-heading compact">
      <div>
        <h3>最近运行</h3>
        <span>点击查看历史报告；可恢复任务需要手动启动。</span>
      </div>
      <button class="ghost-button" type="button" data-refresh-runs>刷新列表</button>
    </div>
    <div class="run-history-list">
      ${runs.slice(0, 8).map(renderRunHistoryItem).join("")}
    </div>
  `;
}

function renderRunHistoryItem(run = {}) {
  const status = run.status || "pending";
  const active = run.run_id === state.currentRunId ? " active" : "";
  const cancelRequested = run.cancel_requested && status === "running";
  const label = cancelRequested ? "取消中" : runStatusLabels[status] || status;
  return `
    <button class="run-history-item${active}" type="button" data-connect-run="${escapeHtml(run.run_id)}">
      <span class="status-chip ${toneClass(status)}">${escapeHtml(label)}</span>
      <strong>${escapeHtml(run.run_id || "-")}</strong>
      <small>${escapeHtml((run.symbols || []).join(", ") || "-")}</small>
      <em>${escapeHtml(formatDateTime(run.updated_at))}</em>
    </button>
  `;
}

function renderSummaryCards(runStatus = {}) {
  const summary = runStatus.decision_summary || {};
  const metrics = summary.metrics || {};
  const nodes = runStatus.nodes || {};
  const primaryNodes = primaryWorkflowNodes.map((name) => nodes[name] || { status: "pending" });
  const completedNodes = primaryNodes.filter((node) => ["succeeded", "skipped"].includes(node.status)).length;
  const failedNodes = primaryNodes.filter((node) => node.status === "failed").length;
  const reportCount =
    metrics.report_count ??
    Object.values(runStatus.reports || {}).filter((report) => report.exists).length;
  const riskAdjusted = metrics.risk_adjusted_count ?? 0;
  const riskHeld = metrics.risk_held_count ?? 0;

  const cards = [
    { label: "标的", value: metrics.symbol_count ?? (runStatus.symbols || []).length, tone: "neutral" },
    { label: "累计用时", value: runElapsed(runStatus), tone: failedNodes ? "danger" : "neutral" },
    { label: "风控调整/持平", value: `${riskAdjusted}/${riskHeld}`, tone: riskAdjusted || riskHeld || failedNodes ? "warning" : "neutral" },
    { label: "报告", value: reportCount, tone: reportCount ? "success" : "neutral" },
  ];

  $("summaryCards").innerHTML = cards.map(summaryCard).join("");
}

const visibleStages = [
  ["准备数据", ["Load Persona Markdown", "Load Subscription Symbols"]],
  ["分析标的", ["Technical Position", "News Sentiment", "Fundamental Analysis"]],
  ["形成建议", ["Trader"]],
  ["风险检查", ["Risk Check", "Portfolio Manager"]],
  ["生成报告", ["Explain Run", "Persist Trace"]],
];
const parallelAnalystNames = ["Technical Position", "News Sentiment", "Fundamental Analysis"];

function renderProgress(runId, runStatus = {}) {
  const nodes = runStatus.nodes || {};
  const stages = visibleStages.map(([label, names]) => {
    const selected = names.map(name => nodes[name]).filter(Boolean);
    let status = "pending";
    if (selected.some(node => node.status === "failed")) status = "failed";
    else if (selected.some(node => node.status === "cancelled" || node.error === "cancelled")) status = "cancelled";
    else if (selected.some(node => node.status === "running")) status = "running";
    else if (selected.length && selected.every(node => ["succeeded", "skipped"].includes(node.status))) status = "succeeded";
    return { label, status };
  });
  const current = nodes[runStatus.current_node] || {};
  const activity = current.activity || {};
  const stage = stages.find(item => item.status === "running" || item.status === "failed");
  const completed = stages.filter(item => item.status === "succeeded").length;
  const elapsed = activityDurationLabel(activity);
  let message = runStatus.status === "running" ? `正在${stage?.label || "准备数据"}，已用时 ${runElapsed(runStatus)}` : "选择标的后开始分析";
  const analysts = parallelAnalystNames.filter(name => nodes[name] && nodes[name].status !== "skipped");
  const activeAnalysts = analysts.filter(name => nodes[name].status === "running");
  if (activeAnalysts.length > 1) message = `分析师并行执行，已完成 ${analysts.filter(name => nodes[name].status === "succeeded").length}/${analysts.length}，已用时 ${runElapsed(runStatus)}`;
  if (runStatus.status === "running" && activity.tool_name && activity.status === "running") message += `；等待数据服务响应 ${elapsed}`;
  const activityStart = parseDate(activity.started_at || current.started_at);
  if (runStatus.status === "running" && activityStart && Date.now() - activityStart.getTime() > 60000) message += "；当前步骤耗时较长，尚未确认失败";
  if (runStatus.cancel_requested && runStatus.status === "running") message = "正在取消；等待当前调用结束后保存完整节点。";
  if (runStatus.status === "running" && Object.values(nodes).some(node => node.status === "failed")) message = "正在停止；分析师执行失败，等待其他调用退出。";
  if (runStatus.status === "succeeded") message = (runStatus.errors || []).length || (runStatus.graph_errors || []).length ? "分析完成，但部分数据不完整，请检查报告中的限制。" : "分析完成，结果摘要已更新。";
  if (runStatus.status === "failed") message = `${stage?.label || "分析"}失败；请复制故障编号排查。`;
  if (runStatus.status === "cancelled") message = runStatus.resumable ? "已取消并保存，可在历史记录中手动恢复。" : "已取消，没有可恢复断点。";
  if (runStatus.status === "degraded") message = "运行以降级状态结束（组合不可用或 LLM 未配置）：只保留分析结果，未产生新动作。";
  const tone = runStatus.status === "failed" ? "status-danger" : runStatus.status === "running" || runStatus.status === "degraded" || (runStatus.graph_errors || []).length ? "status-warning" : runStatus.status === "succeeded" ? "status-success" : "status-neutral";
  $("progress").innerHTML = `<ol class="stage-list">${stages.map(item => `<li class="stage-${item.status}"><span>${item.status === "succeeded" ? "✓" : item.status === "failed" ? "!" : item.status === "running" ? "●" : "○"}</span>${item.label}</li>`).join("")}</ol><progress max="5" value="${completed}" aria-label="已完成阶段"></progress><p class="run-message ${tone}" role="status">${escapeHtml(message)}</p>`;
  $("progress").insertAdjacentHTML("beforeend", `<div class="node-grid">${analysts.map(name => nodeCard(name, nodes[name])).join("")}</div>${renderNodeActivityDetails(runStatus)}`);
}

function nodeCard(name, node = {}, optional = false) {
  const status = node.status || "pending";
  const duration = nodeDurationLabel(node);
  const active = state.activeNodeName === name ? " active" : "";
  return `
    <button class="node ${toneClass(status)}${optional ? " optional-node" : ""}${active}" type="button" data-node-name="${escapeHtml(name)}">
      <span>${escapeHtml(nodeStatusLabels[status] || status)}</span>
      <strong>${escapeHtml(name)}</strong>
      <em>${escapeHtml(duration)}</em>
      ${node.activity?.tool_name ? `<small>${escapeHtml(node.activity.tool_name)}</small>` : ""}
      ${node.error ? `<small>${escapeHtml(node.error)}</small>` : ""}
    </button>
  `;
}

function renderNodeActivityDetails(runStatus = {}) {
  const nodeName = state.activeNodeName;
  if (!nodeName) return "";
  if (!parallelAnalystNames.includes(nodeName)) {
    return `
      <section class="node-activity-panel">
        <div class="section-heading compact">
          <div>
            <h3>${escapeHtml(nodeName)}</h3>
            <span>当前版本先支持 Technical Position 的运行中详情。</span>
          </div>
        </div>
      </section>
    `;
  }

  const node = (runStatus.nodes || {})[nodeName] || {};
  const activity = node.activity || null;
  const history = Array.isArray(node.activity_history) ? [...node.activity_history].reverse() : [];
  const current = activity || history[0] || null;
  const status = node.status || "pending";
  return `
    <section class="node-activity-panel">
      <div class="section-heading compact">
        <div>
          <h3>${escapeHtml(nodeName)} 实时详情</h3>
          <span>显示 LLM 阶段、工具调用和脱敏摘要；完整 Trace 在运行结束后查看。</span>
        </div>
        <span class="status-chip ${toneClass(status)}">${escapeHtml(nodeStatusLabels[status] || status)}</span>
      </div>
      ${
        current
          ? renderActivityCurrent(current)
          : `<div class="empty-state compact"><strong>暂无活动</strong><span>节点开始调用 LLM 或工具后会显示实时详情。</span></div>`
      }
      ${
        history.length
          ? `<div class="activity-history">
              <strong>最近活动</strong>
              ${history.map(renderActivityHistoryItem).join("")}
            </div>`
          : ""
      }
    </section>
  `;
}

function renderActivityCurrent(activity = {}) {
  const args = activity.args === undefined ? null : JSON.stringify(activity.args, null, 2);
  return `
    <div class="activity-current">
      <div>
        <span class="status-chip ${toneClass(activity.status || "running")}">${escapeHtml(activity.status || "running")}</span>
        <strong>${escapeHtml(activity.title || activity.phase || "运行中")}</strong>
        <em>${escapeHtml(activityDurationLabel(activity))}</em>
      </div>
      <dl>
        <div><dt>阶段</dt><dd>${escapeHtml(activity.phase || "-")}</dd></div>
        <div><dt>工具</dt><dd>${escapeHtml(activity.tool_name || "-")}</dd></div>
        <div><dt>工具阶段</dt><dd>${escapeHtml(activity.tool_phase || "-")}</dd></div>
        <div><dt>LLM 轮次</dt><dd>${escapeHtml(activity.llm_round ?? "-")}</dd></div>
        <div><dt>缓存</dt><dd>${activity.cache_hit === true ? "命中" : activity.cache_hit === false ? "未命中" : "-"}</dd></div>
      </dl>
      ${args ? `<pre class="activity-json">${escapeHtml(args)}</pre>` : ""}
      ${renderActivitySummary(activity.summary, activity.details, activity)}
    </div>
  `;
}
function renderActivityHistoryItem(activity = {}) {
  const details = renderActivityDetails(activity);
  if (details) {
    const detailKey = activityDetailKey(activity, "history");
    return `
      <details class="activity-item activity-item-expandable" data-activity-detail-key="${escapeHtml(detailKey)}"${activityHistoryOpenAttr(activity, detailKey)}>
        <summary>
          <span class="status-chip ${toneClass(activity.status || "neutral")}">${escapeHtml(activity.status || "-")}</span>
          <strong>${escapeHtml(activity.title || activity.phase || "-")}</strong>
          <em>${escapeHtml(activityDurationLabel(activity))}</em>
        </summary>
        ${details}
      </details>
    `;
  }
  return `
    <button class="activity-item" type="button" disabled>
      <span class="status-chip ${toneClass(activity.status || "neutral")}">${escapeHtml(activity.status || "-")}</span>
      <strong>${escapeHtml(activity.title || activity.phase || "-")}</strong>
      <em>${escapeHtml(activityDurationLabel(activity))}</em>
    </button>
  `;
}

function renderActivitySummary(summary = {}, details = {}, activity = {}) {
  const entries = Object.entries(summary || {});
  const detailEntries = Object.entries(details || {});
  if (!entries.length && !detailEntries.length) return "";
  return `
    <div class="activity-summary">
      ${entries
        .map(([label, payload]) => {
          const parts = Object.entries(payload || {})
            .map(([key, value]) => `${key}: ${value}`)
            .join(" 路 ");
          const detail = details?.[label];
          if (detail) {
            const detailKey = activityDetailKey(activity, label);
            return `
              <details class="activity-detail" data-activity-detail-key="${escapeHtml(detailKey)}"${activityDetailOpenAttr(detailKey)}>
                <summary><strong>${escapeHtml(label)}</strong>${escapeHtml(parts)}</summary>
                <pre>${escapeHtml(JSON.stringify(detail, null, 2))}</pre>
              </details>
            `;
          }
          return `<span><strong>${escapeHtml(label)}</strong>${escapeHtml(parts)}</span>`;
        })
        .join("")}
      ${detailEntries
        .filter(([label]) => !entries.some(([summaryLabel]) => summaryLabel === label))
        .map(
          ([label, payload]) => {
            const detailKey = activityDetailKey(activity, label);
            return `
            <details class="activity-detail" data-activity-detail-key="${escapeHtml(detailKey)}"${activityDetailOpenAttr(detailKey)}>
              <summary><strong>${escapeHtml(label)}</strong></summary>
              <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
            </details>
          `;
          },
        )
        .join("")}
    </div>
  `;
}

function renderActivityDetails(activity = {}) {
  const detailEntries = Object.entries(activity.details || {});
  if (!detailEntries.length) return "";
  return detailEntries
    .map(
      ([label, payload]) => `
        <section class="activity-detail-block">
          <strong>${escapeHtml(label)}</strong>
          <pre>${escapeHtml(JSON.stringify(payload, null, 2))}</pre>
        </section>
      `,
    )
    .join("");
}

function activityDetailKey(activity = {}, label = "") {
  const identity =
    activity.span_id ||
    [
      activity.phase || "",
      activity.title || "",
      activity.llm_round ?? "",
      activity.tool_name || "",
      activity.tool_phase || "",
      activity.started_at || "",
    ].join("|");
  return `${identity}:${label}`;
}

function activityDetailOpenAttr(detailKey) {
  return state.expandedActivityDetails.has(detailKey) ? " open" : "";
}

function activityHistoryOpenAttr(activity = {}, detailKey = "") {
  if (state.expandedActivityDetails.has(detailKey)) return " open";
  return Object.keys(activity.details || {}).some((label) => state.expandedActivityDetails.has(activityDetailKey(activity, label)))
    ? " open"
    : "";
}

function renderNodeGuide(runStatus = {}) {
  const guide = $("nodeGuide");
  if (!guide) return;
  guide.innerHTML = `
    <div>
      <strong>主流程 11 个节点</strong>
      <span>${primaryWorkflowNodes.map((name) => escapeHtml(name)).join(" → ")}</span>
    </div>
  `;
}

function renderDecisionRows(runId, runStatus = {}) {
  const decisions = ["succeeded", "degraded"].includes(runStatus.status) ? runStatus.decision_summary?.symbols || [] : [];
  $("resultSummary").innerHTML = decisions.length ? decisions.map(row => `<article class="result-card"><strong>${escapeHtml(row.symbol)} · ${escapeHtml(directionLabels[row.direction] || "待复核")}</strong><p>${escapeHtml(row.suggested_action || row.execution_message || "请查看完整报告")}</p><small>风控：${row.approved_by_risk === true ? "通过" : row.approved_by_risk === false ? "保持不动" : "待复核"}</small></article>`).join("") : "完成分析后将在此展示结论。";
  const rows = runStatus.decision_summary?.symbols || [];
  if (!rows.length) {
    $("decisionRows").innerHTML =
      '<tr><td colspan="11" class="empty-cell">启动一次分析后，决策摘要会显示在这里。</td></tr>';
    return;
  }

  $("decisionRows").innerHTML = rows
    .map((row) => {
      const status = row.status || "observing";
      const executionStatus = row.execution_status || "";
      const riskLabel =
        row.approved_by_risk === true
          ? row.risk_clamped
            ? "已调整"
            : "通过"
          : row.approved_by_risk === false
            ? "保持不动"
            : "待复核";
      const riskTone =
        row.approved_by_risk === true
          ? row.risk_clamped
            ? "status-warning"
            : "status-success"
          : row.approved_by_risk === false
            ? "status-warning"
            : "status-neutral";
      return `
        <tr>
          <td><strong>${escapeHtml(row.symbol || "-")}</strong></td>
          <td><span class="status-chip ${toneClass(status)}">${escapeHtml(row.status_label || statusLabel(status))}</span></td>
          <td>${formatNumber(row.current_price)}</td>
          <td>${formatPercent(row.target_weight)}</td>
          <td>${formatPercent(row.final_weight)}</td>
          <td>${formatPercent(row.delta_weight)}</td>
          <td>${escapeHtml(directionLabels[row.direction] || row.direction || "-")}</td>
          <td><span class="status-chip ${riskTone}">${riskLabel}</span></td>
          <td><span class="status-chip ${toneClass(executionStatus)}">${escapeHtml(executionStatusLabels[executionStatus] || executionStatus || "-")}</span></td>
          <td class="broker-cell">${renderBrokerControl(runId, runStatus.status, row)}</td>
          <td class="action-cell" title="${escapeHtml(row.execution_message || row.suggested_action || "")}">
            ${escapeHtml(row.execution_message || row.suggested_action || "-")}
          </td>
        </tr>
      `;
    })
    .join("");
}
function renderBrokerControl(runId, runStatus, row) {
  const title = escapeHtml(row.broker_message || row.broker_status || "");
  if (row.submitted_to_broker) {
    const label = row.broker_order_id
      ? `Submitted #${row.broker_order_id}`
      : `Submitted ${row.submitted_quantity || ""}`.trim();
    return `<span class="status-chip status-success" title="${title}">${escapeHtml(label)}</span>`;
  }
  if (row.broker_attempt_count && !row.broker_retry_available) {
    return `<span class="status-chip status-danger" title="${title}">${escapeHtml(row.broker_status || "Failed")}</span>`;
  }
  if (row.pending_broker_order && row.broker_confirmation_required) {
    if (runStatus !== "succeeded") {
      return '<span class="status-chip status-warning">Pending</span>';
    }
    return `
      <button
        class="confirm-order-button"
        type="button"
        data-confirm-simulated="${escapeHtml(row.symbol || "")}"
        data-run-id="${escapeHtml(runId || "")}"
      >
        ${row.broker_attempt_count ? "Retry SIM" : "Confirm SIM"}
      </button>
    `;
  }
  return '<span class="empty-inline">-</span>';
}

async function renderTrace(runId, runStatus = {}) {
  const timeline = $("traceTimeline");
  const details = $("traceDetails");
  if (!timeline || !details) return;
  const traceReport = runStatus.reports?.trace_json;
  if (!runId || !traceReport?.exists) {
    timeline.innerHTML = '<div class="empty-state compact"><strong>暂无 Trace</strong><span>Trace 文件会在 Persist Trace 阶段完成后出现。</span></div>';
    details.textContent = "运行完成后选择一个 trace span 查看详情。";
    state.traceRunId = null;
    state.currentTrace = null;
    renderObservabilityStatus(state.observability || {}, null);
    return;
  }
  if (state.traceRunId === runId && state.currentTrace) {
    renderTracePayload(state.currentTrace);
    return;
  }
  timeline.innerHTML = '<div class="empty-state compact"><strong>正在读取 Trace</strong><span>正在加载本地 trace.json。</span></div>';
  details.textContent = "";
  try {
    const payload = await api(`/api/runs/${encodeURIComponent(runId)}/trace`);
    state.traceRunId = runId;
    state.currentTrace = payload.trace || {};
    state.activeTraceSpanId = null;
    renderTracePayload(state.currentTrace);
  } catch (error) {
    timeline.innerHTML = `<div class="empty-state compact"><strong>Trace 不可用</strong><span>${escapeHtml(error.message || error)}</span></div>`;
    details.textContent = "";
    renderObservabilityStatus(state.observability || {}, null);
  }
}

function renderTracePayload(trace = {}) {
  const timeline = $("traceTimeline");
  const details = $("traceDetails");
  const spans = trace.spans || [];
  if (!spans.length) {
    timeline.innerHTML = '<div class="empty-state compact"><strong>暂无 spans</strong><span>trace.json 中没有 span 记录。</span></div>';
    details.textContent = "";
    renderObservabilityStatus(state.observability || {}, trace);
    return;
  }
  renderObservabilityStatus(state.observability || {}, trace);
  const activeId = state.activeTraceSpanId || spans[0].span_id;
  state.activeTraceSpanId = activeId;
  timeline.innerHTML = spans
    .map((span) => {
      const depth = traceDepth(span, spans);
      const active = span.span_id === activeId ? " active" : "";
      const label = span.attributes?.["graph.node.name"] || span.attributes?.["tool.name"] || span.kind || "";
      return `
        <button class="trace-span${active}" type="button" data-trace-span="${escapeHtml(span.span_id)}" style="--depth:${depth}">
          <span class="trace-status ${toneClass(span.status)}">${escapeHtml(span.status || "-")}</span>
          <strong>${escapeHtml(span.name || "-")}</strong>
          <small>${escapeHtml(label)}</small>
          <em>${formatDuration(span.duration_ms)}</em>
        </button>
      `;
    })
    .join("");
  const activeSpan = spans.find((span) => span.span_id === activeId) || spans[0];
  details.textContent = JSON.stringify(activeSpan, null, 2);
}
function traceDepth(span, spans) {
  const byId = new Map(spans.map((item) => [item.span_id, item]));
  let depth = 0;
  let parent = span.parent_span_id ? byId.get(span.parent_span_id) : null;
  while (parent && depth < 8) {
    depth += 1;
    parent = parent.parent_span_id ? byId.get(parent.parent_span_id) : null;
  }
  return depth;
}

async function confirmSimulatedOrder(symbol, button) {
  if (!state.currentRunId || !symbol) return;
  showError("");
  button.disabled = true;
  button.textContent = "Submitting";
  try {
    const payload = await api(
      `/api/runs/${encodeURIComponent(state.currentRunId)}/orders/${encodeURIComponent(symbol)}/confirm-simulated`,
      { method: "POST" },
    );
    renderRun(payload);
    if (state.activeView === "portfolio") {
      await loadPortfolio();
    }
  } catch (error) {
    button.disabled = false;
    button.textContent = "Confirm SIM";
    showError(error);
  }
}

function renderReportTabs(runId, reports) {
  const tabs = $("reportTabs");
  tabs.innerHTML = "";
  const keys = Object.keys(reports)
    .filter((key) => !["trace_json", "trace_markdown"].includes(key))
    .sort((a, b) => {
      const left = preferredReportOrder.indexOf(a);
      const right = preferredReportOrder.indexOf(b);
      return (left === -1 ? 99 : left) - (right === -1 ? 99 : right) || a.localeCompare(b);
    });
  if (!keys.length) {
    tabs.innerHTML = '<span class="empty-inline">暂无报告。</span>';
    return;
  }

  if (state.activeReportKey && !keys.includes(state.activeReportKey)) {
    state.activeReportKey = null;
  }

  for (const key of keys) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = reportLabel(key);
    button.className = key === state.activeReportKey ? "active" : "";
    button.addEventListener("click", async () => {
      document.querySelectorAll("#reportTabs button").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      state.activeReportKey = key;
      const epoch = state.epoch;
      const text = await api(
        `/api/reports/${encodeURIComponent(runId)}/${encodeURIComponent(key)}`,
      );
      if (epoch === state.epoch && state.currentRunId === runId && state.activeReportKey === key) $("reportContent").textContent = text;
    });
    tabs.appendChild(button);
  }
}

async function loadPortfolio() {
  const button = $("refreshPortfolio");
  if (button) {
    button.disabled = true;
    button.textContent = "刷新中";
  }
  try {
    const payload = await api("/api/portfolio/simulated");
    state.portfolio = payload;
    renderPortfolio(payload);
  } finally {
    if (button) {
      button.disabled = false;
      button.textContent = "刷新账户";
    }
  }
}

function renderPortfolio(payload = {}) {
  const status = $("portfolioStatus");
  const cards = $("portfolioCards");
  const positions = $("portfolioPositions");
  if (!status || !cards || !positions) return;

  const available = payload.available === true;
  status.className = `portfolio-status ${available ? "status-success" : "status-warning"}`;
  status.innerHTML = available
    ? `<strong>Futu 模拟账户已连接</strong><span>更新时间 ${formatDateTime(payload.updated_at)}</span>`
    : `<strong>Futu 模拟账户不可用</strong><span>${escapeHtml(payload.error || "无法读取账户快照")}</span>`;

  cards.innerHTML = [
    summaryCard({ label: "总资产", value: formatCurrency(payload.total_value), tone: available ? "success" : "neutral" }),
    summaryCard({ label: "可用现金", value: formatCurrency(payload.cash), tone: "neutral" }),
    summaryCard({ label: "现金占比", value: formatPercent(payload.cash_weight), tone: "neutral" }),
    summaryCard({ label: "持仓数量", value: Object.keys(payload.position_weights || {}).length, tone: "neutral" }),
  ].join("");

  const entries = Object.entries(payload.position_weights || {}).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
  if (!entries.length) {
    positions.innerHTML = `
      <div class="empty-state compact">
        <strong>暂无持仓权重</strong>
        <span>${available ? "账户当前没有可展示持仓。" : "连接 Futu OpenD 后会显示模拟账户持仓。"}</span>
      </div>
    `;
    return;
  }

  positions.innerHTML = entries
    .map(([symbol, weight]) => {
      const normalized = Math.min(Math.abs(Number(weight) || 0), 1);
      return `
        <article class="position-row">
          <div>
            <strong>${escapeHtml(symbol)}</strong>
            <span>${formatPercent(weight)}</span>
          </div>
          <div class="weight-track">
            <span style="width: ${normalized * 100}%"></span>
          </div>
        </article>
      `;
    })
    .join("");
}
function summaryCard(card) {
  return `
    <article class="summary-card summary-${card.tone}">
      <span>${escapeHtml(card.label)}</span>
      <strong>${escapeHtml(card.value)}</strong>
    </article>
  `;
}

function statusLabel(value) {
  return subscriptionStatusLabels[value] || nodeStatusLabels[value] || runStatusLabels[value] || value || "-";
}

function toneClass(value) {
  if (["actionable", "succeeded", "portfolio_decided", "ok"].includes(value)) {
    return "status-success";
  }
  if (["near_opportunity", "running", "alert_only", "confirmation_required", "degraded"].includes(value)) {
    return "status-warning";
  }
  if (["risk_elevated", "failed", "not_compatible", "error"].includes(value)) {
    return "status-danger";
  }
  return "status-neutral";
}

function isTerminalRunStatus(status) {
  return ["succeeded", "failed", "cancelled", "degraded"].includes(status);
}

function marketLabel(value) {
  return {
    US_STOCK: "美股",
    US_ETF: "美股 ETF",
  }[value] || value || "-";
}

function reportLabel(value) {
  return {
    run_explanation: "运行解释",
    risk_check: "风控检查",
    portfolio_manager: "组合经理",
    simulated_broker_orders: "模拟订单",
    trader: "交易计划",
    news_sentiment: "新闻情绪",
    fundamental_analysis: "基本面分析",
    technical_position: "技术位置",
    futu_portfolio: "Futu 组合",
    run_audit: "运行审计",
  }[value] || value;
}
function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return number.toFixed(Math.abs(number) >= 100 ? 1 : 2);
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return `${(number * 100).toFixed(1)}%`;
}

function formatCurrency(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  }).format(number);
}

function formatDateTime(value) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString("zh-CN", { hour12: false });
}

function formatDuration(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  if (number >= 1000) return `${(number / 1000).toFixed(1)}s`;
  return `${number.toFixed(number >= 10 ? 1 : 2)}ms`;
}

function runElapsed(runStatus = {}) {
  const started = parseDate(runStatus.started_at);
  if (!started) return "-";
  const finished = parseDate(runStatus.finished_at);
  const end = finished || new Date();
  return formatDuration(end.getTime() - started.getTime());
}

function nodeDurationLabel(node = {}) {
  const status = node.status || "pending";
  const started = parseDate(node.started_at);
  const finished = parseDate(node.finished_at);
  if (status === "pending") return "等待";
  if (status === "skipped") return node.error ? `跳过：${node.error}` : "跳过";
  if (!started) return "-";
  const end = finished || new Date();
  const label = formatDuration(end.getTime() - started.getTime());
  return status === "running" ? `运行 ${label}` : `耗时 ${label}`;
}

function activityDurationLabel(activity = {}) {
  if (activity.duration_ms !== null && activity.duration_ms !== undefined) {
    return formatDuration(activity.duration_ms);
  }
  const started = parseDate(activity.started_at);
  if (!started) return "-";
  const ended = parseDate(activity.ended_at);
  const end = ended || new Date();
  return formatDuration(end.getTime() - started.getTime());
}

function parseDate(value) {
  if (!value) return null;
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? null : date;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll('"', "&quot;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}

function splitCsv(value) {
  return String(value || "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);
}

function bindEvents() {
  document.querySelectorAll(".nav-button").forEach((button) => {
    if (button.dataset.view) button.addEventListener("click", () => switchView(button.dataset.view));
  });

  $("subscriptionForm").addEventListener("submit", async (event) => {
    event.preventDefault();
    showError("");
    try {
      await api("/api/subscriptions", {
        method: "POST",
        body: JSON.stringify({
          symbol: $("subSymbol").value,
          market_type: $("subMarket").value,
          reason: $("subReason").value,
          target_action: $("subAction").value,
        }),
      });
      event.target.reset();
      await loadSubscriptions();
    } catch (error) {
      showError(error);
    }
  });

  $("subscriptionList").addEventListener("change", async (event) => {
    const symbol = event.target.dataset.symbol;
    if (!symbol) return;
    if (event.target.type === "checkbox") {
      event.target.checked ? state.selectedSymbols.add(symbol) : state.selectedSymbols.delete(symbol);
      renderSelectedCount();
      return;
    }
    try {
      await updateSubscription(symbol);
      await loadSubscriptions();
    } catch (error) {
      showError(error);
    }
  });

  $("subscriptionList").addEventListener("click", async (event) => {
    const symbol = event.target.dataset.delete;
    if (!symbol) return;
    showError("");
    try {
      await api(`/api/subscriptions/${encodeURIComponent(symbol)}`, { method: "DELETE" });
      state.selectedSymbols.delete(symbol);
      await loadSubscriptions();
    } catch (error) {
      showError(error);
    }
  });

  $("refreshSubscriptions").addEventListener("click", () => loadSubscriptions().catch(showError));
  $("startRun").addEventListener("click", () => startRun().catch(showError));
  $("cancelRun").addEventListener("click", () => cancelRun().catch(showError));
  $("refreshRun").addEventListener("click", () => refreshRun().catch(showError));
  $("runHistory").addEventListener("click", (event) => {
    const refreshButton = event.target.closest("[data-refresh-runs]");
    if (refreshButton) {
      loadRuns().catch(showError);
      return;
    }
    const button = event.target.closest("[data-connect-run]");
    if (!button) return;
    connectRun(button.dataset.connectRun).catch(showError);
  });
  $("progress").addEventListener("click", (event) => {
    const button = event.target.closest("[data-node-name]");
    if (!button) return;
    state.activeNodeName = button.dataset.nodeName;
    renderProgress(state.currentRunId, state.lastRunPayload?.status || {});
  });
  $("progress").addEventListener(
    "toggle",
    (event) => {
      const detail = event.target.closest?.("details[data-activity-detail-key]");
      if (!detail) return;
      const detailKey = detail.dataset.activityDetailKey;
      if (!detailKey) return;
      if (detail.open) {
        state.expandedActivityDetails.add(detailKey);
      } else {
        state.expandedActivityDetails.delete(detailKey);
      }
    },
    true,
  );
  $("decisionRows").addEventListener("click", (event) => {
    const symbol = event.target.dataset.confirmSimulated;
    if (!symbol) return;
    confirmSimulatedOrder(symbol, event.target).catch(showError);
  });
  $("traceTimeline")?.addEventListener("click", (event) => {
    const button = event.target.closest("[data-trace-span]");
    if (!button) return;
    state.activeTraceSpanId = button.dataset.traceSpan;
    renderTracePayload(state.currentTrace || {});
  });
  $("refreshRag").addEventListener("click", () => loadRagStatus(true).catch(showError));
  $("ingestDefaults").addEventListener("click", () => ingestDefaults().catch(showError));
  $("ingestOnline").addEventListener("click", () => ingestOnlineResearch().catch(showError));
  $("ingestText").addEventListener("click", () => ingestTextKnowledge().catch(showError));
  $("refreshPortfolio").addEventListener("click", () => loadPortfolio().catch(showError));
  $("refreshRiskPosition").addEventListener("click", () => loadRiskPosition().catch(showRiskError));
  $("saveRiskPosition").addEventListener("click", () => saveRiskPosition().catch(showRiskError));
  $("resetRiskPosition").addEventListener("click", () => resetRiskPosition().catch(showRiskError));
  $("refreshRiskPortfolio").addEventListener("click", () => loadPortfolio().then(renderRiskPosition).catch(showRiskError));
  $("riskPositionForm").addEventListener("input", (event) => {
    syncRiskInputs(event.target);
    state.riskDirty = true;
    updateRiskFeedback();
  });
  $("riskPositionForm").addEventListener("change", (event) => {
    syncRiskInputs(event.target);
    state.riskDirty = true;
    updateRiskFeedback();
  });
  $("riskPositionForm").addEventListener("submit", (event) => event.preventDefault());
  $("riskTargetAdd").addEventListener("click", () => {
    const symbol = $("riskTargetAddSelect").value;
    if (!symbol) return;
    state.riskPosition.target_weights[symbol] = 0;
    state.riskDirty = true;
    renderRiskTargetRows();
    updateRiskFeedback();
  });
  $("riskTargetRows").addEventListener("click", (event) => {
    const removeSymbol = event.target.dataset.removeRiskTarget;
    const addSymbol = event.target.dataset.addRiskTarget;
    if (!removeSymbol && !addSymbol) return;
    if (removeSymbol) delete state.riskPosition.target_weights[removeSymbol];
    if (addSymbol) state.riskPosition.target_weights[addSymbol] = 0;
    state.riskDirty = true;
    renderRiskTargetRows();
    updateRiskFeedback();
  });
  $("riskTargetRows").addEventListener("input", (event) => {
    const numberInput = event.target.closest("[data-risk-target-number]");
    const rangeInput = event.target.closest("[data-risk-target-range]");
    const symbol = numberInput?.dataset.riskTargetNumber || rangeInput?.dataset.riskTargetRange;
    if (!symbol) return;
    const value = numberInput ? Number(numberInput.value) : Number(rangeInput.value);
    if (!Number.isFinite(value)) return;
    const normalized = Math.max(-100, Math.min(100, value));
    state.riskPosition.target_weights[symbol] = normalized / 100;
    state.riskDirty = true;
    const number = document.querySelector(`[data-risk-target-number="${CSS.escape(symbol)}"]`);
    const range = document.querySelector(`[data-risk-target-range="${CSS.escape(symbol)}"]`);
    if (number) number.value = normalized;
    if (range) range.value = normalized;
    updateRiskFeedback();
  });
  $("refreshMemories").addEventListener("click", () => loadMemories().catch(showMemoryError));
  $("memoryStatusFilter").addEventListener("change", () => loadMemories().catch(showMemoryError));
  $("memoryRows").addEventListener("click", (event) => {
    const button = event.target.closest("[data-memory-action]");
    if (!button) return;
    handleMemoryAction(button.dataset.memoryAction, button.dataset.memoryId).catch(showMemoryError);
  });
  $("memoryOutcomeForm").addEventListener("submit", (event) => {
    recordMemoryOutcome(event).catch(showMemoryError);
  });
}

function init() {
  bindEvents();
  switchView(location.hash === "#memory" ? "memory" : "analysis");
  renderSummaryCards();
  renderProgress(null, {});
  resetRunView();
  renderRagStatus({
    backend: "fundamental_chroma",
    available: null,
    document_count: "-",
    error: "",
  });
  renderPortfolio({
    available: false,
    mode: "simulation",
    total_value: null,
    cash: null,
    cash_weight: null,
    position_weights: {},
    error: "点击刷新账户读取 Futu 模拟账户。",
  });
  loadSubscriptions().catch(showError);

  loadRuntimeConfig().catch(showError);
  restoreLastRun().catch(showError);
}
init();

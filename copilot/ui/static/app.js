const state = {
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

const optionalWorkflowNodes = ["Opportunity Radar"];

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
  pending: "待执行",
  running: "运行中",
  succeeded: "成功",
  failed: "失败",
  skipped: "跳过",
  cancelled: "已取消",
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
  "opportunity_radar",
  "technical_position",
  "futu_portfolio",
  "run_audit",
  "trace_markdown",
  "trace_json",
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
      message = payload.detail || message;
    } catch {
      message = await response.text();
    }
    throw new Error(message);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("application/json") ? response.json() : response.text();
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
}

function showError(error) {
  const errors = $("errors");
  if (errors) {
    errors.textContent = error ? String(error.message || error) : "";
  }
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

async function loadRagStatus() {
  const payload = await api("/api/rag/status?probe=true");
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
      ? "正在检查"
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
        <select data-field="status" data-symbol="${escapeHtml(item.symbol)}">
          ${statusOption("observing", "观察中", item.status)}
          ${statusOption("near_opportunity", "接近机会", item.status)}
          ${statusOption("actionable", "可执行", item.status)}
          ${statusOption("risk_elevated", "风险升高", item.status)}
          ${statusOption("not_compatible", "不兼容", item.status)}
        </select>
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
function renderSelectedCount() {
  const counter = $("selectedCount");
  if (counter) {
    counter.textContent = `宸查€?${state.selectedSymbols.size}`;
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
      }),
    });
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
    startButton.disabled = false;
    startButton.textContent = "启动分析";
  }
}

async function loadRuns() {
  const payload = await api("/api/runs");
  state.recentRuns = payload.items || [];
  renderRunHistory();
  return state.recentRuns;
}

async function restoreLastRun() {
  const runs = await loadRuns();
  const storedRunId = localStorage.getItem(lastRunStorageKey);
  const storedRun = runs.find((run) => run.run_id === storedRunId);
  const runningRun = runs.find((run) => run.status === "running");
  const target = storedRun || runningRun;
  if (!target) return;
  await connectRun(target.run_id);
}

async function connectRun(runId) {
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
  const payload = await api(`/api/runs/${encodeURIComponent(runId)}`);
  renderRun(payload);
  if (payload.status?.status === "running") {
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
  if (!state.currentRunId) return;
  const payload = await api(`/api/runs/${encodeURIComponent(state.currentRunId)}`);
  renderRun(payload);
  const status = payload.status?.status;
  if (isTerminalRunStatus(status)) {
    stopPolling();
    loadRuns().catch(showError);
  }
}

function renderRun(payload) {
  state.lastRunPayload = payload;
  const runStatus = payload.status || {};
  const runState = $("runState");
  const status = runStatus.status || "running";
  runState.textContent = runStatus.cancel_requested && status === "running" ? "正在取消" : runStatusLabels[status] || status;
  runState.className = `run-state ${toneClass(status)}`;
  updateRunControls(runStatus);

  renderSummaryCards(runStatus);
  renderProgress(payload.run_id, runStatus);
  renderNodeGuide(runStatus);
  renderDecisionRows(payload.run_id, runStatus);
  renderTrace(payload.run_id, runStatus);
  renderObservabilityStatus(state.observability || {}, state.currentTrace);

  const errors = payload.errors || runStatus.errors || [];
  $("errors").textContent = errors.map((item) => item.message || item).join("\n");
  renderReportTabs(payload.run_id, payload.reports || {});
}

function updateRunControls(runStatus = {}) {
  const cancelButton = $("cancelRun");
  if (!cancelButton) return;
  const running = runStatus.status === "running";
  const cancelRequested = runStatus.cancel_requested === true;
  cancelButton.disabled = !state.currentRunId || !running || cancelRequested || state.cancelPending;
  cancelButton.textContent = state.cancelPending || cancelRequested ? "取消中" : "取消运行";
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
        <span>刷新页面后可重新连接正在运行或已完成的 run。</span>
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
    { label: "主流程节点", value: `${completedNodes}/${primaryWorkflowNodes.length}`, tone: failedNodes ? "danger" : "success" },
    { label: "风控调整/持平", value: `${riskAdjusted}/${riskHeld}`, tone: riskAdjusted || riskHeld || failedNodes ? "warning" : "neutral" },
    { label: "报告", value: reportCount, tone: reportCount ? "success" : "neutral" },
  ];

  $("summaryCards").innerHTML = cards.map(summaryCard).join("");
}

function renderProgress(runId, runStatus = {}) {
  const nodes = runStatus.nodes || {};
  const primaryEntries = primaryWorkflowNodes.map((name) => [name, nodes[name] || { status: "pending" }]);
  const optionalEntries = Object.entries(nodes).filter(([name]) => !primaryWorkflowNodes.includes(name));
  const currentNode = runStatus.current_node || "-";
  const elapsed = runElapsed(runStatus);
  const primaryStats = primaryEntries.reduce(
    (acc, [, node]) => {
      acc[node.status || "pending"] = (acc[node.status || "pending"] || 0) + 1;
      return acc;
    },
    {},
  );

  $("progress").innerHTML = `
    <div class="progress-heading">
      <div>
        <h3>运行流程</h3>
        <span>${escapeHtml(runId || "等待启动")}</span>
      </div>
      <strong>${escapeHtml((runStatus.symbols || []).join(", ") || "暂无标的")}</strong>
    </div>
    <div class="run-overview">
      <span>当前步骤：${escapeHtml(currentNode)}</span>
      <span>总耗时：${escapeHtml(elapsed)}</span>
      <span>完成 ${primaryStats.succeeded || 0} / 跳过 ${primaryStats.skipped || 0} / 失败 ${primaryStats.failed || 0}</span>
    </div>
    <div class="node-grid">${primaryEntries.map(([name, node]) => nodeCard(name, node)).join("")}</div>
    ${
      optionalEntries.length
        ? `<div class="optional-node-block">
            <div class="mini-heading">可选分析节点</div>
            <div class="node-grid optional">${optionalEntries.map(([name, node]) => nodeCard(name, node, true)).join("")}</div>
          </div>`
        : ""
    }
    ${renderNodeActivityDetails(runStatus)}
  `;
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
    </button>
  `;
}

function renderNodeActivityDetails(runStatus = {}) {
  const nodeName = state.activeNodeName;
  if (!nodeName) return "";
  if (nodeName !== "Technical Position") {
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
          <h3>Technical Position 实时详情</h3>
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
  const optionalPresent = Object.keys(runStatus.nodes || {}).some((name) => optionalWorkflowNodes.includes(name));
  guide.innerHTML = `
    <div>
      <strong>主流程 11 个节点</strong>
      <span>${primaryWorkflowNodes.map((name) => escapeHtml(name)).join(" → ")}</span>
    </div>
    <div>
      <strong>可选节点</strong>
      <span>${optionalPresent ? "Opportunity Radar 会按选择运行或跳过，不计入主流程总数。" : "Opportunity Radar 未进入当前状态。"}</span>
    </div>
  `;
}

function renderDecisionRows(runId, runStatus = {}) {
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
  if (row.broker_idempotency_key) {
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
        Confirm SIM
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
    renderObservabilityStatus(state.observability || {}, state.currentTrace);
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
  const keys = Object.keys(reports).sort((a, b) => {
    const left = preferredReportOrder.indexOf(a);
    const right = preferredReportOrder.indexOf(b);
    return (left === -1 ? 99 : left) - (right === -1 ? 99 : right) || a.localeCompare(b);
  });
  if (!keys.length) {
    tabs.innerHTML = '<span class="empty-inline">鏆傛棤鎶ュ憡銆?/span>';
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
      $("reportContent").textContent = await api(
        `/api/reports/${encodeURIComponent(runId)}/${encodeURIComponent(key)}`,
      );
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
  if (["near_opportunity", "running", "alert_only", "confirmation_required"].includes(value)) {
    return "status-warning";
  }
  if (["risk_elevated", "failed", "not_compatible", "error"].includes(value)) {
    return "status-danger";
  }
  return "status-neutral";
}

function isTerminalRunStatus(status) {
  return ["succeeded", "failed", "cancelled"].includes(status);
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
    opportunity_radar: "机会雷达",
    technical_position: "技术位置",
    futu_portfolio: "Futu 组合",
    run_audit: "运行审计",
    trace_markdown: "Trace",
    trace_json: "Trace JSON",
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
  if (status === "pending") return "绛夊緟";
  if (status === "skipped") return node.error ? `璺宠繃锛?{node.error}` : "璺宠繃";
  if (!started) return "-";
  const end = finished || new Date();
  const label = formatDuration(end.getTime() - started.getTime());
  return status === "running" ? `杩愯 ${label}` : `鑰楁椂 ${label}`;
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
    button.addEventListener("click", () => switchView(button.dataset.view));
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
  $("traceTimeline").addEventListener("click", (event) => {
    const button = event.target.closest("[data-trace-span]");
    if (!button) return;
    state.activeTraceSpanId = button.dataset.traceSpan;
    renderTracePayload(state.currentTrace || {});
  });
  $("refreshRag").addEventListener("click", () => loadRagStatus().catch(showError));
  $("ingestDefaults").addEventListener("click", () => ingestDefaults().catch(showError));
  $("ingestOnline").addEventListener("click", () => ingestOnlineResearch().catch(showError));
  $("ingestText").addEventListener("click", () => ingestTextKnowledge().catch(showError));
  $("refreshPortfolio").addEventListener("click", () => loadPortfolio().catch(showError));
}

function init() {
  bindEvents();
  switchView("analysis");
  renderSummaryCards();
  renderProgress(null, {});
  renderNodeGuide({});
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
  loadRagStatus().catch(showError);
  loadObservabilityStatus().catch(showError);
  restoreLastRun().catch(showError);
}
init();

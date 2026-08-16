const state = {
  activeView: "analysis",
  subscriptions: [],
  selectedSymbols: new Set(),
  currentRunId: null,
  pollTimer: null,
  activeReportKey: null,
  ragStatus: null,
  portfolio: null,
};

const $ = (id) => document.getElementById(id);

const subscriptionStatusLabels = {
  observing: "观察中",
  near_opportunity: "接近机会",
  actionable: "可行动",
  risk_elevated: "风险升高",
  not_compatible: "不兼容",
};

const nodeStatusLabels = {
  pending: "待执行",
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
};

const directionLabels = {
  buy: "买入",
  hold: "持有",
  reduce: "减仓",
  sell: "卖出",
  short: "Short",
  cover: "Cover",
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
];

const workflowSteps = [
  ["technical_position", "技术位置"],
  ["news_sentiment", "新闻情绪"],
  ["fundamental_analysis", "基本面分析"],
  ["opportunity_radar", "机会雷达"],
  ["trader", "交易计划"],
  ["risk_check", "风控检查"],
  ["portfolio_manager", "Portfolio Manager"],
  ["run_explanation", "运行解释"],
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
  const payload = await api("/api/rag/status");
  state.ragStatus = payload;
  renderRagStatus(payload);
}

function renderRagStatus(payload = {}) {
  const status = $("ragStatus");
  if (!status) return;
  const available = payload.available === true;
  const count = payload.document_count ?? 0;
  status.textContent = available
    ? `Chroma 可用 · ${count} docs`
    : `Fallback 检索 · ${payload.error || "Chroma/OpenAI 不可用"}`;
  status.className = available ? "status-success inline-status" : "status-warning inline-status";
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
          ${statusOption("actionable", "可行动", item.status)}
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
    state.activeReportKey = null;
    $("refreshRun").disabled = false;
    $("reportContent").textContent = "运行已启动，报告生成后可在下方切换查看。";
    renderRun(payload);
    startPolling();
  } finally {
    startButton.disabled = false;
    startButton.textContent = "启动分析";
  }
}

function startPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
  }
  state.pollTimer = setInterval(refreshRun, 2000);
}

async function refreshRun() {
  if (!state.currentRunId) return;
  const payload = await api(`/api/runs/${encodeURIComponent(state.currentRunId)}`);
  renderRun(payload);
  const status = payload.status?.status;
  if (status === "succeeded" || status === "failed") {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function renderRun(payload) {
  const runStatus = payload.status || {};
  const runState = $("runState");
  const status = runStatus.status || "running";
  runState.textContent = runStatusLabels[status] || status;
  runState.className = `run-state ${toneClass(status)}`;

  renderSummaryCards(runStatus);
  renderProgress(payload.run_id, runStatus);
  renderDecisionRows(payload.run_id, runStatus);

  const errors = payload.errors || runStatus.errors || [];
  $("errors").textContent = errors.map((item) => item.message || item).join("\n");
  renderReportTabs(payload.run_id, payload.reports || {});
}

function renderSummaryCards(runStatus = {}) {
  const summary = runStatus.decision_summary || {};
  const metrics = summary.metrics || {};
  const nodes = runStatus.nodes || {};
  const nodeValues = Object.values(nodes);
  const completedNodes =
    metrics.completed_nodes ??
    nodeValues.filter((node) => ["succeeded", "skipped"].includes(node.status)).length;
  const failedNodes =
    metrics.failed_nodes ?? nodeValues.filter((node) => node.status === "failed").length;
  const reportCount =
    metrics.report_count ??
    Object.values(runStatus.reports || {}).filter((report) => report.exists).length;
  const riskAdjusted = metrics.risk_adjusted_count ?? 0;
  const riskHeld = metrics.risk_held_count ?? 0;

  const cards = [
    { label: "标的", value: metrics.symbol_count ?? (runStatus.symbols || []).length, tone: "neutral" },
    { label: "节点完成", value: `${completedNodes}/${nodeValues.length || 0}`, tone: failedNodes ? "danger" : "success" },
    { label: "风控调整/持平", value: `${riskAdjusted}/${riskHeld}`, tone: riskAdjusted || riskHeld || failedNodes ? "warning" : "neutral" },
    { label: "报告", value: reportCount, tone: reportCount ? "success" : "neutral" },
  ];

  $("summaryCards").innerHTML = cards.map(summaryCard).join("");
}

function renderProgress(runId, runStatus = {}) {
  const nodes = runStatus.nodes || {};
  const entries = Object.keys(nodes).length
    ? Object.entries(nodes)
    : workflowSteps.map(([name, label]) => [label, { status: "pending", rawName: name }]);
  const nodeHtml = entries
    .map(([name, node]) => {
      const status = node.status || "pending";
      return `
        <div class="node ${toneClass(status)}">
          <span>${escapeHtml(nodeStatusLabels[status] || status)}</span>
          <strong>${escapeHtml(name)}</strong>
        </div>
      `;
    })
    .join("");

  $("progress").innerHTML = `
    <div class="progress-heading">
      <div>
        <h3>运行流程</h3>
        <span>${escapeHtml(runId || "等待启动")}</span>
      </div>
      <strong>${escapeHtml((runStatus.symbols || []).join(", ") || "暂无标的")}</strong>
    </div>
    <div class="node-grid">${nodeHtml}</div>
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
  if (["actionable", "succeeded", "portfolio_decided"].includes(value)) {
    return "status-success";
  }
  if (["near_opportunity", "running", "alert_only", "confirmation_required", "pending", "skipped"].includes(value)) {
    return "status-warning";
  }
  if (["risk_elevated", "failed", "not_compatible"].includes(value)) {
    return "status-danger";
  }
  return "status-neutral";
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
  $("refreshRun").addEventListener("click", () => refreshRun().catch(showError));
  $("decisionRows").addEventListener("click", (event) => {
    const symbol = event.target.dataset.confirmSimulated;
    if (!symbol) return;
    confirmSimulatedOrder(symbol, event.target).catch(showError);
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
}

init();

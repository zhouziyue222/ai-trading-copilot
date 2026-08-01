const state = {
  subscriptions: [],
  selectedSymbols: new Set(),
  currentRunId: null,
  pollTimer: null,
  activeReportKey: null,
  ragStatus: null,
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
  watch: "观察",
};

const executionStatusLabels = {
  blocked_by_risk: "风险阻断",
  alert_only: "仅提醒",
  simulation_ready: "模拟就绪",
  confirmation_required: "需要确认",
  live_ready: "Live 就绪",
};

const preferredReportOrder = [
  "run_explanation",
  "risk_check",
  "execution_alert",
  "trader",
  "opportunity_review",
  "opportunity_radar",
  "technical_position",
  "fundamental_news",
  "futu_portfolio",
  "run_audit",
];

const workflowSteps = [
  ["opportunity_radar", "机会雷达"],
  ["technical_position", "技术位置"],
  ["fundamental_news", "基本面/新闻"],
  ["opportunity_review", "机会复核"],
  ["trader", "交易计划"],
  ["risk_check", "风控检查"],
  ["execution_alert", "执行提醒"],
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

function showError(error) {
  $("errors").textContent = error ? String(error.message || error) : "";
}

async function loadSubscriptions() {
  showError("");
  const payload = await api("/api/subscriptions");
  state.subscriptions = payload.items || [];
  renderSubscriptions();
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
    ? `Chroma 可用 · ${count} 条`
    : `降级检索 · ${payload.error || "Chroma/OpenAI 不可用"}`;
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
  list.innerHTML = "";
  if (!state.subscriptions.length) {
    list.innerHTML = `
      <div class="empty-state">
        <strong>暂无订阅</strong>
        <span>先添加一个股票或 ETF，再启动机会扫描。</span>
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
            <small>${escapeHtml(item.market_type)}</small>
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
    startButton.textContent = "启动";
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
  renderDecisionRows(runStatus);

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
  const riskBlocked = metrics.risk_blocked_count ?? 0;

  const cards = [
    { label: "标的", value: metrics.symbol_count ?? (runStatus.symbols || []).length, tone: "neutral" },
    { label: "节点完成", value: `${completedNodes}/${nodeValues.length || 0}`, tone: "success" },
    { label: "风险/失败", value: `${riskBlocked}/${failedNodes}`, tone: riskBlocked || failedNodes ? "danger" : "neutral" },
    { label: "报告", value: reportCount, tone: "warning" },
  ];

  $("summaryCards").innerHTML = cards
    .map(
      (card) => `
        <article class="summary-card summary-${card.tone}">
          <span>${card.label}</span>
          <strong>${card.value}</strong>
        </article>
      `,
    )
    .join("");
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
    <div class="run-meta">
      <div>
        <span>Run ID</span>
        <strong>${escapeHtml(runId || "-")}</strong>
      </div>
      <div>
        <span>Symbols</span>
        <strong>${escapeHtml((runStatus.symbols || []).join(", ") || "-")}</strong>
      </div>
      <div>
        <span>Current</span>
        <strong>${escapeHtml(runStatus.current_node || "-")}</strong>
      </div>
    </div>
    <div class="node-grid">${nodeHtml}</div>
  `;
}

function renderDecisionRows(runStatus = {}) {
  const rows = runStatus.decision_summary?.symbols || [];
  if (!rows.length) {
    $("decisionRows").innerHTML =
      '<tr><td colspan="9" class="empty-cell">启动一次运行后，决策摘要会显示在这里。</td></tr>';
    return;
  }

  $("decisionRows").innerHTML = rows
    .map((row) => {
      const status = row.status || "observing";
      const executionStatus = row.execution_status || "";
      const riskLabel =
        row.approved_by_risk === true ? "通过" : row.approved_by_risk === false ? "阻断" : "待复核";
      const riskTone =
        row.approved_by_risk === true ? "status-success" : row.approved_by_risk === false ? "status-danger" : "status-neutral";
      return `
        <tr>
          <td><strong>${escapeHtml(row.symbol || "-")}</strong></td>
          <td><span class="status-chip ${toneClass(status)}">${escapeHtml(row.status_label || statusLabel(status))}</span></td>
          <td>${formatNumber(row.current_price)}</td>
          <td>${formatNumber(row.support_level)}</td>
          <td>${formatNumber(row.reward_risk_ratio)}</td>
          <td>${escapeHtml(directionLabels[row.direction] || row.direction || "-")}</td>
          <td><span class="status-chip ${riskTone}">${riskLabel}</span></td>
          <td><span class="status-chip ${toneClass(executionStatus)}">${escapeHtml(executionStatusLabels[executionStatus] || executionStatus || "-")}</span></td>
          <td class="action-cell" title="${escapeHtml(row.suggested_action || row.execution_message || "")}">
            ${escapeHtml(row.suggested_action || row.execution_message || "-")}
          </td>
        </tr>
      `;
    })
    .join("");
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
    button.textContent = key;
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

function statusLabel(value) {
  return subscriptionStatusLabels[value] || nodeStatusLabels[value] || runStatusLabels[value] || value || "-";
}

function toneClass(value) {
  if (["actionable", "succeeded", "simulation_ready", "live_ready"].includes(value)) {
    return "status-success";
  }
  if (["near_opportunity", "running", "alert_only", "confirmation_required", "pending", "skipped"].includes(value)) {
    return "status-warning";
  }
  if (["risk_elevated", "failed", "blocked_by_risk", "not_compatible"].includes(value)) {
    return "status-danger";
  }
  return "status-neutral";
}

function formatNumber(value) {
  if (value === null || value === undefined || value === "") return "-";
  const number = Number(value);
  if (Number.isNaN(number)) return String(value);
  return number.toFixed(Math.abs(number) >= 100 ? 1 : 2);
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
$("refreshRag").addEventListener("click", () => loadRagStatus().catch(showError));
$("ingestDefaults").addEventListener("click", () => ingestDefaults().catch(showError));
$("ingestOnline").addEventListener("click", () => ingestOnlineResearch().catch(showError));
$("ingestText").addEventListener("click", () => ingestTextKnowledge().catch(showError));

renderSummaryCards();
renderProgress(null, {});
loadSubscriptions().catch(showError);
loadRagStatus().catch(showError);

"use strict";
const byId = id => document.getElementById(id);
const statuses = {pending: "等待观察", waiting_data: "等待数据", failed: "需要重试", completed: "复盘完成"};
const categories = {hypothetical: "假设观察", SIMULATE: "模拟成交", REAL: "实盘成交"};
const percent = value => value == null ? "证据不足" : `${(value * 100).toFixed(2)}%`;
function paragraph(parent, label, value) {
  const p = document.createElement("p");
  const strong = document.createElement("strong"); strong.textContent = `${label}：`;
  p.append(strong, document.createTextNode(String(value ?? "暂无"))); parent.append(p);
}
function showReview(detail) {
  const root = byId("review-summary"); root.replaceChildren();
  const {task, snapshot, result} = detail;
  paragraph(root, "计划", `${task.symbol} · ${snapshot.plan?.direction || "—"} · ${snapshot.generated_at}`);
  paragraph(root, "原始入场依据", snapshot.plan?.entry_logic);
  if (!result) { paragraph(root, "进度", task.reason || "尚未到观察时点"); }
  else {
    paragraph(root, "观察截止", result.cutoff_at);
    for (const item of result.observations || []) {
      const h = document.createElement("h3"); h.textContent = `${categories[item.category]}${item.account_id ? ` · 账户 ${item.account_id}` : ""}`; root.append(h);
      paragraph(root, item.category === "hypothetical" ? "标的观察收益" : "计划资金收益（含持仓浮盈）", percent(item.realized_return));
      paragraph(root, "基准收益 / 日收盘回撤", `${percent(item.benchmark_return)} / ${percent(item.max_drawdown)}`);
      if (item.category !== "hypothetical") {
        paragraph(root, "已实现 / 未实现盈亏", `${item.realized_pnl_gross ?? "证据不足"} / ${item.unrealized_pnl_gross ?? "证据不足"}`);
        paragraph(root, "费用", item.fees_complete ? `已知费用 ${item.known_fees}` : "费用不可用，盈亏未扣费用");
      }
      if (!item.eligible) paragraph(root, "验证限制", item.qualification_reason || "当前观察不参与批准评估");
    }
    if (result.reflection) {
      for (const [key, label] of [["judgment", "计划判断"], ["execution_deviation", "执行偏差"], ["counter_evidence", "反证"], ["applicability", "适用条件"]]) paragraph(root, label, result.reflection[key]);
      paragraph(root, "证据引用", (result.reflection.evidence_refs || []).join("、"));
    } else paragraph(root, "模型反思", result.reflection_skipped || task.reason || "等待反思");
    for (const memory of result.memories || []) paragraph(root, "学习结果", memory.memory_id ? `${memory.memory_id} · ${memory.status === "shadow" ? "影子验证中，达标后人工批准" : memory.status}` : "未通过经验安全检查");
    for (const evaluation of result.evaluations || []) paragraph(root, "经验门槛评估", `${evaluation.memory_id} · ${evaluation.eligible ? "已达标，等待人工批准" : "尚未达标"} · ${evaluation.evidence_runs} 个独立运行`);
  }
  byId("review-detail").textContent = JSON.stringify(detail, null, 2);
}
async function api(path, body) {
  const options = body === undefined ? {} : {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body)};
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail));
  return data;
}
async function action(work) {
  byId("feedback").textContent = "处理中…";
  try { await work(); byId("feedback").textContent = "已完成"; }
  catch (error) { byId("feedback").textContent = error.message; }
}
function button(label, work) {
  const node = document.createElement("button");
  node.type = "button";
  node.textContent = label;
  node.addEventListener("click", () => action(work));
  return node;
}
async function refresh() {
  const [reviews, fills] = await Promise.all([api("/api/reviews"), api("/api/review-fills")]);
  byId("reviews").replaceChildren();
  for (const item of reviews.items) {
    const task = item.task || item;
    const row = document.createElement("tr");
    for (const value of [`${task.run_id} / ${task.symbol}`, task.horizon_days, statuses[task.status] || task.status, task.status === "completed" ? "—" : task.next_check_at || "待计算", task.reason || "—"]) {
      const cell = document.createElement("td"); cell.textContent = String(value); row.append(cell);
    }
    const actions = document.createElement("td");
    actions.append(button("详情", async () => {
      showReview(await api(`/api/reviews/${encodeURIComponent(task.id)}`));
    }), button("重试", async () => { await api(`/api/reviews/${encodeURIComponent(task.id)}/retry`, {}); await refresh(); }));
    row.append(actions); byId("reviews").append(row);
  }
  byId("fills").replaceChildren();
  if (!fills.items.length) byId("fills").textContent = "暂无成交记录。";
  for (const fill of fills.items) {
    const block = document.createElement("p");
    block.textContent = `${categories[fill.environment]} · ${fill.account_id} · ${fill.symbol} · ${fill.side} ${fill.quantity} @ ${fill.price} · ${fill.run_id ? `已关联 ${fill.run_id}` : "待关联"} · 成交 ${fill.deal_id}`;
    if (!fill.run_id) block.append(button("填写关联", async () => {
      const form = byId("associate");
      for (const key of ["environment", "account_id", "deal_id", "symbol"]) form.elements.namedItem(key).value = fill[key];
      form.elements.namedItem("run_id").focus();
    }));
    byId("fills").append(block);
  }
}
byId("refresh").addEventListener("click", () => action(refresh));
byId("run").addEventListener("click", () => action(async () => {
  byId("review-detail").textContent = JSON.stringify(await api("/api/reviews/run-due", {}), null, 2); await refresh();
}));
byId("associate").addEventListener("submit", event => {
  event.preventDefault();
  const body = Object.fromEntries(new FormData(event.currentTarget));
  action(async () => { await api("/api/review-fills/associate", body); await refresh(); });
});
action(refresh);

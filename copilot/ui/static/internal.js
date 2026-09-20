const byId = id => document.getElementById(id);
const display = value => { byId("detail").textContent = JSON.stringify(value, null, 2); };

function row(values) {
  const tr = document.createElement("tr");
  for (const value of values) {
    const td = document.createElement("td");
    td.textContent = String(value ?? "—");
    tr.appendChild(td);
  }
  return tr;
}

async function refresh(event) {
  event?.preventDefault();
  const button = byId("search").querySelector("button");
  button.disabled = true;
  try {
    const response = await fetch(`/api/internal/observability?query=${encodeURIComponent(byId("query").value.trim())}`);
    if (!response.ok) throw new Error("诊断入口未启用，或此连接没有访问权限。");
    const data = await response.json();
    byId("feedback").textContent = data.window;
    byId("health").textContent = `会话 ${data.runtime.session_id} · PID ${data.runtime.pid} · ${data.runtime.status === "idle" ? "空闲" : "运行中"} · 最近采集 ${data.health.last_event_at || "暂无"} · 丢弃 ${data.health.dropped} 条 · 导出配置 ${data.export.otel_configured ? "已配置（不代表已送达）" : "本地模式"}`;
    const exportHealth = data.export.export_health || {};
    byId("health").textContent += ` · 最近导出成功 ${exportHealth.last_success_at || "暂无"} · 导出失败批次 ${exportHealth.failed_batches || 0}`;
    byId("workers").replaceChildren();
    for (const run of data.runtime.active_runs) {
      const p = document.createElement("p");
      p.textContent = `${run.run_id} · ${run.status} · 线程存活 ${run.worker_alive ? "是" : "否"} · 观察时间 ${run.heartbeat_at} · 最近业务进展 ${run.last_progress_at}`;
      byId("workers").appendChild(p);
    }
    byId("services").replaceChildren(...data.services.map(s => row([s.name, s.count, s.p50_ms, s.p95_ms, s.errors, s.cache_hits])));
    byId("usage").replaceChildren(...(data.usage || []).map(u => row([u.name, u.count, u.missing_count, u.prompt_tokens, u.completion_tokens, u.total_tokens, u.avg_total_tokens])));
    byId("errors").replaceChildren();
    for (const group of data.errors) {
      const button = document.createElement("button");
      button.className = "ghost-button";
      const e = group.latest;
      button.textContent = `${e["error.type"]} · ${group.count} 次 · ${e["code.file.path"] || "未知位置"}:${e["code.line.number"] || "—"}`;
      button.onclick = () => display(e);
      byId("errors").appendChild(button);
    }
    if (!data.errors.length) byId("errors").textContent = "当前窗口无异常。";
    byId("timeline").replaceChildren();
    const events = [...data.events].reverse();
    const starts = new Map(events.filter(e => e.event === "span.start").map(e => [e.span_id, e]));
    for (const item of events) {
      const button = document.createElement("button");
      const start = starts.get(item.span_id);
      let depth = 0, parent = starts.get(item.parent_span_id);
      while (parent && depth < 8) { depth++; parent = starts.get(parent.parent_span_id); }
      button.className = `trace-span ${item.outcome === "error" || item.level === "ERROR" ? "status-danger" : item.outcome === "ok" ? "status-success" : "status-neutral"}`;
      button.style.setProperty("--depth", depth);
      button.textContent = `${item.timestamp.slice(11, 23)} ${item.node || item.tool || item.name || item.event} · ${item.event} ${item.duration_ms == null ? "" : `${item.duration_ms} ms`}`;
      button.onclick = () => display({ ...start, ...item });
      byId("timeline").appendChild(button);
    }
  } catch (error) { byId("feedback").textContent = error.message; }
  finally { button.disabled = false; }
}
byId("query").value = new URLSearchParams(location.search).get("query") || "";
byId("search").addEventListener("submit", refresh);
refresh();

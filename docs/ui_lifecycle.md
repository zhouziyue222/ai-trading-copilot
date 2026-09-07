# UI 生命周期与 UTF-8

## 启动与编码

在项目目录执行 `./scripts/start-ui.ps1`，默认地址 `http://127.0.0.1:8000`。
首次升级前先停止旧 UI 服务，再使用本项目入口启动，避免旧实例继续写入运行目录。
支持 `-Port 8001`、`-HostName 127.0.0.1`、`-Reload`；热重载也按服务退出丢弃正在运行的进度。
入口固定使用 `.venv/Scripts/python.exe -X utf8`。OTel 启动脚本复用该入口。

`./scripts/enable-utf8.ps1` 配置当前 PowerShell 会话；加 `-Persist` 会设置用户级
`PYTHONUTF8=1`、`PYTHONIOENCODING=utf-8`，并在 PowerShell 5.1 / 7 的用户 profile
追加带标记的配置块。重复执行不重复追加，既有文件先备份，原有字节不转码。
profile 路径及备份路径会打印到终端。恢复配置时使用该备份，并按原值恢复用户环境变量。

Python 源码编码声明只影响源码解析，不能修复已损坏文本。项目 JSON 使用 UTF-8 无 BOM，
JSON 导入允许 UTF-8 BOM。面向 Excel 的 CSV 使用 `encoding="utf-8-sig", newline=""`。
Python 子进程的输出端与父进程解码端分别指定 UTF-8。API 的成功和错误 JSON 均声明
`application/json; charset=utf-8`。普通 fetch 调用直接用 `response.json()`。

如果新功能处理二进制 UTF-8 文本流，应使用持续存在的解码器，避免中文跨块被拆坏：

```javascript
const reader = response.body
  .pipeThrough(new TextDecoderStream("utf-8", { fatal: true }))
  .getReader();
let text = "";
try {
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    text += value;
  }
} finally {
  reader.releaseLock();
}
```

## 状态与数据目录

全局状态来自 `GET /api/runtime` 的本次服务存活任务。历史报告、localStorage、历史
`run_status.json` 均不能启动任务或改变全局运行状态。启动页面不执行 RAG 探测。
运行和恢复只能由对应 POST 请求触发，页面刷新只重新连接存活任务。

| 目录 | 用途 |
| --- | --- |
| `reports/.runtime/<session>/<run>/` | 当前计算的临时状态、报告与消费后的恢复输入 |
| `reports/.checkpoints/checkpoint_*.json` | 用户显式取消后原子提交的完整检查点 |
| `reports/.checkpoints/*.consumed` | 无计算数据的消费凭据，用于对重复恢复返回 409 |
| `reports/run_*/` | 已完成报告、已取消报告或最小失败摘要 |

服务启动先获取报告目录锁，然后清除旧 `.runtime`、未提交 `.tmp`，以及旧版非终态
`run_UI_*`。不清理 CLI 报告、已完成报告、自选股、配置、知识库或长期记忆。
旧版 cancelled 文件没有完整计算状态，不能作为可恢复检查点。
路径越界、符号链接、Windows junction 或清理失败会阻止服务就绪。

## 取消、恢复与退出

每个顺序节点在独立工作副本上执行；只有节点成功返回并合并输出后更新稳定内存快照。
节点内部的临时变更不进入检查点。取消发生在初始化之前时，保存原始输入，恢复时初始化。
断点包含版本、运行参数、图状态、节点顺序、下一节点索引、已完成报告内容和节点状态。
Pydantic 类型在恢复时重新验证，不使用 pickle 或截断的日志数据恢复。

`POST /api/runs/{run_id}/cancel` 只取消当前服务拥有的任务。已结束任务幂等返回；
只有文件而无执行线程的非终态任务返回 409，不改写旧文件。
`GET /api/checkpoints` 列出保存的断点；
`POST /api/checkpoints/{checkpoint_id}/resume` 手动恢复并返回新的 run_id。
缺失返回 404；已消费、不兼容或损坏返回 409。恢复会消费源断点，恢复途中强杀不再回退。

取消为协作式停止，无取消接口的外部调用需要等返回或超时。UI 在此期间显示正在取消。
断点保存成功才标记 resumable；磁盘失败显示保存失败，不宣称可以恢复。

Ctrl+C / SIGTERM 在 Uvicorn 收到退出信号时关闭提交入口，退出最多等工作线程 5 秒。
无法立即结束的 daemon 线程随服务进程退出。Windows 服务及子进程进入 kill-on-close Job；
Job 初始化失败时不继续启动服务。强杀和断电不能执行 Python 清理，临时文件在下次启动
清除，任何运行中的临时数据都不具备恢复资格。外部系统已经发生的副作用不通过删除
本地进度回滚；订单仍须独立人工确认，取消或未完成运行不能确认订单。

## 验证

```powershell
$verifyTemp = Join-Path $PWD ("tmp\verify-" + [guid]::NewGuid().ToString("N"))
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_run_lifecycle.py tests/test_ui_process_lifecycle.py tests/test_run_tracker.py tests/test_ui_app.py tests/test_ui_trace_api.py tests/test_copilot_langgraph.py -q --basetemp $verifyTemp
```

测试使用临时目录、离线两节点图及独立测试服务，覆盖取消/恢复节点执行次数、部分变更
隔离、重复取消、损坏断点、保存失败、锁与目录边界、真实 SIGINT/强杀、重启不自启、
Windows 子进程随服务强杀退出。测试控制 API 仅存在于 `tests.lifecycle_server`。

浏览器手工验证：空闲启动 → 显式运行 → 刷新重连 → 取消 → 等待保存 → 服务重启仍空闲
→ 手动恢复 → 强杀 → 重启确认临时进度已丢弃。Network 中初始化不能出现启动、恢复 POST
或 `rag/status?probe=true` 请求。

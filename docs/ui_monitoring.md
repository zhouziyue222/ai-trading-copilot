# 用户工作台与开发诊断

## 启动

在项目根目录执行 `uv sync --frozen --extra dev` 安装当前项目的独立虚拟环境。

```powershell
.\scripts\start-ui.ps1
```

浏览器打开 http://127.0.0.1:8000 。脚本明确使用本项目 `.venv` 并在 Python 启动前设置 UTF-8，避免 PATH 指向另一个同名项目。无需修改系统区域设置或用户 PowerShell profile。

开发诊断仅在显式启用后开放，并且后端仅接受本机客户端连接：

```powershell
.\scripts\start-ui.ps1 -Internal
```

打开 http://127.0.0.1:8000/internal/observability 。不要将此模式通过代理转发给其他用户；远程部署需要独立认证后才可开放内部接口。

已有 OTel 启动脚本 `scripts/start-ui-with-otel.ps1` 也已统一 UTF-8。OTLP 导出可选，未配置时使用本地诊断；导出失败不影响分析完成。

## 用户操作

1. 选择自选标的或输入代码。
2. 使用默认配置启动分析，高级参数按需展开。
3. 查看“准备数据、分析标的、形成建议、风险检查、生成报告”五个阶段。
4. 阅读结果摘要；完整报告和模拟操作位于折叠区。
5. 如需模拟下单，保留独立确认。Live 仍然只读。

黄色表示进行中、等待或数据不完整，绿色表示成功，红色表示确认失败。颜色同时配合图标和文字。进度反映节点完成情况，不是剩余时间预测。步骤超过 60 秒会提示耗时较长，不推断死锁。状态轮询不重叠，连续三次失败显示连接中断并保留最后状态。

刷新页面只重新连接当前服务的活任务，不启动新分析。服务会话变化后清空旧选择，历史文件不能决定全局运行状态。页面初始化不执行 RAG 探测。

## 取消、退出和检查点

| 操作 | 数据处理 |
| --- | --- |
| 正常完成 | 发布报告到 `reports/run_*` |
| 点击取消并保存 | 保存最后完整节点，丢弃被打断节点的变化；提交完成后才显示可恢复 |
| 手动恢复 | 创建新 Run ID，消费原检查点，从未完成节点继续 |
| Ctrl+C、服务退出、自动重载 | 丢弃正在运行的状态，不创建检查点 |
| 强杀或断电 | 进程无法执行清理；下次启动先删除遗留临时计算目录 |
| 普通异常 | 不创建断点，保存错误摘要供诊断 |

临时计算位于 `reports/.runtime/<session_id>/<run_id>`；只有显式取消生成 `reports/.checkpoints/<id>.json`。检查点包含带版本的类型化状态、节点位置和完整节点报告。原子写入、取消/完成互斥、启动实例锁用于避免半保存及重复管理。启动清理拒绝越界路径和目录链接，失败时不进入就绪状态。

取消是协作式停止：不可取消的外部调用可能需要等到返回或超时。服务退出最多等待线程 5 秒，Windows Job Object 在进程退出时结束所属子进程。不要把报告文件、日志或旧 `run_status.json` 当作可恢复检查点。

生命周期与检查点格式继续沿用主目录实现，详见 [UI 生命周期与 UTF-8](ui_lifecycle.md)。本次移植仅增加 UI 与监控。

## API

| 接口 | 用途 |
| --- | --- |
| `GET /api/runtime` | 当前 Session ID、进程 ID、全局状态、存活任务与最近进展 |
| `POST /api/runs` | 显式创建分析 |
| `GET /api/runs/{run_id}` | 当前或已结束任务详情 |
| `POST /api/runs/{run_id}/cancel` | 仅取消当前受控任务；无执行器的旧运行返回 409 |
| `GET /api/checkpoints` | 可恢复断点列表 |
| `POST /api/checkpoints/{id}/resume` | 显式恢复；已消费或不兼容返回 409，缺失返回 404 |
| `GET /api/internal/observability?query=...` | 按完整 Request ID、Run ID 或 Trace ID 查询内部事件 |

JSON 正常响应和错误响应显式使用 `application/json; charset=utf-8`。HTTP 响应带 `X-Request-ID` 与 `X-Session-ID`。文件和普通文本响应保留自身媒体类型。

## 故障定位

用户点击“复制故障编号”后，开发人员在内部入口粘贴 Request ID。索引会关联创建请求与后台执行，正常轮询的新 Request ID 不覆盖 `origin_request_id`。

调用链为 HTTP 接收 → 后台运行 → 图节点 → 工具/模型 → 外部请求。线程显式传递 Context，服务请求通过 `traceparent` 传播。启用 OTel 时本地 Span ID 与 SDK 导出的 ID 一致，TracerProvider 在服务内复用。

内部页面提供运行线程观察、最近业务进展、工具/模型/外部请求耗时、缓存命中、异常分组及事件时间线。API 心跳只是对线程存活的观察，不能证明计算正在推进。外部请求耗时包含网络和服务端处理，不等于纯网络 RTT；父子耗时不能相加。p50/p95 明确标注当前查询窗口。

日志包含 Request/Run/Trace/Span ID、UTC 时间、单调时钟耗时、异常类型、脱敏堆栈、代码路径、函数、行号和源码哈希。生产部署可设置 `COPILOT_BUILD_VERSION` 为 Git 提交或构建版本；开发工作树的代码位置需对照源码哈希，修改源码后应重启复现。

输入仅保留允许的标的、日期范围等字段，输出默认只记录长度和摘要哈希。认证信息、账户字段和自由文本中的凭据在写入前脱敏；不记录堆栈局部变量。取消不计入系统错误率。监控数据与计算状态独立，不能用于断点恢复。

诊断 SQLite 最多保留 10000 条事件，查询最多返回 1000 条并注明窗口；UTF-8 JSONL 单文件 5 MB、两个轮转备份。导出健康显示最近成功时间与失败批次，不能把“已配置”解释为“已送达”。正常 GET 轮询不产生 INFO 事件。

## 验证

```powershell
.\.venv\Scripts\python.exe -X utf8 -m pytest tests/test_run_lifecycle.py tests/test_ui_process_lifecycle.py tests/test_diagnostics.py tests/test_ui_app.py tests/test_tracing.py -q
```

独立模拟服务可用于浏览器验证，不访问行情、LLM 或券商：

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tests.ui_smoke_server --root tmp/ui-smoke --port 8017
```

启动模拟分析后会停留在“形成建议”，点击取消可保存检查点，手动恢复后继续完成。该测试服务仅用于开发验收。

在全新的模拟报告目录上，可用真实浏览器运行回归脚本：

```powershell
npx --yes --package @playwright/cli playwright-cli -s=copilot-ui open http://127.0.0.1:8017
npx --yes --package @playwright/cli playwright-cli -s=copilot-ui run-code --filename tests/ui_browser_smoke.js
```

该脚本检查刷新不触发启动、取消阶段不显示成功、手动恢复，以及跨标签页结束任务后历史视图的全局状态同步。

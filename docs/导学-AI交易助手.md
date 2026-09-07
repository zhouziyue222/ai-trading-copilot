# AI Trading Copilot 项目导学

> 使用说明：本文只采用当前工作树中可由代码、配置、测试和文档核验的事实。个人职责、业务采用和线上收益等无法从仓库确认的内容统一写为“待补充”，不得直接当作面试事实。

## 1. 前置知识（面试高频标注）

| 知识点 | 为何需要 | 在本项目中的位置 | 高频度 |
| --- | --- | --- | --- |
| Pydantic 与领域建模 | 理解 Agent 之间为何不用自由文本裸传递 | `copilot/domain/models.py`、`copilot/graph/state.py` | 高 |
| LangGraph 状态图 | 理解条件扇出、汇聚与顺序降级 | `copilot/graph/copilot_langgraph.py` | 高 |
| 确定性风控 | 解释 LLM 与不可越权规则的边界 | `copilot/agents/risk_agent.py`、`portfolio_manager.py` | 高 |
| RAG 检索链 | 解释 Query Rewrite、向量/BM25、RRF、重排和父上下文 | `copilot/services/rag_store.py` | 高 |
| 版本化记忆 | 解释 Candidate、Shadow、Approved、回滚和审计 | `copilot/services/memory_*.py` | 高 |
| FastAPI 与异步任务状态 | 解释 UI 如何启动、取消和轮询运行 | `copilot/ui/app.py`、`copilot/services/run_tracker.py` | 中 |
| OpenTelemetry 与本地 Trace | 解释如何定位节点、工具和 LLM 失败 | `copilot/services/tracing.py`、`observability/` | 中 |
| 离线评测与测试门禁 | 区分功能正确、安全门禁和模型质量 | `copilot/evaluation.py`、`copilot/harness_benchmark.py`、`tests/` | 高 |

## 2. 重点亮点与学习顺序（先看这个）

| 亮点标题 | 为什么重要 | 通用技术关键词 | 先看哪些文件 | 建议学习顺序 |
| --- | --- | --- | --- | --- |
| 结构化状态编排 | 多 Agent 不靠聊天串联，而靠明确状态契约汇聚 | 状态图、条件路由、扇出汇聚、降级 | `copilot/graph/state.py`、`copilot/graph/copilot_langgraph.py` | 1 |
| 确定性安全边界 | 把高风险决策从概率模型输出中剥离 | 资格门禁、仓位 clamp、总敞口、fail closed | `copilot/agents/trader_agent.py`、`risk_agent.py`、`portfolio_manager.py` | 2 |
| 双知识域检索 | 基本面资料和交易经验的语义、权限、生命周期不同 | 混合检索、事实源、版本、权限隔离 | `copilot/services/rag_store.py`、`memory_repository.py`、`memory_retrieval.py` | 3 |
| 受控学习闭环 | 运行后经验不会未经验证进入生产 Prompt | Candidate/Shadow/Approved、延迟结果、回滚 | `copilot/services/memory_learning.py`、`memory_evaluation.py` | 4 |
| 可观测与可复盘 | 每次建议都能追溯节点、工具、规则和异常 | Trace、OTel、审计、运行报告 | `copilot/services/tracing.py`、`run_tracker.py` | 5 |
| 多层质量门禁 | 用单测、工作流评测、RAG 指标和固定安全集覆盖不同风险 | pytest、smoke eval、RAGAS、holdout | `copilot/evaluation.py`、`copilot/harness_benchmark.py` | 6 |

## 3. 必备知识点

- 理清 `CopilotGraphState` 中输入、分析证据、计划、风控、组合决策、记忆和 Trace 的字段流转。
- 能说明分析节点为什么可以扇出，而 Trader、Risk Manager、Portfolio Manager 必须顺序执行。
- 能手算单标的上限和组合总敞口缩放，理解削减的敞口为何保留为现金。
- 能区分 Fundamental RAG 与 Trading Memory，避免把两者说成一个向量库。
- 能解释为什么 Shadow 只记录命中、不注入 Prompt，以及 Approved Memory 为什么仍不能覆盖实时证据和硬风控。
- 能区分“229 个测试通过”“离线 smoke 满分”与“真实交易收益有效”三种完全不同的证据等级。
- 能说明 simulation 运行、Futu SIMULATE 二次确认和不支持 Live 真实订单之间的边界。

## 4. 推荐阅读（结合仓库）

| 主题 | 通用技术点 | 建议阅读位置 | 预计时间 | 读完能回答什么 |
| --- | --- | --- | --- | --- |
| 总览 | 产品边界与系统地图 | `README.md`、`docs/pm_prd.md` | 30 分钟 | 项目解决什么问题、不做什么 |
| 状态与编排 | LangGraph 路由、汇聚、fallback | `copilot/graph/state.py`、`copilot/graph/copilot_langgraph.py` | 60 分钟 | 节点如何串联，失败如何降级 |
| 计划与风控 | 概率推理和确定性规则隔离 | `copilot/agents/trader_agent.py`、`risk_agent.py`、`portfolio_manager.py` | 90 分钟 | AI 为什么不能越权，权重如何计算 |
| 基本面 RAG | 混合召回、RRF、重排、父子分块 | `copilot/services/rag_store.py`、`docs/vector_memory_plan.md` | 90 分钟 | 为什么不用纯向量召回，如何评测 |
| 交易记忆 | SQLite 事实源、版本、检索、学习 | `copilot/services/memory_repository.py`、`memory_retrieval.py`、`memory_learning.py` | 120 分钟 | 如何防止错误经验污染生产决策 |
| 评测体系 | 工作流、单 Agent、RAG、Harness | `copilot/evaluation.py`、`copilot/harness_benchmark.py`、`docs/memory_harness.md` | 75 分钟 | 当前证据能证明什么，不能证明什么 |
| UI/API | 任务创建、轮询、取消、模拟单确认 | `copilot/ui/app.py`、`copilot/ui/static/app.js` | 60 分钟 | 前后端如何管理长任务和执行边界 |
| 测试证据 | 边界用例和回归意图 | `tests/` | 90 分钟 | 关键规则由哪些测试保护 |

## 5. 自学提醒

若某文件或原理看不懂，请继续追问 AI；本技能负责给学习路径与题目，不提供逐行讲解。

## 6. 项目技术定位

这是一个 AI 应用工程、后端系统与产品设计交叉项目：核心不是训练基础模型，而是把 LLM、多源金融数据、确定性风控、检索记忆、Web/CLI 交互和可观测性组合成可验证的本地决策辅助系统。

## 7. 核心原理解析

### 7.1 结构化多智能体编排

问题：自由文本串联容易造成字段漂移、错误难定位、节点难替换。机制：以 Pydantic 领域模型和 `CopilotGraphState` 作为状态契约，分析节点按用户选择条件扇出，Trader 汇聚证据后再进入风控与组合决策；缺少 LangGraph 或显式配置时走相同节点顺序的降级执行。落点：`copilot/graph/state.py` 与 `copilot/graph/copilot_langgraph.py`。

### 7.2 LLM 与硬规则分层

问题：模型擅长综合与解释，但不适合独自承担资金安全边界。机制：Trader 的买入建议先经过确定性买入门禁，Risk Manager 再执行订阅资格、禁用品种、单标的上限和组合总敞口 clamp，Portfolio Manager 只把风控后的权重差转换为动作。落点：`trader_agent.py`、`risk_agent.py`、`portfolio_manager.py`。

### 7.3 双知识域与权限隔离

问题：公司基本面材料与历史交易经验在更新频率、可信度和使用边界上不同。机制：基本面材料进入 Chroma，通过 Query Rewrite、向量与 BM25 双通道、RRF、重排和父上下文聚合供 Fundamental Analyst 使用；交易经验以 SQLite 为事实源，只向 Trader 和受限的 Portfolio Memory Advisor 提供 Approved 记忆。落点：`rag_store.py` 与 `memory_*` 服务。

### 7.4 版本化记忆闭环

问题：一次运行反思可能错误，若直接写入 Prompt 会放大偏差。机制：LangMem 只生成提案，Candidate 经过 Shadow 命中和延迟市场结果评测后才能审批为 Approved；更新和回滚都追加版本，保留审计历史。落点：`memory_learning.py`、`memory_repository.py`、`memory_evaluation.py`。

### 7.5 执行安全与人工确认

问题：研究建议容易被误解为自动交易。机制：运行阶段只生成 `ExecutionDecision`；Futu `SIMULATE` 订单需要在结果页再次确认，Live 组合仅用于只读上下文，真实 Live broker order 不受支持。落点：`copilot/ui/app.py` 与 `copilot/adapters/futu_execution.py`。

### 7.6 可观测与多层评测

问题：仅凭最终建议无法判断模型、数据、工具还是规则出了问题。机制：节点、工具、LLM、规则命中、耗时和错误进入本地 Trace，并可导出 OTLP；质量侧分为 pytest 回归、工作流/单 Agent smoke、RAG 检索指标和固定 Memory/Safety/Prompt holdout。落点：`tracing.py`、`run_tracker.py`、`evaluation.py`、`harness_benchmark.py`。

## 8. 关键设计决策

### 决策一：订阅池，而不是全市场自由推荐

- 备选：让 LLM 从全市场直接选股。
- 取舍：订阅池牺牲发现范围，换取用户意图明确、资格可校验和推荐风险可控。
- 风险：用户可能错过池外机会。
- 验证：比较订阅池任务完成时间、报告阅读率和误推荐率；当前业务指标待补充。

### 决策二：确定性风控，而不是让 LLM 自评风险

- 备选：由 LLM 同时给建议和最终仓位。
- 取舍：硬规则表达力较弱，但可复现、可单测、不可被提示词绕过。
- 风险：固定阈值可能不适配不同市场状态。
- 验证：边界单测、随机权重性质测试和历史组合回放；线上阈值效果待补充。

### 决策三：RAG 与交易记忆分库分权

- 备选：全部内容写入同一向量库并由所有 Agent 检索。
- 取舍：分离增加实现复杂度，但避免不同可信度和生命周期的知识混用。
- 风险：跨域信息可能重复或不一致。
- 验证：检索命中审计、来源标签、权限测试与 RAG 对比评测；在线 RAGAS 结果待补充。

### 决策四：Shadow 后再 Approved

- 备选：运行后反思立即写入生产 Prompt。
- 取舍：延迟晋升降低学习速度，但显著减少未经验证经验污染决策的风险。
- 风险：真实结果积累慢，冷启动时间长。
- 验证：至少 3 次 Shadow 命中、3 份延迟结果、置信度不低于 0.65、相对基准收益和回撤门禁。

### 决策五：本地优先存储与可重建索引

- 备选：直接使用托管向量数据库和云端运行记录。
- 取舍：本地 JSON/SQLite/Chroma 降低部署与隐私成本，但多用户、协作和容量能力有限。
- 风险：本地文件损坏、并发和迁移能力受限。
- 验证：版本回滚、原子写入、重建索引和并发测试；规模与恢复演练待补充。

## 9. 量化与验证（含待测，建议）

### 已验证（2026-09-04，当前工作树）

- `uv run python -m pytest tests -q`：229 个测试通过，耗时 14.63 秒。
- 固定 Memory/Safety/Prompt Harness：11/11 通过，`pass_rate`、`safety_gate_pass_rate`、`memory_isolation_rate`、`prompt_contract_rate` 均为 1.0，成本字段为 0。
- 确定性工作流 smoke：overall 1.0；10 个必需节点、8 类报告、结构化输出、3 项决策比较和 1 项安全比较均通过；单次本地 Trace 约 14 ms。
- 单分析器离线 smoke：3 个样例全部通过，字段完整率与期望匹配率均为 1.0；平均 0.215 ms、P95 0.515 ms。该耗时不包含外部 LLM、网络和真实数据源，不可当作线上延迟。

### 待测与建议

- 【待补充：测试覆盖率、关键模块分支覆盖率，以及 CI 是否设门禁】
- 【待补充：真实 LLM + Futu/yfinance/Finnhub 的端到端 P50/P95 延迟、失败率、降级率与单次成本】
- 【待补充：RAG hit@k、context recall/precision、MRR、NDCG、faithfulness、answer relevancy 的真实跑分】
- 【待补充：并发运行、取消、重复确认、幂等模拟单和恢复场景的负载/故障注入结果】
- 【待补充：机会状态与后续走势的离线回放样本量、基准、时间跨度、收益与最大回撤】
- 【待补充：真实用户数、使用频次、报告阅读/采纳率、决策耗时与可解释满意度】
- 【待补充：个人职责、团队规模、开发周期、最难问题及个人提交/PR/演示证据】

# 记忆系统与 Harness 闭环

## 设计结论

本项目采用“SQLite 事实源 + LangMem 提炼 + ACE 式候选管理 + Shadow 延迟评测 + Autoharness 离线优化”的组合。记忆系统可以持续积累证据，但生产决策只读取人工/门禁批准的记忆；LLM 不能修改确定性风控规则。

数据流如下：

```text
完成一次分析
  -> LangMem Reflector 提炼可验证观察
  -> SkillManager 去重/合并
  -> candidate
  -> 人工送入 shadow
  -> 运行时只记录命中，不注入 Prompt
  -> 登记延迟市场结果
  -> holdout + 收益/回撤 + 安全门禁
  -> approved
  -> 分析师先完成当前市场分析
  -> Trader 按结构化分析强制精召回最多 3 条
  -> Trader 可追加工具查询 1 次，总计最多 6 条
  -> Risk Manager 执行确定性门禁
  -> Portfolio Manager 仅在扩大敞口时精召回
  -> 记忆只能缩小或否决新增风险
```

## 为什么这样组合

- `SQLiteMemoryRepository` 是唯一事实源，保存当前记录、所有历史版本、状态事件、运行命中、延迟结果和评测结果。
- 系统不再读写 JSONL；SQLite 同时承担事实源、版本和审计历史。
- LangMem 只负责从结构化运行证据中提炼 insert/update 提案，不维护第二套 Store，避免双写与状态漂移。
- `MemorySkillManager` 做相似候选合并。只有 `candidate` 可原 ID 更新；匹配到其他生命周期状态时会创建独立修订候选，绝不原地改写已冻结记忆。
- 运行前不再设置粗粒度 `Retrieve Memories` 节点。Trader 在分析师完成后、Portfolio Manager 在 Risk Check 后，各自按当前结构化状态检索。
- 每次查询由 SQLite 做状态、有效期、Scope、Symbol 和市场状态硬过滤，再以标签、关键词、置信度和本地向量相似度重排；SQLite 仍是唯一事实源。
- Shadow 记忆使用相同精查询并登记命中，但不会进入任何 Agent Prompt，因此可以先积累结果证据。
- Trader 和 Portfolio Manager 都先由程序强制预取，再允许 LLM 通过 `search_trading_memories` 追加查询一次，避免 LLM 漏查。
- Autoharness 只能改 `config/prompts`。Risk Manager、Portfolio Manager、评测代码、holdout 集和测试均为受保护面。

## 运行时读取与权限

- 分析师不读取记忆，只生成当前市场、技术、新闻和基本面证据。
- Trader 查询市场状态、趋势、支撑距离、Reward/Risk、事件风险和 Persona。输出必须用 `memory_id@version` 引用实际采用的记忆，伪造引用会被删除。
- Risk Manager 不读取记忆，继续独立执行仓位、总敞口和交易资格门禁。
- Portfolio Manager 只有在 `abs(final_weight) > abs(current_weight)` 时查询记忆。缩放比例被确定性限制在 `[0, 1]`，不能放大仓位、改变方向，也不能阻止减仓、卖出或平空。
- 没有 LLM 时继续走原确定性流程，不召回也不登记未实际使用的记忆。
- `memory_usage.mode` 区分 `trader`、`trader_applied`、`portfolio_manager`、`portfolio_applied` 和 `shadow`；运行末统一生成检索审计报告。

## 生命周期

| 状态 | 是否进入生产 Prompt | 用途 |
| --- | --- | --- |
| `candidate` | 否 | LangMem 新提炼或重新审查的观察 |
| `shadow` | 否 | 记录在真实运行中的相关性并等待市场结果 |
| `approved` | 是 | 通过门禁后供后续分析检索 |
| `deprecated` | 否 | 已停用、可审计、可恢复 |
| `rejected` | 否 | 人工或安全门禁拒绝 |

允许的状态迁移由 Repository 强制执行。任何更新都创建递增版本；回滚也是创建新版本，而不是删除历史。

## 运行与审批

正常 CLI/UI 运行结束后会调用 LangMem。没有配置 `DEEPSEEK_API_KEY` 时，主分析照常完成，但不会伪造候选记忆。单次运行可关闭反思：

```powershell
ai-trading-copilot AAPL --no-memory-learning
```

长期记忆默认开启。`COPILOT_LONG_TERM_MEMORY_ENABLED=false` 可全局关闭；
CLI 的 `--long-term-memory/--no-long-term-memory` 和 Web 单次运行开关优先于环境变量。
关闭总开关后既不提炼也不检索，但审批、评测和历史查看仍可使用。

管理 CLI：

```powershell
ai-trading-copilot-memory status
ai-trading-copilot-memory list --status candidate
ai-trading-copilot-memory transition MEMORY_ID shadow --reason "人工确认可进入影子期"
ai-trading-copilot-memory evaluate MEMORY_ID
ai-trading-copilot-memory transition MEMORY_ID approved --reason "shadow 门禁通过"
ai-trading-copilot-memory versions MEMORY_ID
ai-trading-copilot-memory rollback MEMORY_ID 2
```

Web UI 的“记忆审批”页提供相同的候选、Shadow、评测、晋升、拒绝和停用入口。

## 延迟结果

Shadow 晋升默认至少需要：

- 3 个不同运行中的 Shadow 命中；
- 3 份对应的延迟市场结果；
- 置信度不低于 0.65；
- 按记忆的 `validation_target` 方向计算后，平均相对基准收益不小于 0（看空/回避型记忆会反向计分，`risk_reduction` 只走回撤门禁）；
- 最差回撤不低于 -20%；
- 不包含绕过 Risk Manager、止损或人工确认的语义。

登记结果：

```powershell
ai-trading-copilot-memory outcome run_AAPL_20260820_090000 AAPL `
  --horizon-days 5 `
  --realized-return 0.032 `
  --benchmark-return 0.011 `
  --max-drawdown -0.045
```

也可以通过 UI 或 `POST /api/memory-outcomes` 写入。结果按 `run_id + symbol + horizon_days` 幂等更新。

## Autoharness

仓库已提供：

- `autoharness.yaml`：bounded 模式与可编辑/受保护面；
- `benchmarks/memory-screening.yaml`：generic command benchmark；
- `config/memory_harness_holdout.json`：固定且受保护的 holdout 合约；
- `copilot/harness_benchmark.py`：输出 Autoharness 可解析的 metrics/task results。

安装 Autoharness 后：

```powershell
pipx install "git+https://github.com/kayba-ai/autoharness.git"
autoharness doctor
autoharness run-benchmark
autoharness optimize
autoharness report
```

Autoharness 的 champion 仍需走本项目的硬门禁和 Git 晋升命令：

```powershell
ai-trading-copilot-harness benchmark
ai-trading-copilot-harness promote --message "harness: promote trader prompt v2"
```

`promote` 只提交 `config/prompts`，不会带入其他已暂存或未暂存文件。回滚会先恢复目标 Git 版本，再重新跑完整硬门禁，通过后生成新的回滚提交：

```powershell
ai-trading-copilot-harness rollback COMMIT_SHA
```

当前 holdout 是安全/隔离/版本/Prompt 合约的离线集。应在积累足够真实案例后增加决策质量、延迟、成本和 LLM judge 样本；不能用训练/提案集替换固定 holdout。

## 运维文件

- `config/memory.sqlite3`：本地事实库，已忽略 Git。
- `config/memory.vectors.json`：可从 SQLite 当前记忆重建的本地相似度侧索引，已忽略 Git。
- `.autoharness/`：campaign/champion 工作状态，已忽略 Git。
- 每次 Prompt 晋升：独立 Git commit，可由 `ai-trading-copilot-harness rollback` 恢复并重新验证。

## 安全边界

- 记忆永远是上下文，不是订单指令。
- 只有 `approved` 会进入 Prompt；每个 Agent/标的强制预取最多 3 条、最多追加查询一次，总计最多 6 条去重记忆。
- 记忆不能修改最大仓位、总敞口、禁用工具或确认要求。
- 自动晋升默认关闭；设置 `COPILOT_MEMORY_AUTO_PROMOTE=1` 后，登记延迟结果会自动复评相关 Shadow 记忆，但仍必须通过同一硬门禁。
- 没有真实延迟结果时，不允许把一次运行中的“判断”当作已验证收益规律。

# 延迟复盘与记忆学习

每份新计划先保存到原有记忆 SQLite，再登记第 5、10、20 个完整交易日的观察任务。CLI 和 UI 使用同一个登记入口；报告目录清理不会删除复盘证据。即时反思失败也不会撤销已经登记的任务。

## 自动运行与配置

UI 服务启动时补做任务，之后每小时检查一次。关闭服务后不再领取任务；正在执行的请求受 SQLite 租约保护，异常退出后可重新领取。没有常驻 UI 时，用下面的命令运行相同流程；本功能不会安装系统定时任务。

```powershell
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review run-due
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review list
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review show REVIEW_ID
```

可用 `--memory-database PATH` 指定数据库，放在子命令前。安装项目后也可使用 `ai-trading-copilot-review`。

默认不读取任何交易账户。只读同步必须显式配置账户环境和账户 ID，示例中的数字需替换为实际账户 ID：

```powershell
$env:COPILOT_REVIEW_ACCOUNTS = '[{"trd_env":"SIMULATE","acc_id":123},{"trd_env":"REAL","acc_id":456}]'
$env:COPILOT_REVIEW_BENCHMARKS = '{"US":"SPY","HK":"HK.02800"}'
$env:COPILOT_REVIEW_HORIZONS = '[5,10,20]'
```

同样可以写进项目 `.env`。沿用 `FUTU_OPEND_HOST`、`FUTU_OPEND_PORT`、`FUTU_SECURITY_FIRM` 和现有模型配置。基准优先匹配标的，其次匹配市场 `US`、`HK`、`CN`；不猜测未配置的基准。周期必须为正数，且必须包含批准评估所需的 20 日周期。

富途适配器只调用行情、交易日历、订单与成交查询，不发现未配置的账户，不解锁账户、不提交实盘订单。若账户不支持所请求的历史范围，显示等待原因；不能用“今天没有成交”替代历史完整性。

## 页面与人工操作

进入 `/reviews`，查看每个周期的状态、下次检查时间、等待原因、行情指标、执行偏差和反思证据；已有记忆审批页仍负责评估与批准。

- 已知订单 ID 自动关联对应计划；不同账户、不同环境的同名订单和成交不会混用。
- 外部成交或有多个候选计划的成交列为待关联。页面或 CLI 确认关联后，下一周期会纳入它。
- 已完成的周期可手动重试，重新同步执行证据并反思。旧结果进入 `review_result_history`，原始计划、行情窗口和反思上下文保留；普通定时检查不会重复执行已完成任务。
- 已经关联的成交不能静默改挂到另一个计划。券商返回矛盾成交时保留原记录并提示核对。

```powershell
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review fills
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review associate --environment REAL --account-id 456 --deal-id DEAL_ID --run-id RUN_ID --symbol AAPL
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review retry REVIEW_ID
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review run-due
```

历史运行只能显式导入完整 JSON 状态：须包含带时区的 `generated_at`、`run_id`、`trade_plans`、`risk_assessments`、`execution_decisions` 及原始行情证据。不会从缺失证据的报告推测原计划。

```powershell
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.review import-snapshot complete_state.json
```

API 与这些操作对应：`GET /api/reviews`、`GET /api/reviews/{id}`、`POST /api/reviews/run-due`、`POST /api/reviews/{id}/retry`、`GET /api/review-fills`、`POST /api/review-fills/associate`。

## 观察与反思口径

交易日历提供开盘、收盘时间，处理节假日、半日市和夏令时。开盘前生成的计划可以计入当天；盘中或盘后生成则从后续完整交易日开始。只在截止交易日收盘后计算，缺日线、停牌、复权信息不明或窗口内存在除权影响时等待，不把后面的行情补进固定窗口。

假设观察以首个完整交易日开盘为基准，显示标的收益与计划方向收益；这是标准化行情观察，不是条件单回测或实际成交收益。止损和目标只记录日线是否触及，同日同时触及时顺序未知。自然语言失效条件缺少可验证证据时保持不确定。

执行结果只用明确关联、发生在计划之后且截止时间之前的成交。平均成本法计算分批成交和退出，分别显示已实现盈亏、未实现盈亏；费用不可用时显示未扣费用。计划资金收益按累计开仓名义金额归一化，回撤按每日收盘权益计算，并与同一观察窗口的基准比较。该结果不代表整个账户的资金加权收益。缺少期初持仓依据的卖出/回补、超量反向成交不推测盈亏，也不参与批准评估。

订单状态独立保存，不能用订单累计成交量代替成交明细。只采用截止时点可确认的订单历史状态；后来的撤单、改单状态不得倒灌早期复盘。

行情结果、成交同步、反思输入与模型输出分阶段保存。模型失败后复用已存行情；模型只能引用本次证据。既有记忆按截止时间读取历史版本，反思输入在第一次调用前冻结，防止第 20 日经验进入第 5 日重试。

新经验通过安全检查后自动进入 Shadow，仍不进入生产提示词。candidate 可以原位更新；任何冻结状态的内容变化通过 `save_candidate` 建立 revision。第 5、10 日是阶段反馈，批准默认只使用第 20 日结果。同一运行的多个标的合并为一个独立样本，不把三个周期算成三笔交易。假设、模拟、实盘及具体账户分别评估；生成经验的来源运行不验证自身。

默认门槛仍为至少 3 个独立运行、置信度至少 0.65、相对收益和回撤门禁。缺基准不能通过依赖相对收益的门槛。延迟复盘经验始终人工批准，即使旧的 `COPILOT_MEMORY_AUTO_PROMOTE` 已开启。关闭该次运行的记忆学习后仍登记和观察计划，但不调用延迟反思模型、不生成新经验。

新增表均在原 SQLite 内增量创建；旧手工 `RunOutcome` API 保留，未分类的历史结果不计入新延迟经验的批准依据。本流程不读写 JSONL，已有独立下单审计格式保持原状。

## 可复现验收

使用项目 `.venv311/Scripts/python.exe`，所有交易与模型验收使用固定假数据，不连接账户，不提交订单。测试覆盖完整生命周期、跨节假日与开盘边界、冻结版本、来源隔离、多周期去重、部分成交、订单时间、关联歧义、并发租约、服务重启、模型失败与后补复盘。

```powershell
& ./.venv311/Scripts/python.exe -m pytest tests/test_review_data.py tests/test_review_execution.py tests/test_review_interfaces.py tests/test_delayed_review.py tests/test_delayed_memory_evaluation.py tests/test_memory_learning.py tests/test_memory_repository.py tests/test_memory_store.py tests/test_memory_retrieval.py tests/test_memory_cli.py tests/test_memory_ui_api.py tests/test_memory_eval.py tests/test_copilot_langgraph.py tests/test_llm_decision_agents.py tests/test_portfolio_manager.py tests/test_ui_app.py tests/test_cli_run.py tests/test_config_files.py tests/test_react_tracing.py tests/test_run_lifecycle.py -q --basetemp tmp/delayed-review-verification --junitxml tmp/delayed-review-verification.xml
& ./.venv311/Scripts/python.exe -m ai_trading_copilot.copilot.harness_benchmark
```

2026-09-09 验收：**265 passed、1 skipped**；跳过项是当前 Windows 无法创建符号链接的生命周期测试。原 memory harness **11/11** 通过。机器可读结果、环境和源码哈希见 [delayed_review_baseline.json](delayed_review_baseline.json)。工作区同时包含其他未提交改动，复现应核对记录中的文件哈希，不能只依赖 Git 提交号。

浏览器验收使用独立临时数据库，确认列表与详情显示原计划、三个周期及对应指标，并验证跳转到记忆审批和刷新队列。离线验收证明流程与隔离约束，不证明收益提升。

### 2026-09-10 OpenD 与真实模型验证

实测 SDK 10.9.6908，OpenD 连接成功；真实交易日历与历史日线读取成功。AAPL 的初始查询窗口包含 8 月 10 日除权事件，适配器正确标记不可直接评价；改用 8 月 11 日起的完整 20 交易日窗口，截止 9 月 8 日收盘。

- 实盘账户（尾号 7042）的订单和成交查询成功，2026-08-10 至 2026-09-09 均为 0 条。
- 美股模拟账户（尾号 1724）的订单查询成功；成交查询返回“模拟交易不支持成交数据”，无法据此承诺模拟账户历史成交同步可用。
- DeepSeek `deepseek-v4-pro` 使用公开行情与明确标注的联调观察样例完成结构化反思，返回空经验列表，没有把观察结果冒充真实交易。
- 生产数据库没有原始计划快照；未验证非空实盘成交归因，没有修改生产库或账户/基准配置，没有提交订单。
- 用户明确授权发送实盘订单、成交明细至 DeepSeek 后，已完成实盘只读查询到模型反思的联合步骤：`execution_complete=true`、无同步错误、结构化反思存在，重试前后的原始行情一致。该窗口成交为空，因此仍不代表非空实盘交易归因已验证。

脱敏结果见 [真实联调记录](delayed_review_live_20260910.json)。联合流程已通过空成交窗口验证，不应将其解释为非空真实交易的端到端复盘已通过。

模拟账户报错不是“没有数据”：富途的[历史成交查询](https://openapi.futunn.com/futu-api-doc/trade/get-history-order-fill-list.html)和[当日成交查询](https://openapi.futunn.com/futu-api-doc/trade/get-order-fill-list.html)均明确仅支持实盘。模拟订单查询是另一个接口，可以使用；订单累计成交量不能直接冒充逐笔成交明细。

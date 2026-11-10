# 408 考研 RAG + Agent 备考助手 · 面试准备

> 用途：面试时介绍项目、应对追问。本文只使用仓库中可核对的代码、报告与评测数据。
> 核对时间：2026-11-05。凡代码/报告里没有的数字，统一标注「待补」，不做估算。
>
> 主要证据：
> - 代码：`src/cs408rag/`、`app/streamlit_app.py`、`tests/`
> - 报告：`reports/ablation_dev.md`、`reports/ablation_holdout.md`、`reports/ablation_rerank.md`、`reports/agent_eval.md`、`reports/adversarial_eval.md`、`reports/claim_eval.md`
> - 数据：`data/eval/multi_turn_report.json`、`data/eval/eval_set_stats.json`
> - 文档：`README.md`、`docs/optimization_log.md`

---

## 0. 速查卡（面试前 2 分钟看这一页）

| 项 | 事实 |
|---|---|
| 项目定位 | 面向 408 考研的「真题知识库 RAG + 有界 Agent 备考助手」，不是通用聊天机器人 |
| 数据规模 | 720 道真题（2009–2026）；5296 个 chunk；6182 个知识点节点；226 个真题标签 |
| 可用评测集 | dev 452 题有 ground truth；2023–2026 holdout 99 题；多轮 19 轮；Agent 抽样 21 题；60 题 held-out 端到端（记录在 optimization_log） |
| 检索链路 | dense + sparse + 原生 BM25 三路召回；weighted zscore 融合；小节去重；**生产不启用 rerank、不启用图谱路** |
| Agent 边界 | LangGraph StateGraph 编排；规则优先；最多 3 个工具轮次 / 90 秒；只有规则无法决策时才调 1 次 LLM planner |
| 终态 | 7 种 `stop_reason`：clarification / answered / no_evidence / tool_error / budget / max_rounds / verify_retry |
| 会话记忆 | 内存 LRU 200 会话、TTL 2h、最多保留 8 轮、prompt 只注入摘要 + 最近 4 轮；重启即失 |
| 成本 | 多轮评测 19 轮总成本 ¥0.1744；平均 ¥0.00918/轮（约 0.92 分）；按人民币单价表估算，不是账单 |
| 可信度 | 可溯源率 100%（97/97）；编造真题编号 0；押题 0；真题编号/押题校验失败最多重写 1 次 |
| 测试 | 本地全量 223 passed；CI 只跑 `tests/test_adversarial_unit.py` 10 个测试 |
| 最硬的负结果 | rerank 在 dev 上 Recall@10 −7.2pp、MRR −0.0425、P50 +246ms，因此生产关闭 |
| 最该补的证据 | holdout 上的 LLM-only / 单轮 RAG / Agent 端到端对照与显著性；并发/压测与真实账单成本 |

---

## 1. 电梯陈述

### 1.1 30 秒版本（可直接背）

> 我做的是一个 408 考研的 RAG + Agent 备考助手。它把 2009–2026 年的 720 道真题和教材知识切成 5296 个 chunk、6182 个知识点节点，先做 dense、sparse、BM25 三路召回和加权融合，再让一个**有界 Agent**按题型调用检索、图谱、看图、计算器、考频统计等 9 个工具。Agent 不是自由 ReAct：最多 3 轮、90 秒，必须落到 7 种明确终态，回答还要过真题编号/押题校验。项目最大的特点不是「接了大模型」，而是每个取舍都有评测数据：比如 rerank 实测是负收益，我们就把它关了。

### 1.2 2 分钟版本（按四段讲）

**第一段：问题。** 408 备考的痛点不是「没有资料」，而是资料太厚、真题太散：学生想知道「这个考点考过几次、先学什么、这道计算题怎么算」，通用大模型答得流畅，但给不出可核对的真题出处，也可能把 2024 年的题记成 2019 年。

**第二段：检索。** 我们把真题、解析、教材切成 5296 个 chunk，用 dense + sparse + BM25 三路召回，weighted zscore 融合，再做小节去重。这里有一个反直觉结论：bge-reranker-v2-m3 在这个任务上是负收益——dev 上 Recall@10 下降 7.2pp，MRR 从 0.335 降到 0.292，P50 从 85ms 涨到 331ms，所以生产不启用。图谱路对通用召回也是负贡献，只保留为按需工具；它在长尾上只能救 2.5%–3.6% 的失败题。

**第三段：Agent。** Agent 用 LangGraph StateGraph 做显式编排：route 先判题型；缺信息就 clarify 短路，不检索也不调 LLM；否则 retrieve → assess → 必要时 rewrite；计算、图表、考情题走 plan-act-observe，每轮只执行一个工具，最多 3 轮、总预算 90 秒。工具用 Pydantic 契约校验参数，每个工具有独立超时，失败不抛异常而是记录成 trace。最后 answer → verify，真题编号不在白名单或出现押题式表述就重写一次；`[n]` 与上下文编号的匹配度会被记录，但当前不作为硬拦截条件。

**第四段：结果与边界。** 19 轮多轮评测通过率 94.7%，澄清召回 100%，指代消解 100%，工具失败率 3.0%；21 题 Agent 抽样准确率 95.2%，单轮 RAG 85.7%，纯 LLM 90.5%，但置信区间还跨 0，所以我不会说「Agent 显著优于 RAG」。成本约 0.92 分/轮，是估算值。下一步最想补的是 holdout 上的端到端对照和并发压测。

### 1.3 要突出的三个差异化点

1. **有界 Agent，不是「让模型自由发挥」。** 规则优先、每轮一个工具、3 轮/90 秒硬上限、7 种终态、每次决策写进 `plan_history` 和 `steps`，可分层评测。
2. **评测先行，敢讲负结果。** rerank 负收益、图谱路负贡献、节去重牺牲 chunk recall 换 question accuracy，都有报告；不是只挑好看的数字。
3. **可信度工程。** 运行时会计算 `[n]` 与上下文编号的 precision/recall；硬拦截的是「真题编号不在白名单」和「押题/预测」表述；UI 的引用来源可展开看原文和教材插图。

---

## 2. 业务问题与目标用户

### 2.1 目标用户

- 408 考研的**一战/二战/跨考**学生，时间紧、资料多，需要「按考点」而不是「按章节」组织知识。
- 典型诉求：
  - 这个概念/算法是什么，能不能给我教材原文？
  - 这个考点历年考过几次、考过哪些真题、和哪些考点一起考？
  - 这道真题为什么选 B？我选的 A 错在哪？
  - 我下一步先学什么？薄弱科目怎么排时间？
  - 计算题（Cache 命中率、页面置换、TCP 窗口、码距）不要心算，给我确定结果。

### 2.2 为什么需要 RAG，而不是直接问大模型

讲的时候分三层，避免把 RAG 说成万能：

1. **可核对性。** 408 的答案是「有出处的」：某年某题、某本教材某节。通用模型的参数记忆无法给出可点击、可展开的原文；RAG 把 `citation`、`chunk_id`、`source` 带到回答里，UI 可展开核对。
2. **时效与私有数据。** 2023–2026 的真题、教材解析、图谱关系不在通用模型的可靠记忆里；而且我们明确把 2023–2026 作为 holdout，不能靠模型背题。
3. **结构化工具能力。** 「这个考点考几次」「和哪些考点共现」「先学什么」不是语义相似度问题，而是统计/图查询/规则规划问题。这些必须走 `get_exam_stats`、`get_related_kps`、`get_learning_path` 等确定性工具，RAG 只负责给证据。

**必须诚实的一点：** 在 dev 的 21 题抽样里，纯 LLM 准确率 90.5%，Agent 95.2%，单轮 RAG 85.7%；纯 LLM 在公开真题上并不弱，甚至计算题是 100%。所以 RAG 的价值不能只讲「准确率更高」，更要讲「可溯源、可核对、知识可更新、能做统计工具」。这也是为什么文档把「缺 holdout 端到端对照」列为最大短板。

---

## 3. 端到端链路

### 3.1 两个入口的分工

```text
┌────────────────────────────── Streamlit ──────────────────────────────┐
│  问答 tab                             Agent tab                         │
│  单轮、无历史                          多轮、有 session                     │
│  retrieve → top 8 blocks               route → clarify?                  │
│  → stream LLM answer                   retrieve → assess → rewrite?      │
│  → 引用来源可展开                       plan ⇄ act ⇄ observe (≤3 轮)      │
│  → 推荐真题                            answer → verify                   │
│  无工具、无澄清、无成本展示             澄清按钮 / 工具 trace / 成本展示     │
└────────────────────────────────────────────────────────────────────────┘
                    ↑ 共享同一套生产检索配置 ↑
        dense + sparse + BM25 → weighted zscore → 小节去重 → 排除习题小节
```

| 维度 | 问答 tab | Agent tab |
|---|---|---|
| 对话轮次 | 单轮，每次独立 | 多轮，`session_id` + 内存会话 |
| 历史注入 | 无 | 摘要 + 最近 4 轮 |
| 检索 | `DEFAULT_QA_CONFIG`，取 top 8 构造上下文 | `AGENT_RETRIEVAL_CONFIG`，每个 query top 8，最多 3 个 query，合并取前 10 |
| 生成 | 流式 `stream_answer` / `qa_messages` | 非流式，按 `quiz/exam/qa` 三套 prompt |
| 工具 | 无 | 9 个工具，Pydantic 契约 + 独立超时 |
| 澄清 | 无 | 两类规则触发，clarify 短路 |
| 成本 | 不展示 | `tokens` + `cost_cny` |
| 可信度 | 引用 expander + 推荐题参考答案折叠 | 引用来源 expander + verify 真题编号/押题校验 |
| 适用 | 快速定义/概念问答，token 最省 | 追问、选择题、考情、计算、图表、学习路径 |

### 3.2 共享检索链路（按代码实际）

生产配置（`generation/service.py::DEFAULT_QA_CONFIG` 与 `agent/tools.py::AGENT_RETRIEVAL_CONFIG` 基本一致）：

1. **三路召回**
   - dense：BGE-M3 embedding，向量检索；
   - sparse：BGE-M3 sparse 权重；
   - BM25：jieba 分词 + Okapi BM25。
2. **融合**
   - 生产用 `weighted` + `zscore` 归一化（每个分路分数先 zscore，再过 sigmoid 压到 0–1），不是 RRF；
   - 代码仍保留 RRF（`rrf_k` 默认 60，部分实验用 20）作为可切换项。
3. **小节去重**
   - `section_dedup=True`，`section_per=1`：同一小节只保留最高分 chunk，避免同一节多个 chunk 挤占 top-k。
4. **过滤**
   - 生产：`exclude_exercise=True`，排除「本节习题精选」「答案与解析」这类逐字收录真题的小节；`exclude_explanations=False`，**保留真题解析**，因为「这道题为什么选 B」需要它。
   - 评测：`exclude_explanations=True`，避免解析里的答案/原文造成泄漏。
5. **重排**
   - `use_rerank=False`。报告结论是负收益，生产关闭，保留可复现开关。
6. **图谱路**
   - `use_graph=False`。图谱只作为 Agent 的按需工具 `search_knowledge_graph`，不参与通用召回。

Agent 的检索还有两点差异：

- `retrieve` 最多用 3 个 query（原始问题 + 规则关键词/改写），每个 `top_k=8`，按 chunk_id 合并去重后取前 10；
- 计算题会把配置 `replace(cfg, use_bm25=False)`，因为代码注释记录 BM25 对概念/图表题 +1~4pp，但对计算题略负。

### 3.3 各自独有的能力

**问答 tab 独有：**
- 流式输出，首字延迟有明确提示（代码注释记录 15–45 秒）；
- 引用来源先渲染，再生成回答；未配置 LLM key 时仍可看检索结果；
- 推荐真题 + 参考答案折叠（先自测再看）。

**Agent tab 独有：**
- 多轮会话与指代消解；
- 题型路由与工具编排；
- 澄清 HITL（可点击选项）；
- 选择题模式（A/B/C/D 选项）；
- 工具 trace、路由、轮次、`stop_reason`、成本展示；
- 真题编号/押题校验与一次重写；
- 画像驱动的推荐/规划（侧栏 `client_id`）。

---

## 4. Agent 设计

### 4.1 图结构

```text
route ──(needs_clarification)──→ clarify → END
  │
  └──→ retrieve → assess ──(should_rewrite)──→ rewrite → retrieve
                     │
                     └──(足够/走工具)──→ plan ⇄ act ⇄ observe
                                              │
                                              └──(终态/预算)──→ answer → verify
                                                                     │
                                                       (校验失败一次) └→ answer → END
```

对应代码：`agent/graph.py::build_graph`；节点实现：`agent/nodes.py`。

### 4.2 节点职责与路由

| 节点 | 做什么 | 关键分支/规则 |
|---|---|---|
| `route` | 判题型，初始化 `attempts`，做澄清检查 | `EXAM_HINTS` 优先判 exam；否则按 qid 的题型 `figure/calc/cross_subject`；再按 `FIGURE_HINTS`、`RE_CALC_ASK`；默认 concept |
| `clarify` | 生成澄清话术，直接终态 | `stop_reason="clarification"`；不检索、不调 LLM |
| `retrieve` | 多 query 检索、合并去重 | 最多 3 个 query；`top_k=8`；合并后取前 10；计算题关 BM25 |
| `assess` | 判断证据是否足够 | `sufficient = len(contexts) >= 3 and (covered >= 1 or len(contexts) >= 6)`；不足且 `attempts < 2` 且非工具题 → rewrite |
| `rewrite` | 规则改写 query | 只在检索不足时；不调 LLM |
| `plan` | 决定下一个工具 | 预算/轮次检查；规则优先；规则无法决定才 LLM planner（最多 1 次，排除 `verify_answer`） |
| `act` | 执行**一个**工具 | `tools.guard.begin_round(round_no)`；结果写 `tool_outputs[OUTPUT_KEYS[action]]` |
| `observe` | 压缩观察结果并判终态/下一步 | summary 300 字、snippet 600 字；预算 → budget；有 next_plan 且已 3 轮 → max_rounds；有证据/成功 → answered；有失败 → tool_error；否则 no_evidence |
| `answer` | 生成最终回答 | 无 blocks 且无 evidence → no_evidence；无 LLM → tool_error；quiz/exam/qa 三套 prompt；verify feedback 时重写一次；空正文 → no_evidence |
| `verify` | 真题编号/押题校验 | `check_citations` 计算 citation precision/recall；`detect_predictions` + `extract_exam_refs` 决定 `ok`（无效真题编号或押题即失败）；失败且未重试 → `verify_retry` |

`route` 的优先级有一个细节：**先看 `EXAM_HINTS`，再看 qid 题型**。所以「2021-15 这题考过几次」会走 exam，而不是因为 qid 是 figure 就走看图。

### 4.3 有界 plan-act-observe

约束来自 `agent/state.py` 和 `agent/nodes.py`：

- `MAX_TOOL_ROUNDS = 3`；
- `DEFAULT_BUDGET_S = 90.0`；
- 每次 `act` 只执行一个工具；`observe` 必须给出终态或明确的 `next_plan`；
- `plan` 每次先检查预算，再检查 `round_no > MAX_TOOL_ROUNDS`；
- 规则 planner 的决策：
  - `calc` → `_plan_calc`（计算器，必要时先提取表达式）；
  - `figure` → `read_figure`；
  - `concept` / `cross` → `search_knowledge_graph`；
  - `exam` → 按意图顺序 `get_exam_stats` → `get_learning_path` → `get_related_kps` → `get_kp_questions`；
- `_llm_plan` 只在规则无法决定时调用，`MAX_PLANNER_LLM_CALLS = 1`，输出必须能通过 Pydantic 参数校验；`verify_answer` 不在 planner 可选工具里。

**为什么这样设计：** 代码注释记录本环境 LLM 长输出约 12 token/s、单次完整问答 50s 以上；自由 ReAct 的多轮会把延迟放大到不可接受。规则优先能把概念题压到「检索 1 轮 + 1 次生成」，把计算/图表/考情题变成可预测的工具序列。

### 4.4 7 种 stop_reason

| stop_reason | 含义 | 实际触发条件 |
|---|---|---|
| `clarification` | 信息不足，反问用户 | `route` 的 `_clarify_check` 设置 `needs_clarification`；`_after_route` 走 `clarify`；`clarify` 节点写终态并返回选项。**不检索、不调 LLM** |
| `answered` | 正常给出回答 | `observe`：没有 `next_plan`，且 `contexts` 非空或任一工具成功，写 `answered`；`AgentRunner` 兜底：没有显式 stop_reason 但有 answer 时也记 `answered` |
| `no_evidence` | 没有可用证据，不硬答 | `observe`：没有 `next_plan`、没有 contexts、没有成功工具；`answer`：无 blocks 且无 evidence；`answer`：模型返回空正文；`AgentRunner` 兜底：无 answer 时记 `no_evidence` |
| `tool_error` | 工具/LLM 失败 | `observe`：没有 `next_plan`、没有 contexts、没有成功工具、至少一个失败；`answer`：LLM 不可用；`AgentRunner`：图执行异常 |
| `budget` | 超 90 秒预算 | `plan` 或 `observe` 检查 `elapsed > DEFAULT_BUDGET_S` |
| `max_rounds` | 超过 3 轮工具上限 | `plan` 看到 `round_no > MAX_TOOL_ROUNDS`；或 `observe` 看到还有 `next_plan` 但 `rounds >= 3` |
| `verify_retry` | 真题编号/押题校验失败，触发一次重写 | `verify`：`invalid_exam_refs` 或 `predictions` 非空、`verify_retries < 1`、answer 非空、无 LLM error；写 feedback 并 `verify_retries += 1`。该标记会保留在最终结果里，表示发生过一次重写 |

**评测覆盖情况：** 19 轮多轮评测只实际出现 `answered` 15 次、`clarification` 4 次；其余 5 种在单测里覆盖（`tests/` 中有对应断言），但**没有线上/评测集分布数据**，面试时不要编造比例，直接说「其余终态有单测、缺评测分布，待补」。

### 4.5 工具契约与超时

9 个工具（`agent/contracts.py::TOOL_SPECS`）：

| 工具 | 用途 | 超时 | 重试 |
|---|---|---|---|
| `search_knowledge_base` | 三路检索 | 30s | 默认 0 |
| `search_knowledge_graph` | 图谱/前置/关联查询 | 15s | 0 |
| `read_figure` | 看图（T1 标注 / T2 VLM / T3 位图降级） | 15s | 0 |
| `calculate` | simpleeval 确定性计算 | 5s | 1 |
| `get_exam_stats` | 考频/年份统计 | 15s | 0 |
| `get_kp_questions` | 真题推荐（评测模式排除当前题/答案） | 15s | 0 |
| `get_related_kps` | 共现/关联考点 | 15s | 0 |
| `get_learning_path` | 前置/学习顺序 | 15s | 0 |
| `verify_answer` | 引用校验工具的契约（当前流程未直接调用，`verify` 节点复用同一套函数） | 10s | 0 |

契约细节：

- 每个工具一个 Pydantic 参数模型，`extra="forbid"`，坏参数返回 `error_type="validation"`，不会静默吞掉；
- 工具白名单是唯一入口，planner 和 LLM fallback 都只能从 `TOOL_SPECS` 选；
- `ToolResult` 带 `ok/error_type/error/data/summary/took_ms/attempt/round/args_digest`；trace 不落原始参数，只存短哈希；
- 失败不抛异常，Agent 降级为「其余工具可用」，但失败会计入 `tool_failures`，避免空结果被误读成「工具没用」；
- `ToolGuard`：单轮最多 6 次、每分钟最多 60 次；超时线程不会被强杀（Python 线程限制），只保证主流程不阻塞。

### 4.6 澄清 HITL

`_clarify_check` 只有两类触发（这是实际实现，不要扩大成「所有选择题都澄清」）：

1. **指代题但缺上下文：** 问句包含 `这题 / 这道题 / 选什么 / 下图 / 如图` 之一，且没有 `qid`，且四个选项都为空 → 返回两个选项：
   - 「告诉我真题编号（如 2009-04），我可以解析配图并作答」
   - 「直接把 A/B/C/D 四个选项的文字发给我」
2. **考点类问题但定位不到考点：** `route == "exam"` 或问句含「考点/知识点」，且没有 qid，且 `extract_topic` 返回 None → 返回 top 5 高频标签 + 「我换个说法，把考点名称写全」。

`clarify` 是终态：
- 不调用 `search_knowledge_base`；
- 不调用 LLM；
- 直接把结构化选项返回给 UI，UI 渲染成按钮；用户点击后把选项文本作为下一轮问题发回。

多轮评测中澄清召回率 1.0（4 个澄清用例全部召回），但**澄清准确率/误澄清率没有单独指标，待补**。

### 4.7 成本追踪

- `generation/llm.py::LLMClient` 记录每次调用的 `prompt_tokens` / `completion_tokens`；默认 `temperature=0.0`、`timeout=60s`、`max_retries=3`；主模型返回空时可切 backup 模型。
- `agent/cost.py::compute_cost_cny(model, prompt_tokens, completion_tokens)` 按 `PRICE_PER_1K` 人民币单价表计算，保留 6 位小数；未知模型用默认 `0.002 元/1K prompt + 0.008 元/1K completion`。
- `AgentRunner.run` 在结束时把节点累计的 tokens 转成 `cost_cny`，写进 `AgentResult`；UI 在每轮回答下方展示。
- **口径限制（面试要主动说）：** 当前 `.env` 用 `agnes-3.0-flash`，不在单价表里，因此成本走默认价，是**估算值/下界**，不是供应商账单；另外 `_propose_expression` 的 LLM 调用 token 没有汇入 state，严格说成本会略低。多轮评测的 ¥0.1744 应理解为「按默认单价的估算」。

---

## 5. 关键技术取舍表

| 取舍 | 选了什么 | 放弃了什么 | 为什么 |
|---|---|---|---|
| Agent 控制流 | 规则优先的**有界 plan-act-observe** + LangGraph StateGraph | 自由 ReAct / 无限工具循环 | LLM 长输出约 12 token/s、单次问答 50s+；自由 ReAct 延迟不可控、不可复现。有界循环让「概念题 1 轮检索」「计算题按规则调工具」可预测 |
| Agent 数量 | 单 Agent + 确定性工具 + 规则 Critic | 多 Agent 协作/辩论 | 多 Agent 的延迟、成本、调试复杂度都更高；本任务工具是确定性的，单 Agent 足够。对抗规划里的 Planner/Critic/Expert 是**顺序角色**，不是自主多 Agent 对话 |
| 模型适配 | 不改模型，靠检索/工具/prompt | 微调 / LoRA / 继续预训练 | 有 GT 的 dev 只有 452 题、holdout 99 题，微调验证噪声大、易泄漏；当前瓶颈在检索与工具编排，不在模型风格 |
| 会话记忆 | 进程内 LRU(200) + TTL 2h | SQLite / Redis / LangGraph checkpointer / 向量长期记忆 | 明确不做「向量库持久化会话记忆」；内存实现简单、无隐私落盘、重启即失是可接受的代价。长期偏好另走 profile 的 SQLite 事件表 |
| 指代消解 | 纯规则（代词表 + last_qid/last_topic + 年份正则） | LLM 改写 / 通用指代模型 | 规则零成本、零延迟、可解释；无法消解返回 None，交给澄清，不硬猜 |
| 表达式求值 | `simpleeval` + 字符白名单 + 禁用词 + 幂上限 | 自研 AST / `eval()` | 安全边界清晰；包不可用时 fail closed（返回 `ok=False`），不偷偷降级 |
| 检索融合 | weighted zscore（sigmoid 归一化） | RRF 作为生产默认 | chunk 级 RRF 更高（R@10 0.5161 vs 0.5042），但 zscore 的 MRR 更高（0.3443 vs 0.3311），且 zscore+节去重题级 Acc 0.6327，高于 RRF 基线（未去重）0.5973；生产要题级准确率，所以选 zscore，RRF 保留可切换 |
| 重排 | 关闭 rerank | bge-reranker-v2-m3 | dev Recall@10 −7.2pp、MRR −0.0425、P50 +246ms；holdout 也没有净收益。不是「rerank 无用」，而是这个模型/任务不匹配 |
| 图谱 | 不进入通用召回，只做按需工具 | 图谱路常开 | 图谱对概念/计算/图表题都是负贡献；长尾救援只有 2.5%（dev）/3.6%（holdout） |
| 小节去重 | 开启 `section_dedup=True, section_per=1` | 保留同节多 chunk | 同一小节多个 chunk 会挤占 top-k；去重牺牲 chunk-level recall，但提升 question-level Acc@10（dev 0.5863 → 0.6327） |
| 生成温度 | `temperature=0.0` | 创造性采样 | 考研答案要稳定、可复现；低温度减少随机发挥，但仍需真题编号/押题校验兜底 |

---

## 6. 量化结果

### 6.1 检索消融（dev 452 题）

来源：`reports/ablation_dev.md`。`PROD` = dense+sparse+bm25 | no-expl | no-ex | sec-dedup1 | weighted。

| 配置 | Recall@10 | Acc@10 | MRR@10 | nDCG@10 | P50 |
|---|---:|---:|---:|---:|---:|
| S0 dense | 0.4731 | 0.5553 | 0.3308 | 0.3423 | 51ms |
| S1 hybrid（dense+sparse+graph） | 0.5161 | 0.5973 | 0.3311 | 0.3569 | 82ms |
| S1b + BM25 | 0.5104 | 0.5996 | 0.3347 | 0.3549 | 85ms |
| S2 + rerank | 0.4385 | 0.5354 | 0.2923 | 0.3079 | 331ms |
| **PROD** | **0.3822** | **0.6681** | **0.3708** | **0.2724** | **50ms** |

> 口径提醒：S1/S1b 还开着图谱路，PROD 关闭图谱、加了节去重和过滤，所以 S0→PROD 不是严格单变量递增；严格的单组件结论看报告里的「各组件净贡献」和 rerank 固定候选集实验。

关键结论：

- rerank 相对 S1b：**Recall@10 −0.0720、MRR −0.0425、nDCG −0.0471、P50 +246ms**；
- PROD 的 Recall@10 比 S1 低，不是「检索变差」，而是**小节去重后同一小节只留 1 个 chunk**，chunk-level recall 必然下降，但 question-level Acc@10 提升：zscore 基线 0.5863 → zscore+节去重 0.6327，完整 PROD（再加 BM25/过滤）0.6681；
- 图谱在 163 个失败题里只救回 4 题（2.5%），所以不作为默认路。

### 6.2 检索消融（holdout 99 题）

来源：`reports/ablation_holdout.md`。

| 配置 | Recall@10 | Acc@10 | MRR@10 | P50 |
|---|---:|---:|---:|---:|
| S0 dense | 0.5017 | 0.6162 | 0.3561 | 46ms |
| S1 hybrid | 0.5402 | 0.6566 | 0.2979 | 86ms |
| S1b + BM25 | 0.5330 | 0.6465 | 0.2989 | 86ms |
| S2 + rerank | 0.4876 | 0.5960 | 0.3286 | 428ms |
| **PROD** | **0.3401** | **0.6869** | **0.3673** | **54ms** |

- rerank 相对 S1b：Recall@10 −0.0454、MRR +0.0297、P50 +342ms；仍不进入生产；
- 图谱救援：28 个失败题中 1 题（3.6%）。

### 6.3 Agent 三路对照（dev 抽样 21 题）

来源：`reports/agent_eval.md`。

| 指标 | 纯 LLM | 单轮 RAG | Agent |
|---|---:|---:|---:|
| Accuracy | 90.5% | 85.7% | **95.2%** |
| 95% bootstrap CI | 76.2–100% | 66.7–100% | 85.7–100% |
| 空回答 | 0 | 0 | 0 |
| P50 | 3.0s | 5.1s | 5.6s |
| P95 | 19.6s | 53.5s | 23.0s |
| 平均 token | 309 | 4048 | 2127 |
| 平均步数 | — | — | 6.0 |
| 平均工具数 | — | — | 2.0 |

分层与配对：

- 概念题：纯 LLM 86.7%、单轮 93.3%、Agent 100%；
- 计算题：纯 LLM 100%、单轮 66.7%、Agent 83.3%（**Agent 在计算题上反而低于纯 LLM，这是短板**）；
- 配对 21 题：Agent 相对单轮「改对 2 题、改错 0 题」（2016-15、2011-22）；
- 差值 CI：Agent − 单轮 +9.5pp [0.0, +23.8]（**下界刚好 0，不能宣称显著**）；单轮 − 纯 LLM −4.8pp [−23.8, +9.5]；Agent − 纯 LLM +4.8pp [−9.5, +19.1]。

### 6.4 多轮对话评测（19 轮）

来源：`data/eval/multi_turn_report.json`。

| 指标 | 值 |
|---|---:|
| 轮次数 | 19 |
| pass_rate | 0.947（18/19） |
| 澄清召回率 | 1.0（4/4） |
| 指代消解率 | 1.0 |
| 工具失败率 | 0.03（1/33 = 3.03%） |
| 平均工具轮次 | 0.95（含 4 个澄清的 0 轮） |
| 总成本 | ¥0.1744 |
| 平均成本 | ¥0.00918/轮（0.92 分） |
| 单轮最高成本 | ¥0.018672 |
| route 分布 | exam 10 / concept 5 / figure 2 / calc 2 |
| stop_reason 分布 | answered 15 / clarification 4 |

唯一失败：`mt-12` 的第 2 轮，期望 `exam`、实际 `concept`。可讲的解释：规则路由对「我想问的是『死锁』」这类没有明显考情词、也没有 qid 的追问，会默认 concept；这属于规则路由的边界，不是检索失败。**修复方向是补追问意图特征，而不是加一个自由 planner。**

### 6.5 引用与声明评测

来源：`reports/claim_eval.md`。

| 指标 | 值 |
|---|---:|
| 可溯源率 | 100%（97/97） |
| 本考点引用率 | 95.9%（93/97） |
| 编造真题编号 | 0 |
| 押题/预测表述 | 0 |
| 考频陈述准确率 | 83.3%（6 条） |
| 逐条通过 | 20/20 |

### 6.6 对抗规划评测

来源：`reports/adversarial_eval.md`。

| 指标 | 值 |
|---|---:|
| 黄金 plan 误报 | 0/10（0.0%） |
| 注入检出 | 40/40（100%），6 类各 100% |
| 挑战集命中 | 40/48，精确率 83.3% |
| 严重度分布 | high 35 / medium 13 |
| 单轮 → 2 轮 | 收敛 0% → 100% |
| 版本数 | 1 → 3 |
| 残留 high | 2 → 0 |
| LLM 调用 | 2 → 4 |
| token | 4657 → 6925 |
| P50 | 26.9s → 103.3s |
| stop_reason | max_rounds → converged |

**注意：** 只有 1 组「单轮 vs 2 轮」对照，不能据此说 2 轮普遍更好；但能说明「规则 Critic + 一次修订」在注入场景下确实提升了收敛率，代价是延迟和 token 约 2 倍。

### 6.7 60 题 held-out 端到端（记录在 optimization_log）

来源：`docs/optimization_log.md`（不是 `reports/` 下的独立报告）。

| 指标 | 值 |
|---|---:|
| 端到端准确率 | 78.3% |
| 可判题准确率 | 90.4%（52 题可判） |
| 空回答 | 8/60（13.3%） |
| P50 | 15.8s |
| 数据泄漏 | 0 |

分科目：操作系统 100%、组成原理 86.7%、数据结构 73.3%、计算机网络 53.3%。
分年份：2023 73.3%、2024 79.2%、2025 81.0%。

**口径提醒：** 这组数字与 `reports/agent_eval.md` 的 21 题抽样不是同一批；且没有在 holdout 上跑纯 LLM 对照。面试时可以说「我们做过 60 题端到端」，但不能说「RAG 比纯 LLM 高 X」。

### 6.8 测试

| 项 | 值 |
|---|---:|
| 本地全量 `pytest -q` | **223 passed**（2026-11-05 实跑） |
| CI | 只跑 `tests/test_adversarial_unit.py`，**10 passed** |
| README 旧数字 | 135 passed（已过时，见「偏差」） |

### 6.9 明确「待补」的数字

- holdout 上的纯 LLM / 单轮 RAG / Agent 端到端准确率、成本、P50/P95；
- 7 种 `stop_reason` 的真实分布（目前只有 answered / clarification 有评测数据）；
- 澄清的**误澄清率/精确率**（只有召回率 1.0）；
- 并发吞吐、QPS、连接数、内存占用；
- 真实供应商账单成本（当前按默认单价估算）；
- 用户满意度、留存、真实学习提升；
- 规则指代消解的覆盖面（只有 19 轮里的 1.0，样本小）；
- 计算题 Agent 低于纯 LLM 的原因分析（缺分题型错误样本）。

---

## 7. 可深挖的技术细节

### 7.1 会话记忆的压缩策略

代码：`agent/memory.py`。

常量：

```text
MAX_SESSIONS = 200
SESSION_TTL_S = 2 * 3600
MAX_TURNS = 8
RECENT_TURNS = 4
ANSWER_CLIP = 300
SUMMARY_MAX_CHARS = 800
```

机制：

1. `get_or_create` 时先 `_prune_locked`：删除 `now - updated_at > 2h` 的会话；超过 200 个时按 LRU 逐出最久未用；
2. 每轮 `append_turn` 后，如果 `len(turns) > 8`，就循环 `_compress_oldest`，把最早一轮弹出；
3. `_compress_oldest` 是**规则压缩，不调 LLM**：
   - 回答去换行、截前 60 字；
   - 生成一行 `· 问过「问题前30字」（route）→ 回答前60字`；
   - 追加到 `summary`；`summary` 超过 800 字时按行从头部丢弃，保留最近摘要；
4. `render_history()` 只返回 `summary` + `turns[-4:]`，不是全部 8 轮；
5. `Turn.question` 存**原始问句**，不是改写后的 query；`AgentRunner` 回写 `answer[:600]`，而 `Turn.render()` 再截到 300 字。

为什么这样设计：

- 控制 prompt token，避免长会话线性膨胀；
- 规则摘要零成本、可复现，不需要额外 LLM 调用；
- 只保留「问过什么 + 结论开头」，足够支撑「那 2021 年呢」这类追问的 topic 延续；
- 代价是细节丢失：如果用户追问很早之前的某个细节，摘要里可能没有。

面试可举的例子：

> 用户连续问 10 轮，前 2 轮会被压成两行摘要；prompt 里看到的是「更早的对话摘要 + 最近 4 轮」，而不是 10 轮全文。这样第 10 轮的 token 不会随轮数线性增长。

### 7.2 指代消解的规则逻辑

代码：`agent/followup.py`。

按顺序匹配 4 类：

1. **年份追问**：正则 `那 2021 年呢？` / `2021 年呢？`；要求 `session.last_topic` 存在，否则返回 None；改写为「`{last_topic}在2021年考过几次？分别是哪些真题？`」。
2. **学习路径追问**：问句含 `先学/怎么学/学习路径/学习顺序/前置`，且 `last_topic` 存在，且问句里没写 topic；改写为「`{last_topic}{原问句}`」。
3. **考点指代**：`这个考点/该考点/这个知识点/该知识点/这一章`，替换成 `last_topic`；替换后与原句不同才返回。
4. **题目指代**：`last_qid` 存在，且问句含 `这题/这道题/该题/此题/上述题/刚那题/刚才那题`，或含「它」且问句长度 ≤ 12；返回原问句 + `qid=last_qid`（**不改写问题文本**，让下游按 qid 取题）。

返回 `None` 的语义：不需要消解，或无法消解。`AgentRunner` 不会强行改写；如果后续 `clarify` 检测到「这题/选什么」且仍无 qid/选项，就反问。

**天花板：** 规则只覆盖固定代词和年份/学习路径模式；「它为什么错」「上面那个算法」等未覆盖表达会漏；多轮里 topic 指代也可能指错。19 轮评测 1.0 只是小样本，不能外推。

### 7.3 检索融合与 rerank 参数

代码：`retrieval/pipeline.py`、`retrieval/fusion.py`。

- 三路：dense、sparse、BM25；`use_graph=False`；
- `fusion="weighted"`，`normalize="zscore"`；RRF 的 `rrf_k` 默认 60；
- `section_dedup=True`，`section_per=1`；
- `exclude_exercise=True`，`exclude_explanations=False`（生产）；
- `rerank_top_n`：QA 10，Agent 8；rerank 关闭时它相当于最终返回条数；
- QA 取 `top_hits(8)` 构造上下文；Agent 每个 query 取 8，合并后取前 10；
- 计算题 `use_bm25=False`。

为什么 weighted zscore 而不是 RRF：

- 不是所有指标都赢：chunk-level R@10 上 RRF 0.5161 高于 zscore 0.5042；
- 但 zscore 的 MRR 更高（0.3443 vs 0.3311），且 zscore + 节去重的题级 Acc@10 是 0.6327，RRF 基线（未去重）是 0.5973；生产目标是「这道题答对」，不是 chunk 召回；
- zscore 经 sigmoid 压到 0–1，避免某一分路分数尺度压倒其他路；RRF 只看排名、不看分数差距，在「dense 很自信、BM25 勉强命中」时区分度不够。

rerank 的负结果怎么讲：

> 我们不是拍脑袋关的。`reports/ablation_rerank.md` 做了三组对照：默认混合、固定候选集纯重排、α 分数融合扫描。固定候选集重排时 MRR 从 0.2938 降到 0.2789；α=0.2 时 Recall 虽然 +0.4pp，但 MRR/nDCG 都降。结论是 bge-reranker-v2-m3 和 408 教材/真题的分布不匹配，生产关闭，保留 S2 可复现。

### 7.4 引用来源如何做到可展开核对

三层：

1. **构造上下文时编号**：`generation/prompt.py::ContextBlock` 给每个 chunk 一个 `[n]`，带 `citation`、`subject`、原文；prompt 要求「只引用资料中真实存在的真题编号 YYYY#NN」，禁止押题。
2. **生成后校验**：`generation/citation.py`
   - `check_citations`：抽取回答里的 `[n]`，计算 citation precision/recall，并记录无效编号；**当前 `verify` 的 `ok` 不读取这个结果**，所以 `[n]` 越界不会单独触发重写（这是实际偏差）；
   - `detect_predictions`：检测「押题/预测/今年必考」等表述；
   - `extract_exam_refs`：抽取 `YYYY#NN` 和「YYYY 年第 N 题」，和 `valid_exam_refs` 白名单比对；
   - `verify` 的 `ok = not predictions and not invalid_exam_refs`；失败且未重试 → `verify_retry`，answer 消费 feedback 重写一次；第二次仍失败就 END，不无限重试。
3. **UI 展开**：
   - 问答 tab 先渲染「引用来源」，每个 `expander` 标题是 `[n] citation`，展开看原文；有教材插图时显示图片；
   - Agent tab 每轮展示引用来源、工具 trace、路由、`stop_reason`、成本；推荐真题的答案折叠，先自测再看。

**边界：** 运行时硬拦截能保证「真题编号真实、没有押题」；`[n]` 的 precision/recall 目前只记录、不阻断。两者都不能保证每一句自然语言陈述都被引用覆盖；claim_eval 的 100% 是离线评测的「可溯源率」，不是「所有事实都正确」。面试官追问时主动承认这一点。

### 7.5 画像标签如何加权到规划

两套机制不要混：

- **会话记忆**：内存 LRU/TTL，重启即失，只服务当前对话；
- **用户画像**：SQLite 事件表（`profile/store.py`），问题事件 + 规划事件（以及历史遗留的 quiz 事件），重启不丢，跨会话。

派生逻辑（`profile/stats.py`）：

- 不从画像表读「结论」，而是从事件重算（单一事实来源）；
- 聚合出 `weak_subjects`、`subject_counts`、`top_tags`、`recent_questions`、`exam_focus` 等；
- 前端侧栏用匿名 `client_id`，把薄弱科目预填到对抗规划表单。

加权到规划（`adversarial/planner.py` + `roi.py`）：

- `profile` 的高频考点转成 `boost_tags`，上限 `MAX_PROFILE_BOOST = 8`；
- `roi.select_high_value` 把这些 tag 设为 `mandatory`，排序 key 里 `tag in boost` 优先；
- `allocate_subject_hours` 对每个 boost tag 给对应科目 **+0.04 的时长占比（归一化前）**；薄弱科目另有 +0.08、进度「未开始/一轮中」+0.05、「二轮完」−0.05；
- 最后把每科占比夹在 5%–60%，归一化；薄弱科目有 30% 下限规则，避免被高频考点挤掉。

**可讲的取舍：** 这是可解释的启发式，不是学习出来的排序模型；好处是用户能看懂「为什么给我排这些考点」，坏处是权重（0.04/0.08）没有经过线上 A/B，只有黄金 plan 误报 0/10 的规则级验证。

---

## 8. 已知短板与升级路径

| 短板 | 证据 | 影响 | 升级路径 |
|---|---|---|---|
| **缺 holdout 端到端 baseline** | `reports/agent_eval.md` 只有 dev 21 题；60 题 held-out 只有 Agent 数字（78.3%），没有纯 LLM / 单轮 RAG 对照 | 无法证明 RAG/Agent 在未见年份上的净增益；21 题 CI 下界为 0 | 在 99 题 holdout 上跑三路同 prompt 对照，报告准确率、成本、P50/P95 和 bootstrap CI |
| **规则指代消解有天花板** | `followup.py` 只覆盖 4 类固定模式；19 轮 1.0 是小样本 | 复杂指代会漏，退回澄清或答错 | 先补模式（「它为什么错」「上面那个」），再考虑小模型分类器；用多轮集做覆盖率/错误率 |
| **内存记忆重启即失** | `memory.py` 明确不落盘 | 单机重启、换实例就丢历史；多副本不一致 | Redis 按 session TTL + LRU；或 LangGraph checkpointer；需先定义隐私/保留策略 |
| **单机串行、无并发数据** | `AgentRunner` 用 `Lock` 串行化；`ToolGuard` 是进程级窗口 | 多用户同时提问会排队；分钟级限流不能按会话隔离 | 工具实例 per-run / 池化；会话外置；补压测（QPS、P50/P95、错误率） |
| **7 种终态只覆盖 2 种** | 多轮评测只有 answered 15 / clarification 4 | budget/max_rounds/tool_error/no_evidence/verify_retry 没有真实分布 | 构造故障注入评测：超时、空检索、LLM 不可用、强制超轮次 |
| **引用编号未纳入硬拦截** | `verify` 的 `ok` 只看 predictions 和 invalid exam refs，`check_citations` 只记录 | `[n]` 越界不会触发重写；claim_eval 100% 是离线结果 | 把 citation precision/recall 阈值纳入 verify feedback，低分重写一次 |
| **计算题 Agent 低于纯 LLM** | 21 题：Agent 83.3% vs 纯 LLM 100% | 计算题是 Agent 的弱项，工具链没有稳定超过模型心算 | 分析 1 道错题：是表达式抽取错、单位错还是计算器边界；补 expression prompt 与单测 |
| **图表 T3 仍是位图降级** | `read_figure` 分 T1/T2/T3；T3 没有 VLM | 低质量图/无标注图答不好 | 接 VLM 或人工标注；按题型分层评测 |
| **空回答仍有 13.3%** | 60 题 held-out：8/60 空回答 | 空回答被单独计数，不占总准确率，但用户体验差 | flash 模型 4k 输出上限 + 上下文预算；重试/切换 backup；空回答单独监控 |
| **CI 只跑 10 个测试** | `.github/workflows/ci.yml` 只跑 `test_adversarial_unit.py` | 检索/Agent/记忆的回归没有 CI 保护 | 把 223 个测试分层：纯单测进 CI，集成/LLM 测试夜间跑 |
| **README 旧数字/口径** | README 写 135 passed；检索 P50 汇总写 35–49ms，报告是 50/54ms | 新人/面试官看到的文档和代码不一致 | 更新 README；以本文「偏差」节为准 |

### Week 4 收尾项的真实状态

| 项 | 状态 | 证据 |
|---|---|---|
| 60 题全量评测 | **已有记录**：78.3%（可判 90.4%，空回答 13.3%） | `docs/optimization_log.md`；但不在 `reports/`，且无 baseline 对照 |
| 3 个 demo | **未完成**：仓库只有 `docs/demo.png` 和 Streamlit 应用，没有 3 个独立 demo 资产 | 目录检查 |
| 文档 | **基本完成**：README + optimization_log + resume/interview；本文补齐面试版 | `docs/` |
| CI | **部分完成**：只跑 10 个 adversarial 单测，不是全量 223 | `.github/workflows/ci.yml` |

---

## 9. 高频追问 Q&A（20 组）

> 回答原则：先给结论，再给代码/数据证据，最后主动说边界。不要把所有问题都答成「我们做得很好」。

### Q1. 为什么不用 LangChain / LangGraph？

**先澄清：编排层我们用了 LangGraph 的 `StateGraph`，但决策层没有用 LangChain 的 Agent/ReAct/Chain。**

- 用 LangGraph 的原因：节点、条件边、状态合并显式可见；`AgentState` 是扁平、可 JSON 序列化的 dict，便于写 `steps`/`plan_history` 和做分层评测；
- 不用 LangChain Agent 的原因：它的 AgentExecutor 默认是「模型决定下一步」的自由循环，和我们要的「规则优先 + 3 轮硬上限 + 7 种终态」相反；工具契约、超时、澄清短路都需要自己控制；
- 不用 LangGraph 的 checkpointer/长期记忆：我们明确不做持久化会话记忆，历史放在自己的 `InMemorySessionStore`，避免和框架生命周期绑定。

**要补的数据点：** 没有做「LangGraph vs 自研图」的性能对照；这是工程选择，不是实验结论。

### Q2. 3 轮上限怎么定的？为什么不是 5 轮或 10 轮？

- 代码硬约束 `MAX_TOOL_ROUNDS = 3`，总预算 90 秒；
- 依据：概念题通常 1 轮检索就够；计算题 1 次计算器；图表题 1 次看图；考情题按意图最多 3 个工具。3 轮覆盖主路径；
- 成本/延迟：本环境 LLM 长输出约 12 token/s，单次问答 50s+；更多轮次会把 P95 拉到不可接受；
- 可观测：`observe` 每轮都必须给出终态或 `next_plan`，超过 3 轮直接 `max_rounds`，不会静默继续。

**要补的数据点：** 没有「2/3/4 轮」的准确率-延迟曲线；这是拍脑袋+经验的硬边界，面试官追问时承认，并说下一步会用故障注入/分题型评测来定。

### Q3. 怎么证明回答不是幻觉？

分三层，不夸大：

1. **输入约束**：prompt 只允许基于 `ContextBlock` 回答，要求引用真实 `[n]` 和 `YYYY#NN`；禁止押题；
2. **输出校验**：`detect_predictions` 拦押题词；`extract_exam_refs` 核对真题编号白名单；失败重写一次，仍失败就结束；`check_citations` 会计算 `[n]` 的 precision/recall 并写入 `verification`，但当前不阻断；
3. **人工可核对**：UI 的引用来源可展开看原文和插图；claim_eval 可溯源率 100%（97/97）、编造编号 0、押题 0、逐条通过 20/20。

**边界：** 这证明的是「真题编号真实 + 没有押题 + 引用可溯源」，不是「每一句都事实正确」；`[n]` 越界目前只记录不拦截。要证明事实性，需要人工标注的事实性评测；目前没有，标「待补」。

### Q4. 会话记忆为什么不做持久化？

- 明确约束：不做 SQLite/Redis/LangGraph checkpointer/向量库持久化会话记忆；
- 原因：① 会话是短期的，2h 不活跃即可丢；② 避免把用户对话原文落盘带来的隐私和清理成本；③ 内存实现简单，LRU 200 + TTL 2h 足够单机 demo；
- 代价：重启/换实例就丢历史；多副本不一致；
- 区分：长期偏好走 `profile` 的 SQLite 事件表，**不是会话记忆**；两套机制独立。

**升级路径：** Redis `session_id -> turns/summary`，TTL 2h + LRU；或 checkpointer；需要先定保留期限和脱敏策略。

### Q5. 怎么评估效果？

分四层，避免「只跑一个准确率」：

1. **检索层**：dev 452 / holdout 99 的 Recall@10、Acc@10、MRR、nDCG、P50；组件消融 S0/S1/S1b/S2/PROD；
2. **Agent 层**：dev 21 题三路对照（纯 LLM / 单轮 RAG / Agent），准确率、token、P50/P95、配对改对/改错、bootstrap CI；
3. **多轮层**：19 轮，pass_rate、澄清召回、指代消解、工具失败率、route/stop_reason 分布、成本；
4. **可信度层**：claim_eval（可溯源、编造编号、押题、考频准确）、adversarial_eval（注入检出、黄金 plan 误报、收敛）。

**边界：** 样本小、CI 宽；没有用户实验；没有 holdout 端到端 baseline。这是当前评测体系最大的缺口。

### Q6. 成本多少？

- 多轮 19 轮：总 ¥0.1744，平均 ¥0.00918/轮（约 0.92 分），单轮最高 ¥0.018672；
- 21 题 Agent 平均 token 2127，单轮 RAG 4048；Agent 的 token 更低，因为它按题型少检索/少生成；
- 成本公式：`compute_cost_cny(model, prompt_tokens, completion_tokens)`，按人民币/1K token 单价表；未知模型默认 0.002 输入 + 0.008 输出；
- **口径限制**：`agnes-3.0-flash` 不在单价表，成本是估算；`_propose_expression` 的 token 未汇入 state，成本偏低；不是账单。

**要补的数据点：** 供应商实际账单、按工具/节点的成本拆解、holdout 60 题总成本。

### Q7. 如果用户问的问题知识库没有怎么办？

- 检索层：`assess` 判断证据不足，最多再 rewrite 一次（`attempts < 2`）；
- Agent 层：`observe` 没有 contexts、没有成功工具 → `no_evidence`；`answer` 无 blocks 且无 evidence → `no_evidence`；
- 生成层：prompt 要求「资料没有就说没有」，不硬编；
- UI：显示「无回答/证据不足」，而不是假装答对；
- 另一个相关终态是 `tool_error`：工具挂了但问题本身可能可答，会和 `no_evidence` 区分。

**要补的数据点：** `no_evidence` 在真实问题上的比例和用户满意度；目前只有单测，没有分布。

### Q8. 为什么不用多 Agent？

- 任务工具是确定性的（检索、统计、计算、看图），单 Agent + 工具就够；多 Agent 主要增加协调成本和延迟；
- 我们的对抗规划有 Planner/Critic/Expert 三个**顺序角色**，但不是自主多 Agent 对话：每步输入输出结构化，stop_reason 明确（converged/max_rounds/stalled/degraded/budget），可评测；
- 多 Agent 的收益需要用数据证明；当前没有「多 Agent 更好」的证据，所以不做。

**升级路径：** 如果要做，先定义「哪个子任务需要独立上下文/独立工具集」，再用 A/B 证明；不要为了架构图好看而上多 Agent。

### Q9. 为什么不做微调？

- 数据量：dev 只有 452 题有 GT，holdout 99 题；微调的验证集太小，容易过拟合/泄漏；
- 当前瓶颈：检索融合、题型路由、工具编排、空回答，不是模型「不会 408」；
- 成本：微调需要标注、训练、回归评测；收益不确定；
- 已有负结果：rerank 都因为分布不匹配变负，说明「加模型/加参数」不是默认正确。

**如果必须做：** 先做 hard negative + 领域 embedding/reranker 微调，在 dev/holdout 上同时看 Recall/MRR/成本；不要直接 SFT 生成答案。

### Q10. rerank 为什么关了？不是通常都能涨点吗？

- 不是拍脑袋：`reports/ablation_rerank.md` 做了默认混合、固定候选集纯重排、α 分数融合扫描；
- 结果：dev Recall@10 −7.2pp、MRR −0.0425、P50 +246ms；固定候选集重排 MRR 0.2938 → 0.2789；α=0.2 时 Recall +0.4pp 但 MRR/nDCG 降；
- 解释：bge-reranker-v2-m3 的训练分布和 408 教材/真题不匹配；教材行文长、术语密集，交叉编码器容易把「字面相似但考点不同」的 chunk 排前面；
- 处置：生产 `use_rerank=False`，保留 S2 可复现；未来尝试领域微调 reranker。

### Q11. 为什么计算题要关 BM25？

- `nodes.retrieve` 对 `route=="calc"` 执行 `replace(cfg, use_bm25=False)`；
- 代码注释记录：BM25 对概念/图表题 +1~4pp，但对计算题略负；
- 解释：计算题问句里数字、单位、公式符号多，BM25 的词项匹配容易被数字/单位噪声带偏；dense+sparse 更稳；
- 这是按题型自适应，不是全局关 BM25。

### Q12. 小节去重为什么会让 Recall 下降？这不是变差吗？

- 指标口径：Recall@10 是 chunk-level，同一小节被切成多个 chunk，去重后只留 1 个，chunk recall 必然降；
- 产品目标：用户要的是「答对这道题」，不是「召回同一节的 3 个 chunk」；question-level Acc@10 从 0.5863 升到 0.6327（dev，zscore 基线）；
- 另外去重让 top-k 覆盖更多不同小节，减少同一节内容重复占位；
- 所以看板要同时看 Recall 和 Acc，不能只看一个。

### Q13. 澄清会不会误触发？怎么保证不多问？

- 实际只有两类触发，条件很窄：① 指代词 + 无 qid + 无选项；② 考情/考点问题 + 抽不出 tag；
- 多轮评测澄清召回 1.0（4/4），但**没有误澄清率指标**；
- 风险：关键词式触发不是通用选择题识别——如果用户问「选什么」但没有 qid/选项，会澄清；如果用户发了选项但没有「这题/选什么」字样，可能不澄清，直接走检索；
- 这是规则实现的边界，面试时主动说；升级方向是「题型分类器 + 选项完整性校验」，而不是把所有问题都澄清。

### Q14. 指代消解具体怎么做的？消解不了怎么办？

- 纯规则，4 类：年份追问、学习路径追问、考点代词、题目代词；
- 依赖 `session.last_topic` 和 `session.last_qid`；题目代词不改写问题文本，只带 qid；
- 返回 `None` 表示「不需要或消解不了」；`AgentRunner` 不硬猜；如果 `clarify` 检测到「这题/选什么」且无 qid/选项，就反问；
- 19 轮评测消解率 1.0，但样本小；规则覆盖不到「它为什么错」「上面那个」等表达。

### Q15. 模型返回空回答怎么办？

- `answer` 节点：空正文 → `no_evidence`；
- `LLMClient`：主模型返回空时可切 backup（`.env` 配 `glm-5.2`）；
- 代码注释记录 flash 模型 4k 输出上限，`_blocks(max_chars=2400)` 限制上下文总字符，避免上下文过长导致空返回；
- 60 题 held-out 仍有 8/60 空回答（13.3%），这是待优化项；21 题 dev Agent 抽样空回答 0。

**注意：** `_blocks` docstring 写 3200，实际参数是 2400，属于文档不一致（见「偏差」）。

### Q16. 怎么保证 Agent 一定停？7 种 stop_reason 怎么落地？

- 图是显式条件边，不存在「模型想继续就继续」：`observe` 每次必须给出 `stop_reason` 或 `next_plan`；
- 3 轮硬上限 + 90 秒预算，`plan`/`observe` 都检查；
- 7 种终态：clarification、answered、no_evidence、tool_error、budget、max_rounds、verify_retry；
- verify 最多回一次：`_after_verify` 只在 `verify_feedback` 存在时回 answer，answer 消费后清空；
- 单测覆盖 7 种终态；多轮评测只出现 2 种，其余缺真实分布（待补）。

### Q17. 工具超时了怎么办？能杀掉线程吗？

- 工具跑在 `ThreadPoolExecutor`，主流程 `future.result(timeout=spec.timeout_s)`；超时返回 `error_type="timeout"`，Agent 继续走降级路径；
- **不能强杀 Python 线程**：超时线程可能继续占用资源；代码注释明确这是已知限制，只保证不阻塞 Agent；
- 重试默认 0，只有 `calculate` retries=1；慢的外部查询不自动重试，避免延迟翻倍；
- `ToolGuard` 单轮 6 次、每分钟 60 次；当前是进程级窗口，不是按会话分桶（见「偏差」）。

### Q18. 用户画像怎么影响规划？会不会变成「信息茧房」？

- 画像从事件重算：问题事件 + 规划事件；输出 `weak_subjects`、`top_tags` 等；
- 规划时 `boost_tags` 上限 8，命中 tag 的科目在时长占比上 +0.04（归一化前），并强制进入候选/排序靠前；
- 薄弱科目另有 +0.08、进度调整 +0.05/−0.05；每科夹在 5%–60%，薄弱科目 30% 下限，避免高频考点把薄弱科目挤没；
- 风险：用户一直问某个考点，boost 会持续加权；上限 8 和 30% 下限是缓解手段，但没有线上 A/B；
- 这是可解释启发式，不是学习排序；黄金 plan 误报 0/10 只验证规则级正确性。

### Q19. 单机能扛多少并发？延迟多少？

- 当前 `AgentRunner` 用 `Lock` 串行化图调用，因为 `AgentTools` 持有 trace/current_qid，不能并发共享；单进程 Agent 请求会排队；
- 检索 P50：dev 50ms、holdout 54ms；Agent 端到端 P50 5.6s、P95 23.0s（dev 21 题）；
- **没有压测/QPS/并发数据**；这是明确短板，不要编造。

**升级路径：** 工具实例 per-run 或对象池；会话外置 Redis；把 LLM 调用异步化；补 Locust/k6 压测。

### Q20. 这个项目最大的失败/负结果是什么？你从中学到了什么？

可讲三个：

1. **rerank 负收益**：直觉上「加 reranker 会涨点」，实测 dev Recall −7.2pp、MRR −0.0425、P50 +246ms；学会「先做固定候选集对照，再决定是否加模型」；
2. **图谱路负贡献**：通用召回加图谱反而降 Acc，长尾救援只有 2.5%–3.6%；学会「结构化知识不是默认越多越好，要按需调用」；
3. **Agent 计算题低于纯 LLM**：21 题 Agent 83.3% vs 纯 LLM 100%，说明工具链不是万能的；下一步要拆错误样本，而不是继续加工具。

**收获：** 评测口径比功能数量重要；负结果要写进文档；不要用「接了 Agent」当卖点，要用「有界、可解释、可评测」当卖点。

---

## 10. 与既有设定的偏差（如实记录）

以下不是「包装」，是代码实际实现与项目背景描述/README 的不一致。面试时如果被追问，主动说明比被拆穿好。

1. **编排层实际使用了 LangGraph。**
   - 背景约束是「禁止自由 ReAct」，不是「禁止 LangGraph」；代码 `agent/graph.py` 明确 `from langgraph.graph import END, StateGraph`。
   - 没有使用 LangChain 的 AgentExecutor/ReAct/Chain；决策仍是规则优先 + 有界循环。

2. **README 的测试数与部分汇总口径已过时。**
   - README 写「135 passed」，当前实跑 223 passed；
   - README 汇总表写检索 P50 35–49ms，`reports/ablation_dev.md` 是 50ms、`reports/ablation_holdout.md` 是 54ms；以报告为准；
   - CI 只跑 `tests/test_adversarial_unit.py` 10 个测试，不等于全量。

3. **`_blocks` 的 docstring 与参数不一致。**
   - docstring 提到 3200 字，实际 `max_chars=2400`；以参数为准。

4. **ToolGuard 限流是进程级，不是按会话分桶。**
   - `guardrails.py` 注释写「Week 2 接入 session 后按会话分桶」，但实现仍是进程级 `_minute_calls`；注释是计划口径，不是现状。

5. **澄清不是通用「选择题识别」，是关键词触发。**
   - 只有问句含 `这题/这道题/选什么/下图/如图` 且无 qid、无选项才触发；
   - 如果用户发了 A/B/C/D 选项但没有这些词，不会澄清；如果没有选项但也没有这些词，也不会澄清。背景里「选择题无明确题号时返回澄清」应理解为「命中这两类规则的场景」，不是所有选择题。

6. **生产保留真题解析，只排除习题小节。**
   - 背景/总结里容易说成「排除解析和习题」，实际 `exclude_explanations=False`、`exclude_exercise=True`；
   - 评测模式才 `exclude_explanations=True`，避免答案泄漏。这是有意的产品取舍。

7. **成本单价表不含当前主模型。**
   - `.env` 的 `agnes-3.0-flash` 不在 `cost.py::PRICE_PER_1K`，走默认 0.002/0.008 元/1K；
   - 因此所有 `cost_cny` 是估算值/下界，不是供应商账单。

8. **成本统计不是「每次工具/每次 LLM 调用都完整落账」。**
   - `LLMClient` 有调用级 token 统计，但 `AgentResult` 只汇总 planner/answer 的 token；
   - `_propose_expression` 的 LLM token 未汇入 state；严格说成本偏低。
   - 背景说「每次调用记录 token 与人民币成本」，实际更接近「每次 Agent 运行汇总 token 与估算成本」。

9. **`verify_answer` 在工具白名单里，但当前流程没有直接调用它；planner 也不能选它。**
   - `verify` 节点直接复用 `check_citations` / `detect_predictions` / `extract_exam_refs`，不是通过 `tools.call("verify_answer")`；
   - 它不属于 plan-act 的可选动作；LLM planner 也被排除。校验是流程强制的，不由模型决定。

10. **引用编号校验没有纳入 `verify` 的硬失败条件。**
    - `check_citations` 会计算 `[n]` 的 precision/recall 并写入 `verification`；
    - 但 `verify` 的 `ok` 实际是 `not predictions and not invalid_exam_refs`，没有读取 citation 报告；
    - 所以「`[n]` 越界」不会单独触发 `verify_retry`；claim_eval 的 100% 是离线评测结果，不是运行时拦截率。这是需要补的缺口。

11. **7 种 stop_reason 中只有 2 种有评测分布。**
    - 多轮评测实际出现 `answered` 15、`clarification` 4；
    - `no_evidence`/`tool_error`/`budget`/`max_rounds`/`verify_retry` 只有单测覆盖，缺真实分布；不要编造比例。

12. **画像持久化与「会话记忆不持久化」是两套机制。**
    - 会话记忆：内存 LRU/TTL，重启即失；
    - 画像：SQLite `data/profile/profile.db` 事件表，跨会话持久化；
    - 另外画像事件历史上包含 quiz 事件，API 仍有 `/v1/quiz/answer`，但前端已移除错题本 tab；属于遗留，不影响 Agent 主链路。

13. **对抗规划的终态是 5 种，不是 Agent 的 7 种。**
    - `adversarial/graph.py::advance` 的 stop_reason：`converged`、`max_rounds`、`stalled`、`degraded`、`budget`；
    - 运行时异常会体现在 `PlanResult.error` 字段，不应和 Agent 的 7 种 stop_reason 混为一谈。

14. **Agent v2 规划文件未在仓库中找到。**
    - 没找到独立的 `plan`/Agent v2 规划文档；设计意图可追溯至 `graph.py`/`nodes.py`/`memory.py` 的 docstring 和 `docs/optimization_log.md`；
    - 面试时不要声称「有一份完整的 v2 plan 文档」；可以说「设计约束沉淀在代码注释和优化日志里」。

15. **README 数据口径与报告存在批次差异。**
    - README 汇总写检索 P50 35–49ms；`reports/ablation_dev.md` 写 50ms、holdout 54ms；
    - 以报告为准；README 的汇总表可能是不同批次/取整。

16. **60 题 held-out 端到端评测存在，但不在 `reports/`。**
    - 数字记录在 `docs/optimization_log.md`（78.3%，可判 90.4%，空回答 8/60）；
    - 没有独立报告、没有 baseline 对照；引用时要说明来源和口径。

---

## 附：面试现场「数字口播」小抄

- 数据：720 题 / 5296 chunk / 6182 节点 / 226 标签。
- 检索：dev Recall@10 0.3822、Acc@10 0.6681、MRR 0.3708、P50 50ms；holdout Acc@10 0.6869、P50 54ms。
- rerank：Recall −7.2pp、MRR −0.0425、P50 +246ms，所以关。
- 多轮：19 轮、94.7% 通过、澄清召回 100%、指代消解 100%、工具失败 3.0%、¥0.00918/轮。
- Agent 21 题：95.2% vs 单轮 85.7% vs 纯 LLM 90.5%；CI 下界 0，不能称显著。
- 可信度：可溯源 100%（97/97）、编造编号 0、押题 0、逐条 20/20。
- 测试：223 passed；CI 10 passed。
- 负结果：rerank 负收益、图谱救援 2.5%/3.6%、计算题 Agent 83.3% vs 纯 LLM 100%。
- 最短板：holdout 端到端 baseline、并发/压测、真实成本账单。

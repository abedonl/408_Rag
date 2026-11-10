# 408-RAG 面试技术篇

> 本文只写代码和报告里能核对的内容。数字来自仓库内的评测产物，缺失项直接标「待补」，不做估算。

## 1. 定位与目标用户

**一句话**：面向 408 考研的「真题检索 + 多轮备考 Agent」，把教材、720 道历年真题（2009–2026）和 6182 个知识点节点组织成可溯源、可追问、可评测的备考系统。

目标用户是三类人：

1. 不知道自己「该背什么」的考生——需要真题频次和考点关联，而不是泛泛的重点章节；
2. 概念查不动的考生——教材厚重，需要检索到具体小节并给出引用；
3. 计算题、图表题卡住的考生——纯文本检索无解，需要确定性计算器和图形解析。

**为什么不能直接问大模型**：通用模型的参数化知识无法提供「某考点在历年真题中出现过几次、分别是哪几题」这种可核对的事实，也无法保证引用真实存在。项目里做了 LLM-only 对照组：21 道开发集题上纯 LLM 准确率 90.5%，单轮 RAG 85.7%，Agent 95.2%——单轮 RAG 并没有赢过纯 LLM，这说明「接上检索」本身不等于「答得更对」。RAG 在这里的核心价值是**可溯源、可核对、可统计**，不是无条件的准确率提升；这也是项目把引用校验、真题编号白名单、空回答率单独统计的原因。

---

## 2. 端到端链路

### 2.1 问答 Tab 与 Agent Tab 的分工

| 维度 | 问答 Tab（单轮 RAG） | Agent Tab（备考助手） |
|---|---|---|
| 入口 | `QAService.answer` / `/v1/query` | `AgentRunner.run` / `/v1/agent/query`、`/v1/agent/chat` |
| 会话记忆 | 无 | 内存 LRU + TTL，按 `session_id` 隔离 |
| 指代消解 | 无 | 规则消解「这题 / 这个考点 / 那 2021 年呢」 |
| 工具调用 | 无 | 9 个只读工具，最多 3 轮 |
| 输出 | 答案 + 引用块 | 答案 + steps + tool_calls + plan_history + verification + cost |
| 典型场景 | 快速查一个知识点 | 连续追问、选择题求解、考点统计、真题推荐 |

两个 Tab 共享同一条底层检索链路，区别只在上层编排。界面上也做了互相指路：问答页提示「需要多轮追问、选择题求解、真题推荐 → Agent」，Agent 页提示「只看单轮完整引用原文 → 问答」。

### 2.2 共享检索链路

代码主线在 [pipeline.py](file:///d:/agent/RAG/src/cs408rag/retrieval/pipeline.py)：

```text
query
  ├─ dense（BGE-M3 向量）
  ├─ sparse（BGE-M3 learned sparse）
  ├─ bm25（jieba + Okapi，K1=1.5, B=0.75）
  └─ graph（Neo4j 知识点图谱，生产默认关闭）
        │
        ▼
  融合（加权 zscore / RRF，k=60）
        │
        ▼
  小节去重（同一 section 只留 1 条）
        │
        ▼
  取 top-N →（问答侧）父节扩展 → 带编号上下文 → LLM
```

关键配置：

- 生产检索 `PROD` / `DEFAULT_QA_CONFIG`：dense + sparse + BM25，**不启用 graph、不启用 rerank**，`fusion="weighted"`、`normalize="zscore"`、`section_dedup=True`、`section_per=1`、`exclude_exercise=True`。
- 每路召回 `top_k=20`，融合后 `fusion_top_k=50`，最终返回 `rerank_top_n=10`（问答侧）。
- Agent 侧 [tools.py](file:///d:/agent/RAG/src/cs408rag/agent/tools.py) 的 `AGENT_RETRIEVAL_CONFIG` 与生产配置基本一致，但每次工具只取 `top_k=8`；计算题会临时 `use_bm25=False`（实测 BM25 对计算题略负）。
- 四路召回用线程池并行，串行约 124ms，并行后约 57ms（代码注释的实测值）。

**重排为什么没上**：交叉编码器 `bge-reranker-v2-m3` 做过完整对照，S2_rerank 相对 S1_hybrid：Recall@10 从 0.4952 降到 0.4232，MRR@10 从 0.3140 降到 0.2812，P50 延迟从 111ms 涨到 350ms。固定候选集、去章节头、扫 α 融合都复现了负收益，因此生产关闭，只保留为可复现阶段 S2。详见 [ablation_rerank.md](file:///d:/agent/RAG/reports/ablation_rerank.md)。

**父节扩展（Parent-Child）**：检索仍用子块（术语密度高），给 LLM 的上下文换成所属父级小节全文，问答侧每命中最多 1200 字，同一父节只保留一次；解析 chunk 不扩展。它只改变上下文，不改变检索指标。Agent 侧不做父节扩展，而是把工具结果压到 500 字/条、总上下文 2400 字，避免 flash 模型上下文超限返回空正文。

### 2.3 各自独有能力

- 问答 Tab：父节扩展、完整引用块可展开核对、流式输出、未配置 key 时仍可只跑检索。
- Agent Tab：多轮会话、指代消解、澄清 HITL、9 工具编排、推理链展示、成本显示、按考点推荐真题并支持卡片内作答（判分同步错题本与画像）。

---

## 3. Agent 实现

### 3.1 状态图节点与路由

图定义在 [graph.py](file:///d:/agent/RAG/src/cs408rag/agent/graph.py#L132-L174)：

```text
route ──(信息不足)──→ clarify → END
  │
  └──→ retrieve → assess ──(不足，且未达重试上限)──→ rewrite → retrieve
                       │
                       └──(足够 / 工具题型)──→ plan ⇄ act ⇄ observe
                                                   │
                                                   └──(终态)──→ answer → verify
                                                                          │
                                                        (失败一次) ────────┘→ answer → END
```

- `route`：规则路由到 `concept / calc / figure / exam / cross`。含「考过/几次/历年/关联」等词走 `exam`；真题标注为含图/计算/跨科目时优先按标注走；问句出现「下图/如图」或求值型表达时分别走 `figure` / `calc`。
- `clarify`：澄清短路，不检索、不调 LLM，直接把选项抛回用户。
- `retrieve / assess / rewrite`：检索评估与最多 2 次改写重试；计算/图表/考点题即使检索弱也继续走工具。
- `plan / act / observe`：有界循环，`observe` 每次都必须写终态或给出下一步。
- `answer / verify`：生成后用引用编号、真题编号白名单、押题表述三类检查，失败最多重写一次。

路由和张量字段都尽量扁平、可 JSON 序列化，不把 `RetrievalResult` 这类重对象塞进图状态，便于 trace 和评测复现。

### 3.2 有界 plan-act-observe 循环

- 最多 **3 个工具轮次**（`MAX_TOOL_ROUNDS = 3`）。
- 总预算 **90 秒**（`DEFAULT_BUDGET_S = 90.0`），在 `plan` 与 `observe` 各检查一次。
- `plan` 优先走纯规则 `_rule_plan`；规则无法决定时才调 LLM planner，且**最多 1 次**（`MAX_PLANNER_LLM_CALLS = 1`）。
- `observe` 用同一套规则 planner 预判「还有没有下一步」，所以观察阶段不会偷偷再花一次 LLM 调用。
- `act` 每轮只执行 `plan` 选中的**一个**工具，失败也继续，由 `observe` 决定下一步或终态。

和自由 ReAct 的差别：这里没有「模型想调几次就调几次」的循环，轮次、预算、planner 调用次数都是硬上限；循环出口必须落到 7 种 `stop_reason` 之一。

### 3.3 7 种 stop_reason

| stop_reason | 含义 | 触发条件（代码位置） |
|---|---|---|
| `answered` | 正常作答完成 | `observe` 预判无下一步且有上下文或至少一个成功工具结果；或图外兜底时 `answer` 非空 |
| `no_evidence` | 没有可用证据 | `answer` 发现既无资料也无工具结论；或模型返回空正文；或图外兜底时 `answer` 为空 |
| `clarification` | 需要用户补充信息 | `route` 命中澄清规则，`clarify` 节点直接返回选项 |
| `max_rounds` | 达到工具轮次上限 | `plan` 发现 `round_no > 3`；或 `observe` 认为还有下一步但 `rounds >= 3` |
| `budget` | 超过总预算 | `plan` 或 `observe` 发现 `time.monotonic() - started_at > 90s` |
| `tool_error` | 工具/生成链路失败 | 所有观察都失败且无上下文；LLM 未配置/不可用；或图执行抛异常（图外兜底） |
| `verify_retry` | 自检失败后重写 | `verify` 发现不存在的真题编号或押题表述，且重试次数 < 1，写入后回到 `answer` 重写一次 |

补充：`answered` 与 `no_evidence` 在图外还有一次兜底（`graph.py` 第 273 行），避免异常路径留下空 `stop_reason`。

### 3.4 工具契约与独立超时

契约在 [contracts.py](file:///d:/agent/RAG/src/cs408rag/agent/contracts.py)，护栏在 [guardrails.py](file:///d:/agent/RAG/src/cs408rag/agent/guardrails.py)。

九个工具：`search_knowledge_base`、`search_knowledge_graph`、`read_figure`、`calculate`、`get_exam_stats`、`get_kp_questions`、`get_related_kps`、`get_learning_path`、`verify_answer`。

每个工具都有独立的 Pydantic 参数模型（`extra="forbid"`，`min_length/max_length/ge/le/pattern` 都写死在契约里），并声明独立超时：

| 工具 | 超时 | 重试 |
|---|---|---|
| `search_knowledge_base` | 30s | 0 |
| `calculate` | 5s | 1（幂等） |
| `verify_answer` | 10s | 0 |
| 其余工具 | 15s（默认） | 0 |

护栏顺序是固定的：**白名单 → 预算 → Pydantic 校验 → 工具特定检查 → 线程池执行 + 超时 → 审计**。

- 白名单之外的工具直接 `unknown_tool`，LLM planner 也不能绕过。
- 预算：单轮最多 6 次、每分钟最多 60 次；超限返回 `error_type="budget"`，不静默重试。
- `calculate` 在进入 simpleeval 前还会做一次表达式前置检查。
- 超时后主流程不再等待；Python 线程无法强杀，超时线程可能继续跑完，这是代码里明确承认的限制，不假装能终止。
- 审计只记录 `args_digest`（参数 JSON 的 12 位 SHA256 短哈希），不落原始参数。
- 错误类型固定为 `validation / not_found / timeout / internal / budget / unknown_tool`，planner 可以据此修正参数而不是把坏参数吞成「工具没用」。

### 3.5 澄清 HITL

`_clarify_check` 有两类触发：

1. **指代题但没上下文**：问句含「这题/这道题/选什么/下图/如图」，但 `qid` 为空、选项也全空——返回两条选项：给真题编号，或直接贴 A/B/C/D 选项文字。
2. **考点类问题但定位不到考点**：`route == "exam"` 或问句含「考点/知识点」，但 `extract_topic` 抽不出 tag——返回按真题频次排序的 Top 5 考点 + 「我换个说法，把考点名称写全」。

澄清节点是**终态**：不检索、不调 LLM，成本为 0。多轮评测里 19 轮中有 4 轮走澄清，澄清召回率 1.0。

### 3.6 成本追踪

[cost.py](file:///d:/agent/RAG/src/cs408rag/agent/cost.py) 按每千 token 的人民币价格估算：

```text
cost = prompt_tokens / 1000 * 输入价 + completion_tokens / 1000 * 输出价
```

| 模型（按名称子串匹配） | 输入价（元/1K） | 输出价（元/1K） |
|---|---|---|
| `deepseek-reasoner` | 0.004 | 0.016 |
| `deepseek-chat` | 0.002 | 0.008 |
| `qwen-plus` | 0.0008 | 0.002 |
| `qwen-turbo` | 0.0003 | 0.0006 |
| `default`（未命中时） | 0.002 | 0.008 |

- planner 的 token 也会累加进 `state["tokens"]`，不只是最终 answer 的 token。
- 结果保留 6 位小数写入 `AgentResult.cost_cny`，同时出现在 trace 和 UI 元信息里。
- 价格表是**估算口径**，用于观测与讲解，不作计费依据。

---

## 4. 技术取舍

| 选择 | 放弃了什么 | 为什么 | 代价 / 边界 |
|---|---|---|---|
| 固定状态图 + 有界 plan-act-observe | 自由 ReAct | 轮次、预算、planner 次数都可硬控，评测可复现；不会出现无限循环和成本失控 | 覆盖不了开放式多步任务；规则 planner 的覆盖范围就是能力上限 |
| 单 Agent + 9 工具 | 多 Agent 架构 | 408 场景的工具职责清晰、依赖共享检索链路；多 Agent 会引入通信成本和评测口径分裂 | 复杂任务没有角色分工；但这套任务本身不需要 |
| 不微调模型 | SFT / LoRA / 领域继续预训练 | 项目目标是工程闭环与可复现评测，不是模型训练；用检索、工具和规则解决领域问题 | 领域表达仍受基座模型限制；答案风格依赖 prompt |
| 内存 LRU + TTL 会话记忆 | SQLite / LangGraph checkpointer / 长期向量记忆 | 会话是短时上下文，不是用户资产；重启丢历史是有意为之，避免隐私与迁移成本 | 服务重启即失忆；不能跨设备续聊 |
| 规则指代消解 | 让 LLM 改写追问 | 规则可测、零成本、不会硬猜；解析不到就返回 `None`，交给澄清 | 只能覆盖「这题/这个考点/那 2021 年呢/先学什么」等固定模式 |
| `simpleeval` 计算器 | 自维护 AST + `eval()` 回退 | 不重复造安全求值器；包不可用时 **fail closed**，不偷偷降级；幂上限压到 10000 | 表达式能力受 simpleeval 限制；复杂公式仍需 LLM 先翻译成算式 |
| 不做重排 | 交叉编码器 rerank | 实测在本任务上系统性负收益，延迟还翻三倍 | 换中文/考试语料微调过的重排模型后可能需要重估 |
| 规则画像聚合 | LLM 生成用户画像 | 画像要可解释、可核对、可复现；事件是单一事实来源 | 标签提取依赖题库 tag 词表，俗称覆盖有限 |

---

## 5. 量化结果

只列仓库里能查到的数字。

### 5.1 检索

| 指标 | 结果 | 来源 |
|---|---|---|
| 生产检索 dev 452 题 Accuracy@10 | 0.668 | README |
| 生产检索 dev 452 题 Recall@10 | 0.382 | README |
| 生产检索 dev P50 | 35–49ms | README |
| held-out 2023–2026（99 题）Accuracy@10 | 0.687 | README |
| held-out P50 | 54ms | README |
| 重排对照 Recall@10（S1 → S2） | 0.4952 → 0.4232 | `reports/ablation_rerank.md` |
| 重排对照 MRR@10 | 0.3140 → 0.2812 | 同上 |
| 重排对照 nDCG@10 | 0.3389 → 0.2965 | 同上 |
| 重排对照 P50 | 111ms → 350ms | 同上 |

### 5.2 Agent 三路径（dev 21 题）

| 路径 | 准确率 | 空回答率 | 平均 token | 平均工具调用 | P50 | P95 |
|---|---|---|---|---|---|---|
| 纯 LLM（无检索） | 90.5% | 0.0% | 309 | 0.0 | 3.0s | 19.6s |
| 单轮 RAG | 85.7% | 0.0% | 4048 | 0.0 | 5.1s | 53.5s |
| Agent | 95.2% | 0.0% | 2127 | 2.0 | 5.6s | 23.0s |

配对比较：21 题中 Agent 改对 2 题（2016-15、2011-22），改错 0 题；Agent − 单轮 +9.5%，95% CI [+0.0%, +23.8%]，区间下界为 0，不能宣称显著。来源 [agent_eval.md](file:///d:/agent/RAG/reports/agent_eval.md)。

### 5.3 多轮对话（19 轮）

| 指标 | 结果 |
|---|---|
| 端到端通过率 | 94.7%（18/19） |
| 澄清召回率 | 1.0 |
| 指代消解率 | 1.0 |
| 工具失败率 | 3% |
| 平均工具轮次 | 0.95 |
| 总成本 | ¥0.1744（19 轮合计） |
| 单轮成本范围 | ¥0.0000–¥0.0187 |
| route 分布 | exam 10 / concept 5 / figure 2 / calc 2 |
| stop_reason 分布 | answered 15 / clarification 4 |

来源 [multi_turn_report.json](file:///d:/agent/RAG/data/eval/multi_turn_report.json)。唯一失败项是 `mt-12` 第二轮「我想问的是『死锁』」被路由成 `concept`，预期 `exam`。

### 5.4 对抗性 Plan

| 指标 | 结果 |
|---|---|
| 黄金 Plan 误报 | 0 / 10（0.0%） |
| 6 类注入错误检出 | 40 / 40（100.0%） |
| 挑战精确率 | 83.3%（48 条挑战中 40 条命中注入标签） |
| 单轮 vs 2 轮收敛率 | 0% → 100%（仅 1 组，不外推） |

来源 [adversarial_eval.md](file:///d:/agent/RAG/reports/adversarial_eval.md)。

### 5.5 测试

| 指标 | 结果 |
|---|---|
| 测试文件 | 10 个 |
| 测试函数 | 196 个（`def test_` 计数） |
| README 记录的最近一次通过数 | 135 passed（数据相关测试需本地题库） |

> 196 是静态计数，135 是 README 里的运行结果；两者口径不同，未在本机重新跑全量套件，因此并列展示，不合并成一个数。

### 5.6 待补

- 澄清触发后的二次解决率：待补。
- 真实计费账单核对：待补（当前只有估算口径）。
- 3 轮以上长会话的记忆压缩效果：待补（评测集最长 3 轮）。

---

## 6. 可深挖细节

### 6.1 会话记忆压缩策略

[memory.py](file:///d:/agent/RAG/src/cs408rag/agent/memory.py)：

- 全局 `MAX_SESSIONS = 200`，`SESSION_TTL_S = 2h`，`OrderedDict` + `Lock` 实现 LRU + TTL；访问即 `move_to_end`，过期与超量在每次读写前清理。
- `MAX_TURNS = 8`：会话内最多保留 8 个完整轮次；第 9 轮进来时把最早一轮弹出，压成一行规则摘要，**不调 LLM**。
- 摘要行格式：`· 问过「问题前 30 字」（route）→ 答案前 60 字`；`SUMMARY_MAX_CHARS = 800`，超限从头部按行丢弃，保留最近摘要。
- 注入 prompt 的只有 `summary + 最近 RECENT_TURNS = 4 轮`；每轮答案截断 300 字。
- `turn_count` 包含被压缩掉的轮次，所以 `turn_no` 是真实累计轮次，不会因为压缩而回退。
- `SessionStore` 独立于 `AgentTools`：工具实例每次 `run` 会 `reset` trace，会话历史挂在工具上会被顺带清掉（代码注释明确警告过这一点）。
- 会话不落盘；`/v1/agent/session/{id}` 只能查到当前进程内的会话，重启即 404。

### 6.2 指代消解规则

[followup.py](file:///d:/agent/RAG/src/cs408rag/agent/followup.py) 是纯规则：

1. **年份追问**：正则匹配「那 2021 年呢 / 2021 年呢？」，若上轮有 `last_topic`，改写成「{topic}在{year}年考过几次？分别是哪些真题？」；没有 topic 返回 `None`。
2. **学习路径追问**：问句含「先学/怎么学/学习路径/学习顺序/前置」，且上轮有 `last_topic`、问句里又没出现该 topic，则把 topic 前置拼成完整问题。
3. **考点指代**：问句含「这个考点/该考点/这个知识点/该知识点/这一章」，把指代词替换成 `last_topic`。
4. **题目指代**：问句含「这题/这道题/该题/此题/上述题/刚那题/刚才那题」，或「它」且问句不超过 12 字，且上轮有 `last_qid`——**只补 qid，不改写问题本身**。

原则是「解析不到就返回 `None`，不硬猜」；消解结果里的 `note` 会写进 steps，前端可解释「为什么这轮变成了另一个问题」。

### 6.3 检索融合与 rerank 参数

- RRF：`score(d) = Σ w_route / (k + rank)`，`k = 60`，缺省权重 1.0；代码在 [fusion.py](file:///d:/agent/RAG/src/cs408rag/retrieval/fusion.py#L45-L78)。
- 加权融合：每路分数先归一化（生产用 `zscore`，也支持 `minmax` / `rank`），再按 `w_dense / w_sparse / w_bm25` 加权求和；生产等权。
- 召回：每路 `top_k = 20`；融合后截 `fusion_top_k = 50`；最终取 10（问答）或 8（Agent 工具）。
- 节去重：同一小节只保留 1 条，避免同一章节的多个 chunk 挤占 top-k；实测 dev 题级命中率从 0.595 升到 0.662，代价是 chunk 级 Recall 下降。
- 重排：`bge-reranker-v2-m3`，`batch_size=16`、`max_length=1024`，只对融合后 top-N 候选逐对打分；生产关闭，保留 `unload()` 以便和 BGE-M3 在 16GB 机器上交替驻留。
- 降级：重排不可用时回退融合结果；无 reranker 时还有字符级 Jaccard 的 `lexical_overlap_score`，但评测必须标注用的是哪一路，否则数字不可比。

### 6.4 引用可展开核对

- 生成前，`build_context` 把检索结果转成 `[1] [2] [3]` 编号块，system prompt 明确「只许引用给定编号，不允许凭空引王道第 X 章」。问答侧上下文预算 6000 字，Agent 侧 2400 字。
- 生成后，`check_citations` 解析 `[1]`、`【1】`、`［1］` 等括号形式，计算引用精确率/召回率，并检查引用的编号是否在可用集合内；`extract_exam_refs` 解析 `2011#21` 这类真题编号，`detect_predictions` 拦截「今年可能考」等押题表述。
- 校验失败时 `verify` 只允许重写一次，并把具体问题（不存在的编号、押题词）带回给模型，而不是原样再问。
- UI 侧问答 Tab 对每个引用块用 `st.expander` 展示，标签是 `[编号] 章节路径`，点开就是原文；带教材插图的块默认展开并渲染图片。Agent Tab 展示「引用校验✅/校验⚠️」，推理链可展开看到每一步的 `detail`。
- 真题编号白名单来自题库真实编号，`verify_answer` 工具也可单独调用做同一套检查。

### 6.5 画像标签如何加权到规划

事件存储：`profile_events` 表（SQLite，`client_id + kind + payload + ts`），`kind` 为 `question / plan / quiz`。

聚合 [stats.py](file:///d:/agent/RAG/src/cs408rag/profile/stats.py)：

- 从提问文本提取考点：整标签命中 > 最长公共片段 ≥3 字；2 字片段只在没有更强命中时兜底，且只取频次最高的 2 个——这是为了避免「映射」「地址」「划分」这类泛化词污染画像。
- `boost_tags` 的顺序是**答错考点 > 高频提问考点 > 最近一次规划的重点考点**，最多 8 个。

规划侧 [planner.py](file:///d:/agent/RAG/src/cs408rag/adversarial/planner.py)：

1. `normalize_profile_tags` 把画像标签转成 `{subject, tag}`，并入 `effects["boost_tags"]`，与用户主动填写的「担心点」走同一条加权通道。
2. `allocate_subject_hours` 对每个 boost 标签给对应科目 **+0.04 小时份额**，同时把薄弱科目钉在 30% 下限，避免被其他科目稀释。
3. `build_priority_kps` 把 `boost_tags` 传给 `select_high_value`，boost 考点**必进** `priority_kps`；科目内权重按 `roi / Σroi` 归一化。
4. ROI 本身是启发式，不是押题：`roi = 0.7 × yield + 0.3 × (1 − cost)`，`yield` 由频次、近 5 年、计算/图表占比、年份跨度加权；标签 S/A/B/C 只表示排序。

前端闭环：提问落 `question` 事件 → 画像更新 → 规划页自动预填薄弱科目 → 生成 Plan 落 `plan` 事件 → 推荐真题卡片作答落 `quiz` 事件 → 答错考点以最高优先级进入 `boost_tags`。

---

## 7. 与既有设定的偏差

如实列出代码与「既定设定」不一致或需要补充说明的地方。

| 设定 | 实际代码 | 偏差说明 |
|---|---|---|
| 内存 LRU 200 + TTL 2h | `MAX_SESSIONS = 200`、`SESSION_TTL_S = 2 * 3600` | 一致 |
| 超过 8 轮自动压缩为规则摘要 | `MAX_TURNS = 8`，第 9 轮触发；`RECENT_TURNS = 4` | 一致；但「只保留摘要 + 最近 4 轮」只指**注入 prompt**的内容，内存里仍保留最多 8 个完整轮次 |
| 工具调用最多 3 轮 | `MAX_TOOL_ROUNDS = 3` | 一致。但 `AgentTools.call` 的文档字符串声称 `retrieve` 内部检索会传 `count_budget=False`，实际 [nodes.py](file:///d:/agent/RAG/src/cs408rag/agent/nodes.py#L250-L256) 并没有传；检索子调用因此会先计入 `_round_calls`，只有到 `act` 的 `begin_round` 才被清零。两次改写各 3 条查询时，第 7 次检索会撞上单轮 6 次预算而被拒 |
| 7 种 stop_reason | 代码中确有 7 种：`answered / no_evidence / clarification / max_rounds / budget / tool_error / verify_retry` | 数量一致。但 `verify_retry` 写入后图会继续走一次 `answer`，第二次校验成功后**不会清除**该字段，最终结果可能仍是 `verify_retry`，语义上更像「发生过重试」而不是「当前仍失败」 |
| 规则消解返回 None | `resolve_followup` 在无 session、无 turns、无 last_topic、无 last_qid 时返回 `None` | 基本一致。额外规则：「它」且问句 ≤12 字也会走 qid 指代，比纯「这题」类规则更宽 |
| 澄清只针对无题号/无选项的选择题 | `_clarify_check` 除选择题外，还会对 `route == "exam"` 或含「考点/知识点」但抽不出 tag 的问题触发澄清 | 触发面比设定更宽，属于有意扩展；代价是可能多问一句 |
| 「不触发检索与 LLM」 | `clarify` 是终态，直接返回文本 | 一致，且多轮报告里 4 次澄清成本均为 0 |
| 成本追踪写入 trace 与 UI | `compute_cost_cny` + `AgentResult.cost_cny` + Streamlit 元信息 | 一致；但价格表按模型名子串匹配，未收录的模型（如 `deepseek-v4-flash`、`glm-5.2`）会落到 `default` 价，成本只能算估算值 |
| 画像事件聚合、答题同步画像与错题本 | `ProfileStore` + `build_profile` + notebook judge 接口；前端推荐卡片可作答 | 一致。画像是 SQLite 持久化事件库，会话记忆才是内存态——两者不是同一层，不违反「会话记忆不落盘」 |
| 禁止持久化会话记忆 | 会话只在 `InMemorySessionStore` | 一致 |
| 不允许无限循环 | 轮次 + 90s 预算 + planner 调用上限 | 一致；但预算只在 `plan`/`observe` 边界检查，单个 30s 工具仍可能让总耗时超过 90s，预算属于「协作式」而非硬超时 |

另外两点值得在面试里主动说明：

- **单轮 RAG 在 21 题小样本上没有赢过纯 LLM**（85.7% vs 90.5%），项目没有隐藏这个负结果；Agent 的 +9.5% 配对差值置信区间下界为 0，不能宣称显著。
- **测试口径**：README 记录 135 passed，仓库静态统计到 196 个 `def test_`；两者不是同一个口径，引用时应说明来源，不混用。

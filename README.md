# 408-RAG

[![ci](https://github.com/abedonl/408_Rag/actions/workflows/ci.yml/badge.svg)](https://github.com/abedonl/408_Rag/actions/workflows/ci.yml)

面向考研 408 的真题检索问答与考点分析系统：**720 道历年真题（2009–2026）+ 6182 个知识点节点 + 5296 个教材 chunk**，多路混合检索按题型选路，Agent 编排 13 个确定性工具，每条结论都能追溯到具体真题与教材段落。

## 这个项目解决什么问题

备考 408 的痛点不是「找不到资料」，而是：

1. **不知道什么值得背** —— 需要真题频次的客观统计，而不是泛泛的「重点章节」
2. **不懂的知识点查不动** —— 教材厚重，通用检索质量差
3. **计算题、图表题看不懂** —— 单纯文本检索对这类题目无解

对应地：混合检索解决第 2 点；Agent 编排解决第 1、3 点（计算题交给确定性计算器，图表题做 SVG 拓扑重建，考情咨询只归纳真实真题频次）。

## 核心能力

| 能力 | 说明 |
|---|---|
| 混合检索 | dense（BGE-M3）+ sparse + 原生 BM25，按题型自适应选路，zscore 融合 + 小节去重 + 父节上下文；交叉编码器重排经消融验证为负收益，未启用 |
| Agent 编排 | 单页聊天入口，状态机编排 13 个工具：检索 / 图谱 / 真题 / 考点统计 / 图形解析 / 计算器 / 答案校验 / 规划 / 画像 / 判分；支持指代消解、会话记忆（LRU + TTL + 摘要压缩）、失败重试 |
| 对抗性 Plan | 规则引擎排骨架 + 三角色（Planner / Critic / Expert）实时质疑 + HITL 人工裁定；规划是独立记录（按身份绑定），不写入会话历史 |
| 用户闭环 | 匿名 `client_id` 身份隔离（数据层 + 内存会话层双层防越权）、事件溯源学习画像、错题本判分、学习总结 |
| 引用溯源 | 回答附带教材出处与真题编号，图表题分层降级（T1/T2/T3）并说明不可用原因 |
| 评测 | 检索消融、Agent 三路径对比、考点结论质量、Plan 注入检出率，负结果如实写入报告 |

## 架构

```text
真题 / 教材 / 思维导图
        │  清洗、切块、父子结构、知识点图谱
        ▼
   Qdrant（dense+sparse+BM25）      Neo4j（6182 节点 + 148 前置边）
        │                                   │
        └──────────────┬────────────────────┘
                       ▼
用户问题 → 规则路由 → 检索 → zscore 融合 + 小节去重 → 父节上下文
                       │
        ┌──────────────┼──────────────────┐
        ▼              ▼                  ▼
     单轮 RAG      Agent（13 工具）    对抗性 Plan
                  计算器/图解析/考点    规则质疑 + LLM 文案
                  + 规划/画像/判分
```

## 关键指标

| 指标 | 结果 |
|---|---|
| 规模 | 720 题（2009–2026）/ 5296 chunk / 6182 节点 / 148 前置边 |
| 生产检索（dev 452 题） | Accuracy@10 **0.668**，Recall@10 0.382，P50 **35–49ms** |
| held-out（2023–2026，99 题） | Accuracy@10 **0.687**，P50 54ms |
| Agent 三路径（dev 21 题） | 纯 LLM 90.5% / 单轮 RAG 85.7% / Agent **95.2%**（小样本，不作显著性结论） |
| 对抗 Plan 规则 | 黄金 Plan **0 误报**，6 类注入 **40/40 检出** |
| 测试 | **268 passed**（数据相关测试需本地题库） |

## 快速开始

```bash
uv sync                              # 安装依赖
cp .env.example .env                 # 填 LLM key（不填也能跑检索、规则评测、离线 Plan）
docker compose up -d                 # Qdrant + Neo4j
uv run --no-sync streamlit run app/streamlit_app.py          # 前端 :8501
uv run --no-sync uvicorn cs408rag.api.main:app --port 8123   # 接口 :8123（含 /docs）
```

> `--no-sync` 不能省：`uv run` 会按 pyproject.toml 重新同步环境，把手工安装的 CUDA 版 torch 覆盖回 CPU 版。

首次构建数据（约 10 分钟，顺序有依赖）：

```bash
uv run --no-sync python -m cs408rag.ingestion.download --source all   # 下载语料
uv run --no-sync python -m cs408rag.ingestion.quiz_loader             # 真题切分 + 考点统计
uv run --no-sync python -m cs408rag.ingestion.xmind_loader            # 知识点图谱
uv run --no-sync python -m cs408rag.ingestion.tag_mapper              # tag → 知识点映射
uv run --no-sync python -m cs408rag.ingestion.image_index             # 教材插图索引
uv run --no-sync python -m cs408rag.ingestion.indexer                 # 建向量索引
uv run --no-sync python -m cs408rag.retrieval.graph_store             # 建 Neo4j 图谱
uv run --no-sync python -m cs408rag.eval.build_eval_set               # 构造评测集
```

也可以直接用脚本：`bash scripts/start.sh`（起界面）、`--api`（同时起接口）、`--bootstrap`（数据缺失时先跑完整管道）。

常用接口：

```bash
curl -X POST localhost:8123/v1/agent/query -H 'Content-Type: application/json' \
  -d '{"query":"死锁这个考点历年考过几次？和哪些知识点关联？","mode":"qa"}'
curl "localhost:8123/v1/exam/stats?tag=死锁"      # 考点频次统计
curl "localhost:8123/v1/figures/2009-04"          # 图表题分层解析
```

![对抗规划 demo](docs/demo.png)

## 目录结构

```
src/cs408rag/
├── ingestion/      # 数据管道：下载 / 真题 / 教材 / 图谱 / 插图索引
├── retrieval/      # 检索层：多路召回 + 融合 + 小节去重（含可选重排/图谱扩展）
├── figures/        # 图表题：SVG 拓扑重建 + T1/T2/T3 分层读取
├── generation/     # 生成层：prompt + 引用校验 + LLM 客户端
├── agent/          # Agent 编排：状态机、13 个工具、会话记忆、计算器沙箱、考点统计
├── adversarial/    # 对抗 Plan：规则引擎 + 三角色状态机 + 注入数据集
├── profile/        # 用户画像：SQLite 事件溯源 + 会话/规划持久化
├── notebook/       # 错题本：判分（答错入本、答对移出）
├── eval/           # 检索消融、Agent 对比、考点质量、对抗 Plan 评测
├── observability.py# 检索 trace / 消融留档（Markdown + /v1/traces）
└── api/            # FastAPI 服务层（/v1/agent/*、/v1/exam/*、/v1/figures/*）
app/                # Streamlit 前端：streamlit_app.py（单页中枢）+ plan_ui.py（规划看板）
tests/              # 268 个测试；CI 只跑不依赖数据的纯函数单测
scripts/            # start.sh / stop.sh / 模型基准 / 数据与镜像脚本
```

## 文档入口

| 文档 | 内容 |
|---|---|
| [docs/README_full.md](docs/README_full.md) | 完整版 README：逐轮消融、负结果、调参过程与复现命令 |
| [docs/interview-prep.md](docs/interview-prep.md) | 项目讲解与追问应对（只用仓库内可核对的事实） |
| [docs/interview-tech.md](docs/interview-tech.md) | 技术细节（检索、Agent、Plan 的实现取舍） |
| [reports/experiment_data.md](reports/experiment_data.md) | 检索实验数据存档 |

## 使用边界与已知限制

1. **语料与真题仅本地使用**：教材 PDF、真题、思维导图不入库（见 `.gitignore`），`data/processed` 需自行构建。
2. **ROI 是启发式**：频次 / 近年 / 题型 / 学习成本都是代理指标，不等于「必考」；8 周只排每科前 10 个考点（约覆盖 36–45% 历史考查），长尾不新学。
3. **前置边覆盖有限**：148 条边只连相邻兄弟节点，跨章依赖（如存储系统 → Cache）缺失，无边时跳过并计数。
4. **注入集是合成的**：Plan 错误为人为构造，检出率不等于真实规划质量；LLM 主观挑战单列，不计入检出率/误报率。
5. **小样本评测**：Agent 三路径仅 dev 21 题、每科目 Plan 只跑 1 次，均为描述性统计。
6. **生成侧延迟偏高**：特定 endpoint 长输出吞吐受限（完整问答 50–56s，TTFT 14.2s），已实现 SSE 流式路径，前端应优先使用。
7. **结构化 Plan 不依赖 LLM**：LLM 只润色方法与理由，其失败不会导致规划缺失，也不会改变阶段、小时、考点与真题号。

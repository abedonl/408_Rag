"""全局配置。

所有可调参数集中在这里，消融实验通过覆盖 Settings 实例来实现，
不改代码即可跑不同配置。
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ---------------------------------------------------------------------------
# 路径
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EVAL_DIR = DATA_DIR / "eval"
INDEX_DIR = DATA_DIR / "index"
REPORTS_DIR = PROJECT_ROOT / "reports"


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- LLM ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="https://openrouter.ai/api/v1", alias="OPENAI_BASE_URL")
    llm_model: str = Field(default="deepseek/deepseek-chat", alias="LLM_MODEL")
    llm_model_cheap: str = Field(default="deepseek/deepseek-chat", alias="LLM_MODEL_CHEAP")
    # 主模型返回空正文时的备用模型。
    # 实测 sensenova 有 10–23% 的题会把 max_tokens 全花在思考上；
    # glm-5.2 能把 reasoning 关到 0、空回答率 0，但准确率略低。
    # 因此用「主模型优先、空正文时降级」而不是直接换模型。
    llm_backup_model: str = Field(default="", alias="LLM_BACKUP_MODEL")
    # 对抗 Plan 的三个角色各自用不同模型，避免同一模型自证与同质化盲点：
    #   Planner 生成 -> 主模型（中文表达与规划能力最好）
    #   Critic  挑错 -> 备用模型（实测空回答为 0，适合短 JSON 判断）
    #   Expert  裁定 -> 默认回到主模型，与 Critic 形成独立意见
    # 留空时按上述默认链解析；也可分别指定第三个模型。
    adv_planner_model: str = Field(default="", alias="ADV_PLANNER_MODEL")
    adv_critic_model: str = Field(default="", alias="ADV_CRITIC_MODEL")
    adv_expert_model: str = Field(default="", alias="ADV_EXPERT_MODEL")
    # 是否允许模型「思考」（推理模型会在正文前生成 reasoning token）。
    #
    # 实测差异极大：
    #   开：TTFT 44s，且 max_tokens 较小时 reasoning 会吃光预算、content 为空
    #   关：TTFT 1.6s
    # 408 问答的答案不需要长链推理，关掉体验好得多。
    # 若换的供应商不支持该参数（会返回 400），客户端会自动降级重试一次。
    llm_enable_thinking: bool = Field(default=False, alias="LLM_ENABLE_THINKING")
    # 二期图表题 Tier3 用；留空则 read_figure 只跑 SVG 两层
    vlm_model: str = Field(default="", alias="VLM_MODEL")

    # --- 网络 ---
    # 实测：huggingface.co 与 raw.githubusercontent.com 不可达，必须走镜像
    hf_endpoint: str = Field(default="https://hf-mirror.com", alias="HF_ENDPOINT")
    hf_home: str = Field(default=str(PROJECT_ROOT / "models" / "hf"), alias="HF_HOME")

    # --- 本地模型 ---
    embedding_model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    embedding_device: str = Field(default="cuda", alias="EMBEDDING_DEVICE")
    embedding_batch_size: int = Field(default=8, alias="EMBEDDING_BATCH_SIZE")
    # BGE-M3 dense 输出维度
    embedding_dim: int = 1024

    # --- Qdrant ---
    qdrant_url: str = Field(default="http://localhost:6333", alias="QDRANT_URL")
    qdrant_api_key: str = Field(default="", alias="QDRANT_API_KEY")
    qdrant_collection: str = Field(default="cs408_chunks", alias="QDRANT_COLLECTION")

    # --- Neo4j ---
    neo4j_uri: str = Field(default="bolt://localhost:7687", alias="NEO4J_URI")
    neo4j_user: str = Field(default="neo4j", alias="NEO4J_USER")
    neo4j_password: str = Field(default="cs408rag_dev_pw", alias="NEO4J_PASSWORD")

    # --- 检索默认参数 ---
    retrieve_top_k: int = Field(default=20, alias="RETRIEVE_TOP_K")
    fusion_top_k: int = Field(default=50, alias="FUSION_TOP_K")
    rerank_top_n: int = Field(default=10, alias="RERANK_TOP_N")
    rrf_k: int = Field(default=60, alias="RRF_K")
    # 各召回路在 RRF 融合中的权重（加权融合策略用）
    alpha_dense: float = 0.5
    alpha_sparse: float = 0.3
    alpha_graph: float = 0.2

    # --- 评测防泄漏隔离（红线）---
    # 这些年份的真题解析禁止入索引，只用于最终评测。
    # 违反此项会让准确率虚高且评测失去意义。
    holdout_years: str = Field(default="2023,2024,2025,2026", alias="HOLDOUT_YEARS")

    # --- 服务 ---
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    @field_validator("hf_home", mode="before")
    @classmethod
    def _resolve_hf_home(cls, v: str) -> str:
        """把相对路径锚定到项目根，避免受 cwd 影响。"""
        p = Path(v)
        return str(p if p.is_absolute() else PROJECT_ROOT / p)

    @property
    def holdout_year_set(self) -> set[int]:
        return {int(y) for y in self.holdout_years.split(",") if y.strip()}

    def export_hf_env(self) -> None:
        """把 HF 镜像设置写进 os.environ。

        必须在 import transformers / FlagEmbedding **之前**调用，
        否则模型下载仍会指向不可达的 huggingface.co。

        三个实测踩过的坑，都在这里一次性堵掉：
          1. huggingface.co 不可达 -> 走 hf-mirror
          2. hf-mirror 对模型仓库里的 .DS_Store 返回 403 -> 下载时需 ignore_patterns
          3. HF 的 Xet 存储后端（cas-server.xethub.hf.co）连不上，会报 401
             -> 必须禁用 Xet 走普通 HTTP
        """
        os.environ["HF_ENDPOINT"] = self.hf_endpoint
        os.environ["HF_HOME"] = self.hf_home
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        # 关闭 Xet：否则 snapshot_download 会尝试连 xethub 并 401
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
        # 不要在 import 时发联网请求，避免离线环境卡住
        os.environ.setdefault("HF_HUB_DISABLE_IMPLICIT_TOKEN", "1")


# 下载模型时应忽略的文件模式（README 配图与 macOS 垃圾文件，对推理无用）
HF_IGNORE_PATTERNS = ["*.DS_Store", ".DS_Store", "imgs/*", "*.png", "*.jpg", "*.gif"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.export_hf_env()
    return s


settings = get_settings()


def ensure_dirs() -> None:
    """建好所有工作目录。"""
    for d in (RAW_DIR, PROCESSED_DIR, EVAL_DIR, INDEX_DIR, REPORTS_DIR, Path(settings.hf_home)):
        d.mkdir(parents=True, exist_ok=True)
"""集中配置：统一从环境变量 / .env 读取，全局单例 ``settings``。

设计要点：
- 全部配置项集中于此，其余模块不直接读 os.environ，便于测试与容器化覆盖。
- ``has_llm_key`` 用于判断是否具备真实调用 DeepSeek 的条件。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 仓库根目录（research_agent/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default


_PLACEHOLDER_KEYS = {"", "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}


@dataclass
class Settings:
    # ---- LLM (DeepSeek) ----
    deepseek_api_key: str = field(default_factory=lambda: _env("DEEPSEEK_API_KEY"))
    deepseek_base_url: str = field(default_factory=lambda: _env("DEEPSEEK_BASE_URL", "https://api.deepseek.com"))
    deepseek_model: str = field(default_factory=lambda: _env("DEEPSEEK_MODEL", "deepseek-chat"))
    llm_temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.0))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("LLM_MAX_TOKENS", 4096))

    # ---- Embedding ----
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "auto"))
    embedding_model: str = field(default_factory=lambda: _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"))

    # ---- RAG ----
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 500))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 80))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))

    # ---- Agent ----
    max_steps: int = field(default_factory=lambda: _env_int("MAX_STEPS", 6))

    # ---- 论文检索 ----
    # Semantic Scholar 的免费 Key 可把限流从「几乎必 429」提升到 1 req/s，
    # 是 arXiv 被限流时唯一可靠的在线兜底源，建议配置。
    semantic_scholar_api_key: str = field(default_factory=lambda: _env("SEMANTIC_SCHOLAR_API_KEY"))

    # ---- MCP ----
    mcp_transport: str = field(default_factory=lambda: _env("MCP_TRANSPORT", "stdio"))

    # ---- 路径 ----
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    papers_dir: Path = PROJECT_ROOT / "data" / "papers"
    index_dir: Path = PROJECT_ROOT / "data" / "vector_index"
    reports_dir: Path = PROJECT_ROOT / "data" / "reports"

    @property
    def has_llm_key(self) -> bool:
        """是否配置了可用的 DeepSeek Key（占位符视为未配置）。"""
        return self.deepseek_api_key not in _PLACEHOLDER_KEYS

    @property
    def has_s2_key(self) -> bool:
        """是否配置了 Semantic Scholar Key。"""
        return self.semantic_scholar_api_key not in _PLACEHOLDER_KEYS

    def ensure_dirs(self) -> None:
        for d in (self.data_dir, self.papers_dir, self.index_dir, self.reports_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()

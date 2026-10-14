"""Настройки подключения (.env) отделены от параметров эксперимента (YAML).

Пароли нельзя включать в манифесты, логи и идентификаторы экспериментов.
Поэтому мы никогда не сериализуем Settings целиком.
"""

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    chat_model: str = "gpt-4o-mini"
    judge_model: str = "gpt-4o-mini"
    judge_max_output_tokens: int = Field(default=4096, ge=256)
    judge_embedding_model: str = "text-embedding-3-small"
    judge_api_key: str = ""
    judge_base_url: str = ""
    database_url: str = "sqlite:///data/chat.db"
    history_turns: int = Field(default=6, ge=1, le=50)
    max_question_chars: int = Field(default=4000, ge=1)
    max_output_tokens: int = Field(default=1200, ge=1)
    request_timeout: float = Field(default=90, gt=0)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_path: Path = Path("data/qdrant")
    documents_dir: Path = Path("data/knowledge")
    index_dir: Path = Path("artifacts/indexes")
    rag_config: Path = Path("configs/baseline.yaml")
    langfuse_enabled: bool = False
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"
    ragas_do_not_track: bool = True

    def require_llm(self):
        if not self.openai_api_key or self.openai_api_key == "replace-me":
            raise ValueError("Укажите OPENAI_API_KEY и OPENAI_BASE_URL в .env")


class RagConfig(BaseModel):
    model_config = {"extra": "forbid"}
    name: str = Field(default="baseline", pattern=r"^[a-zA-Z0-9_-]+$")
    chunk_strategy: Literal["fixed", "paragraph", "recursive_tokens"] = "paragraph"
    token_encoding: str = "cl100k_base"
    chunk_size: int = Field(default=700, ge=50, le=10000)
    chunk_overlap: int = Field(default=80, ge=0)
    embedding_model: str = "text-embedding-3-small"
    top_k: int = Field(default=4, ge=1, le=20)

    @model_validator(mode="after")
    def check_overlap(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap должен быть меньше chunk_size")
        return self

    @classmethod
    def read(cls, path: Path):
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def digest(value) -> str:
    """Стабильный SHA-256: одинаковые данные дают одинаковый fingerprint."""
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()

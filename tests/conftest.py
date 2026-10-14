"""Тесты не отправляют запросы моделям и не требуют Docker/ключей."""

import os

os.environ["RAGAS_DO_NOT_TRACK"] = "true"

import pytest

from rag_lab.config import Settings


@pytest.fixture
def settings(tmp_path):
    docs = tmp_path / "knowledge"
    docs.mkdir()
    (docs / "courses.md").write_text(
        "Курс Python длится 8 недель.\n\nКурс RAG длится 6 недель.\n\n" * 8,
        encoding="utf-8",
    )
    return Settings(
        _env_file=None,
        openai_api_key="test-key",
        documents_dir=docs,
        database_url=f"sqlite:///{tmp_path / 'chat.db'}",
        qdrant_url="local",
        qdrant_path=tmp_path / "qdrant",
        index_dir=tmp_path / "indexes",
        langfuse_enabled=False,
    )


class FakeModels:
    """Предсказуемая заглушка для проверки транспорта данных, НЕ оценки качества."""

    def __init__(self):
        self.embedding_calls = 0
        self.chat_calls = []

    def embed(self, texts, model):
        self.embedding_calls += 1
        return [
            [float(text.lower().count("python") + 1), float(text.lower().count("rag") + 1), 1.0]
            for text in texts
        ]

    def chat(self, messages, *, purpose, json_mode=False):
        self.chat_calls.append((purpose, messages))
        if purpose == "rewrite-query":
            return "Сколько длится курс RAG?"
        return "Курс RAG длится 6 недель. [1]"

    def close(self):
        pass


@pytest.fixture
def provider():
    return FakeModels()

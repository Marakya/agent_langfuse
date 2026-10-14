"""Здесь используются настоящие Qdrant local и SQLite, подменена только LLM."""

import json

import pytest
from qdrant_client import QdrantClient

from rag_lab.config import RagConfig
from rag_lab.documents import Document, split_document
from rag_lab.index import Index
from rag_lab.memory import ConcurrentTurnError, Memory
from rag_lab.pipeline import Rag
from rag_lab.telemetry import Telemetry


@pytest.mark.parametrize("strategy", ["fixed", "paragraph"])
@pytest.mark.parametrize("overlap", [0, 15, 49])
def test_chunking_preserves_coverage_and_offsets(strategy, overlap):
    doc = Document("a.md", "Начало.\n\n" + "важный текст. " * 40 + "КОНЕЦ")
    config = RagConfig(chunk_strategy=strategy, chunk_size=50, chunk_overlap=overlap)
    chunks = split_document(doc, config)
    covered = set()
    for chunk in chunks:
        assert chunk.text == doc.text[chunk.start : chunk.end]
        assert len(chunk.text) <= 50
        covered.update(range(chunk.start, chunk.end))
    assert covered == set(range(len(doc.text)))
    assert chunks[-1].text.endswith("КОНЕЦ")


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        RagConfig(chunk_size=100, chunk_overlap=100)


def test_index_idempotency_and_isolation(settings, provider):
    client = QdrantClient(":memory:")
    try:
        config = RagConfig(chunk_size=100, chunk_overlap=20)
        index = Index(settings, config, client)
        assert not index.ready()
        manifest = index.build(provider)
        calls = provider.embedding_calls
        assert index.build(provider) == manifest
        assert provider.embedding_calls == calls  # Повторный ingest не платит за API.
        assert client.count(index.collection).count == manifest["chunks"]
        assert index.search("RAG", provider)[0]["source"] == "courses.md"
        changed_model = Index(
            settings, config.model_copy(update={"embedding_model": "other"}), client
        )
        assert changed_model.collection != index.collection
        changed_top_k = Index(settings, config.model_copy(update={"top_k": 2}), client)
        assert changed_top_k.collection == index.collection
        (settings.documents_dir / "new.md").write_text("Новый документ", encoding="utf-8")
        assert Index(settings, config, client).collection != index.collection
    finally:
        client.close()


def test_chat_memory_restart_and_no_cross_session_leak(settings, provider):
    client = QdrantClient(":memory:")
    memory = Memory(settings.database_url)
    try:
        index = Index(settings, RagConfig(), client)
        index.build(provider)
        rag = Rag(settings, index, provider, Telemetry(settings))
        first, second = memory.create(), memory.create()
        rag.chat(memory, first, "Расскажи про RAG")
        memory.close()
        memory = Memory(settings.database_url)
        answer = rag.chat(memory, first, "А сколько он длится?")
        assert answer.search_query == "Сколько длится курс RAG?"
        rewrite = next(
            messages for purpose, messages in provider.chat_calls if purpose == "rewrite-query"
        )
        assert "Расскажи про RAG" in rewrite[-1]["content"]
        assert len(memory.history(first)[1]) == 4
        assert memory.history(second)[1] == []
        generation = provider.chat_calls[-1][1]
        payload = json.loads(generation[-1]["content"])
        assert payload["sources"][0]["text"] == answer.contexts[0]["text"]
        assert memory.history(first)[1][-1]["details"]["contexts"] == answer.contexts
    finally:
        memory.close()
        client.close()


def test_atomic_memory_and_stale_revision(settings):
    memory = Memory(settings.database_url)
    try:
        session = memory.create()
        memory.save_turn(session, 0, "question", "answer", {})
        with pytest.raises(ConcurrentTurnError):
            memory.save_turn(session, 0, "stale", "must not save", {})
        assert len(memory.history(session)[1]) == 2
        assert memory.history(session)[0] == 1
    finally:
        memory.close()


def test_provider_failure_does_not_save_half_turn(settings, provider, monkeypatch):
    client = QdrantClient(":memory:")
    memory = Memory(settings.database_url)
    try:
        index = Index(settings, RagConfig(), client)
        index.build(provider)
        rag = Rag(settings, index, provider, Telemetry(settings))
        session = memory.create()

        def fail(*args, **kwargs):
            raise RuntimeError("provider unavailable")

        monkeypatch.setattr(provider, "chat", fail)
        with pytest.raises(RuntimeError):
            rag.chat(memory, session, "Вопрос")
        assert memory.history(session) == (0, [])
    finally:
        memory.close()
        client.close()

import json

import pytest
import tiktoken
from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
from qdrant_client import QdrantClient

from rag_lab.config import RagConfig
from rag_lab.dataset import Example, validate_sources
from rag_lab.documents import Document, load_documents, split_document
from rag_lab.index import Index
from rag_lab.pipeline import Rag
from rag_lab.telemetry import Telemetry


def make_pdf(path, texts):
    writer = PdfWriter()
    font = writer._add_object(
        DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
    )
    for text in texts:
        page = writer.add_blank_page(width=600, height=800)
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 20 750 Td ({text}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    writer.write(path)


def test_pdf_pages_overlap_and_blank_warning(tmp_path):
    make_pdf(tmp_path / "book.pdf", ["Python " * 60, "", "RAG " * 60])
    with pytest.warns(UserWarning, match="2.*OCR"):
        docs = load_documents(tmp_path)
    assert [d.page for d in docs] == [1, 3]
    assert docs[0].parent_id != docs[1].parent_id
    config = RagConfig(chunk_size=100, chunk_overlap=20)
    for doc in docs:
        chunks = split_document(doc, config)
        assert chunks[0].start == 0 and chunks[-1].end == len(doc.text)
        assert all(c.text == doc.text[c.start : c.end] for c in chunks)
        assert all(c.parent_id == doc.parent_id and c.page == doc.page for c in chunks)
        assert all(a.end - b.start == 20 for a, b in zip(chunks, chunks[1:]))
    validate_sources(
        [
            Example(
                id="pdf",
                question="q",
                reference="a",
                reference_sources=["book.pdf"],
                reference_contexts=["Python", "RAG"],
            )
        ],
        docs,
    )


def test_empty_pdf_requires_ocr(tmp_path):
    make_pdf(tmp_path / "scan.pdf", [""])
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="OCR"):
        load_documents(tmp_path)


@pytest.mark.parametrize("strategy", ["paragraph", "recursive_tokens"])
def test_parent_retrieval_persistence_and_generation(settings, provider, strategy):
    (settings.documents_dir / "courses.md").unlink()
    make_pdf(settings.documents_dir / "book.pdf", ["Python " * 100, "RAG " * 90])
    config = RagConfig(chunk_strategy=strategy, chunk_size=100, chunk_overlap=20, top_k=2)
    client = QdrantClient(path=str(settings.qdrant_path))
    index = Index(settings, config, client)
    manifest = index.build(provider)
    assert manifest["documents"] == 1 and manifest["parent_pages"] == 2
    points, _ = client.scroll(index.collection, limit=100, with_vectors=True)
    assert len(points) == manifest["chunks"] > 2
    if strategy == "recursive_tokens":
        assert all(p.payload["token_count"] <= 100 for p in points)
    else:
        assert all(len(p.payload["text"]) <= 100 for p in points)
    assert all("parent_id" in p.payload and "parent_text" not in p.payload for p in points)
    client.close()
    client = QdrantClient(path=str(settings.qdrant_path))
    try:
        index = Index(settings, config, client)
        calls = provider.embedding_calls
        assert index.build(provider) == manifest
        assert provider.embedding_calls == calls
        contexts = index.search("Python", provider)
        assert len(contexts) == 2 and {c["page"] for c in contexts} == {1, 2}
        assert contexts[0]["page"] == 1
        for context in contexts:
            doc = next(d for d in index.documents if d.page == context["page"])
            assert context["text"] == doc.text
            assert context["end"] == len(doc.text)
            assert context["matched_child"]["text"] in doc.text
        rag = Rag(settings, index, provider, Telemetry(settings))
        answer = rag.answer("Python")
        sent = json.loads(provider.chat_calls[-1][1][-1]["content"])["sources"]
        assert sent[0]["text"] == answer.contexts[0]["text"]
        assert sent[0]["page"] == 1
        index.parents_path.write_text("{}", encoding="utf-8")
        assert not index.ready()
        with pytest.raises(ValueError, match="ingest"):
            index.search("Python", provider)
        index.build(provider)
        assert index.ready()
    finally:
        client.close()


@pytest.mark.parametrize(
    "text",
    [
        "Начало страницы.\n\n"
        + "Переменные данные чертежа относятся к исполнению. " * 90
        + "КОНЕЦ",
        "Python " * 300,
        "абвгд😀中文" * 100,
        "Первая строка\n" * 100 + "\n\n<|endoftext|> обычный текст документа",
    ],
    ids=["russian", "repeated", "unicode", "special_marker"],
)
def test_recursive_tokens_preserve_text_offsets_and_budget(text):
    config = RagConfig(chunk_strategy="recursive_tokens", chunk_size=80, chunk_overlap=15)
    doc = Document("book.pdf", text, page=7)
    chunks = split_document(doc, config)
    encoder = tiktoken.get_encoding(config.token_encoding)
    assert len(chunks) > 1
    assert chunks[0].start == 0 and chunks[-1].end == len(text)
    covered = set()
    for chunk in chunks:
        assert chunk.text == text[chunk.start : chunk.end]
        assert chunk.token_count == len(encoder.encode(chunk.text, disallowed_special=()))
        assert chunk.token_count <= config.chunk_size
        assert chunk.parent_id == doc.parent_id and chunk.page == 7
        covered.update(range(chunk.start, chunk.end))
    assert all(i in covered for i, char in enumerate(text) if not char.isspace())
    # Целое предложение может быть длиннее overlap: между такими блоками
    # LangChain не создаёт перекрытие. На коротких повторяющихся словах оно есть.
    if text.startswith("Python"):
        assert any(a.end > b.start for a, b in zip(chunks, chunks[1:]))


def test_tokenizer_changes_index_identity(settings):
    client = QdrantClient(":memory:")
    try:
        config = RagConfig(chunk_strategy="recursive_tokens")
        first = Index(settings, config, client)
        second = Index(settings, config.model_copy(update={"token_encoding": "o200k_base"}), client)
        assert first.collection != second.collection
        assert first.spec["chunk_unit"] == "tokens"
    finally:
        client.close()

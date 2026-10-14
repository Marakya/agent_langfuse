"""Проверка фактического корпуса и его загрузки без сетевых запросов."""

import hashlib
import json
import re
from pathlib import Path

from rag_lab.config import RagConfig
from rag_lab.documents import load_documents, split_document

ROOT = Path(__file__).resolve().parents[1]


def test_standard_texts_and_local_illustrations():
    manifest = json.loads((ROOT / "data/sources/gost/manifest.json").read_text(encoding="utf-8"))
    documents = {d.source: d for d in load_documents(ROOT / "data/knowledge")}
    config = RagConfig.read(ROOT / "configs/baseline.yaml")
    assert len(manifest["documents"]) == 6
    assert sum(d["text_characters"] for d in manifest["documents"]) > 190_000
    for entry in manifest["documents"]:
        path = ROOT / entry["markdown"]
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry["markdown_sha256"]
        assert hashlib.sha256((ROOT / entry["raw_html"]).read_bytes()).hexdigest() == entry["raw_sha256"]
        text = raw.decode("utf-8")
        assert "captcha" not in text.lower()
        assert len(text) > 10_000
        images = re.findall(r"!\[[^]]*\]\(([^)]+)\)", text)
        assert len(images) == len(entry["illustrations"])
        for relative in images:
            assert (path.parent / relative).is_file()
        for illustration in entry["illustrations"]:
            assert (ROOT / illustration["path"]).stat().st_size > 0
        offline = (ROOT / entry["offline_html"]).read_text(encoding="utf-8")
        assert "<script" not in offline
        source = path.relative_to(ROOT / "data/knowledge").as_posix()
        chunks = split_document(documents[source], config)
        assert len(chunks) > 10
        assert chunks[0].start == 0
        assert chunks[-1].end == len(documents[source].text)
        assert all(right.start <= left.end for left, right in zip(chunks, chunks[1:]))

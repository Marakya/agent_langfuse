"""Индекс — снимок корпуса + способ разбиения + эмбеддер + endpoint.

Манифест с ready=true публикуется только после всех upsert. Повторный запуск
проверяет число точек. Частично записанный индекс безопасно дописывается:
UUID точки зависит от fingerprint и номера чанка.
"""

import json
from dataclasses import asdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from rag_lab.config import RagConfig, Settings, digest
from rag_lab.documents import corpus_hash, load_documents, split_document


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    # replace не оставляет полузаписанный JSON при падении процесса.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    temporary.replace(path)


def connect_qdrant(settings: Settings):
    if settings.qdrant_url == "local":
        settings.qdrant_path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(settings.qdrant_path))
    return QdrantClient(
        url=settings.qdrant_url, api_key=settings.qdrant_api_key or None, timeout=60
    )


class Index:
    def __init__(self, settings: Settings, config: RagConfig, client: QdrantClient):
        self.settings, self.config, self.client = settings, config, client
        self.documents = load_documents(settings.documents_dir)
        self.corpus_hash = corpus_hash(self.documents)
        self.spec = {
            "schema": 2,
            "splitter_version": 3,
            "pdf_retrieval": "parent_page",
            "corpus_hash": self.corpus_hash,
            "chunk_strategy": config.chunk_strategy,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
            "chunk_unit": "tokens" if config.chunk_strategy == "recursive_tokens" else "characters",
            "token_encoding": config.token_encoding
            if config.chunk_strategy == "recursive_tokens"
            else None,
            "embedding_model": config.embedding_model,
            "embedding_endpoint_hash": digest(settings.openai_base_url.rstrip("/")),
        }
        # top_k не влияет на векторы: его варианты могут разделять один индекс.
        self.collection = "classroom_" + digest(self.spec)[:24]
        self.manifest_path = settings.index_dir / f"{self.collection}.json"
        self.parents_path = settings.index_dir / f"{self.collection}.parents.json"

    def parent_documents(self) -> dict:
        return {doc.parent_id: asdict(doc) for doc in self.documents if doc.parent_id is not None}

    def ready(self) -> bool:
        if not self.manifest_path.exists() or not self.client.collection_exists(self.collection):
            return False
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if not self.parents_path.exists():
            return False
        try:
            parents = json.loads(self.parents_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return False
        if parents != self.parent_documents():
            return False
        return bool(
            manifest.get("ready")
            and manifest["spec"] == self.spec
            and self.client.count(self.collection, exact=True).count == manifest["chunks"]
        )

    def build(self, provider) -> dict:
        if self.ready():
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        chunks = [chunk for doc in self.documents for chunk in split_document(doc, self.config)]
        if not chunks:
            raise ValueError("После разбиения нет чанков")
        dimensions = None
        for start in range(0, len(chunks), 32):
            batch = chunks[start : start + 32]
            vectors = provider.embed([chunk.text for chunk in batch], self.config.embedding_model)
            if len(vectors) != len(batch):
                raise ValueError("Количество эмбеддингов не совпадает с количеством чанков")
            dimensions = len(vectors[0])
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(
                        size=dimensions, distance=models.Distance.COSINE
                    ),
                )
            points = [
                models.PointStruct(
                    id=str(uuid5(NAMESPACE_URL, f"{self.collection}:{start + offset}")),
                    vector=vector,
                    payload={**asdict(chunk), "ordinal": start + offset},
                )
                for offset, (chunk, vector) in enumerate(zip(batch, vectors, strict=True))
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)
        manifest = {
            "ready": True,
            "spec": self.spec,
            "collection": self.collection,
            "documents": len({doc.source for doc in self.documents}),
            "parent_pages": len(self.parent_documents()),
            "chunks": len(chunks),
            "dimensions": dimensions,
        }
        if self.client.count(self.collection, exact=True).count != len(chunks):
            raise RuntimeError("Число точек Qdrant не совпало с числом чанков")
        write_json(self.parents_path, self.parent_documents())
        write_json(self.manifest_path, manifest)
        return manifest

    def search(self, query: str, provider) -> list[dict]:
        if not self.ready():
            raise ValueError("Индекс отсутствует или изменился корпус. Выполните rag-chat ingest.")
        vector = provider.embed([query], self.config.embedding_model)[0]
        parents = json.loads(self.parents_path.read_text(encoding="utf-8"))
        # Расширяем выборку, если несколько лучших child принадлежат одной странице.
        total = self.client.count(self.collection, exact=True).count
        limit = min(self.config.top_k, total)
        while limit:
            result = self.client.query_points(
                collection_name=self.collection,
                query=vector,
                limit=limit,
                with_payload=True,
            )
            contexts, seen = [], set()
            for hit in result.points:
                payload = hit.payload
                parent_id = payload.get("parent_id")
                key = parent_id or str(hit.id)
                if key in seen:
                    continue
                seen.add(key)
                context = {"id": str(hit.id), "score": hit.score, **payload}
                if parent_id:
                    parent = parents[parent_id]
                    context.update(
                        id=parent_id,
                        text=parent["text"],
                        start=0,
                        end=len(parent["text"]),
                        matched_child={"id": str(hit.id), **payload},
                    )
                contexts.append(context)
                if len(contexts) == self.config.top_k:
                    return contexts
            if limit >= total or len(result.points) < limit:
                return contexts
            limit = min(limit * 2, total)
        return []

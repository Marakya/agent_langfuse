"""Датасет отделён от ответов RAG: эталон нельзя брать у оцениваемого бота.

Синтетические вопросы создаются по исходным документам, НЕ по чанкам
конкретного эксперимента. Иначе один splitter получает несправедливое
преимущество. Генерация создаёт draft, человек сверяет факты и ставит reviewed.
"""

from pathlib import Path
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, Field, model_validator

from rag_lab.config import digest
from rag_lab.documents import Document


class HistoryMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class Example(BaseModel):
    model_config = {"extra": "forbid"}
    id: str = Field(min_length=1)
    question: str = Field(min_length=1, max_length=4000)
    reference: str = Field(min_length=1)
    reference_sources: list[str]
    reference_contexts: list[str] = Field(default_factory=list)
    history: list[HistoryMessage] = Field(default_factory=list)
    standalone_question: str | None = None
    kind: Literal["single_hop", "multi_hop", "conversational", "unanswerable"] = "single_hop"
    split: Literal["dev", "test"] = "dev"
    reviewed: bool = False

    @model_validator(mode="after")
    def check_ground_truth(self):
        if self.kind != "unanswerable" and not self.reference_sources:
            raise ValueError("Для отвечаемого вопроса нужны reference_sources")
        if self.history:
            if not self.standalone_question:
                raise ValueError("Для истории нужен проверенный standalone_question")
            expected_roles = ["user", "assistant"] * (len(self.history) // 2)
            if [m.role for m in self.history] != expected_roles:
                raise ValueError("history должна состоять из полных пар user/assistant")
        if self.kind == "conversational" and not self.history:
            raise ValueError("Для conversational нужна история")
        return self


def read_dataset(path: Path, *, split=None, require_reviewed=True) -> list[Example]:
    examples = []
    ids, questions = set(), set()
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = Example.model_validate_json(line)
            key = (item.question.casefold().strip(), item.model_dump_json(include={"history"}))
            if item.id in ids or key in questions:
                raise ValueError("Повтор ID или вопроса с той же историей")
            ids.add(item.id)
            questions.add(key)
            if require_reviewed and not item.reviewed:
                raise ValueError(
                    "Непроверенный пример: проверьте эталон и установите reviewed=true"
                )
            if split is None or item.split == split:
                examples.append(item)
        except ValueError as error:
            raise ValueError(f"{path}:{number}: {error}") from error
    if not examples:
        raise ValueError("Датасет/выбранная часть пуста")
    return examples


def validate_sources(examples: list[Example], documents: list[Document]):
    sources = {}
    for doc in documents:
        sources[doc.source] = sources.get(doc.source, "") + "\n\n" + doc.text
    for item in examples:
        for source in item.reference_sources:
            if source not in sources:
                raise ValueError(f"{item.id}: источник {source} отсутствует")
        for quote in item.reference_contexts:
            if not any(quote in sources[source] for source in item.reference_sources):
                raise ValueError(f"{item.id}: эталонная цитата не найдена в указанных документах")


def dataset_fingerprint(examples: list[Example]):
    return digest([item.model_dump() for item in examples])


def publish_dataset(client, examples: list[Example], name: str) -> dict:
    """Версионное имя не даёт незаметно подменить эталоны старого эксперимента."""
    from rag_lab.progress import progress

    progress("dataset.publish.start", dataset=name, examples=len(examples))
    reviewed = all(item.reviewed for item in examples)
    client.create_dataset(
        name=name,
        description="Проверенный датасет для оценки RAG"
        if reviewed
        else "Ragas draft: эталоны не проверены, оценки предварительные",
        metadata={"dataset_hash": dataset_fingerprint(examples), "reviewed": reviewed},
    )
    mapping = {}
    for item in examples:
        stable_id = str(uuid5(NAMESPACE_URL, f"{name}:{item.id}"))
        remote_item = client.create_dataset_item(
            id=stable_id,
            dataset_name=name,
            input={"question": item.question, "history": [m.model_dump() for m in item.history]},
            expected_output={
                "reference": item.reference,
                "sources": item.reference_sources,
                "reference_contexts": item.reference_contexts,
            },
            metadata=item.model_dump(
                exclude={"question", "history", "reference", "reference_contexts"}
            ),
        )
        mapping[item.id] = remote_item
        progress("dataset.item.done", id=item.id, remote_id=stable_id, sent=len(mapping), total=len(examples))
    client.flush()
    progress("dataset.publish.done", dataset=name, items=len(mapping))
    return mapping

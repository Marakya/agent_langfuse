"""Полный локальный прогон и настоящий Langfuse runner с подменой транспорта."""

import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from langfuse import Langfuse
from langfuse.api import DatasetItem
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from rag_lab.config import RagConfig
from rag_lab.dataset import Example
from rag_lab.experiments import run_evaluation
from rag_lab.langfuse_export import item_evaluations, publish_saved_run
from rag_lab.telemetry import Telemetry


class FakeJudge:
    def __init__(self, settings):
        pass

    async def score(self, item, answer):
        assert answer["contexts"]
        return {"faithfulness": 0.75}, {}, {}


@pytest.mark.parametrize("reviewed", [True, False])
def test_full_local_report(settings, provider, monkeypatch, tmp_path, reviewed):
    monkeypatch.setattr("rag_lab.runtime.Models", lambda *args: provider)
    monkeypatch.setattr("rag_lab.experiments.Judge", FakeJudge)
    item = Example(
        id="one",
        question="RAG?",
        reference="6 недель",
        reference_sources=["courses.md"],
        reviewed=reviewed,
    )
    directory, summary = run_evaluation(settings, RagConfig(), [item], tmp_path / "runs")
    assert summary["failed_examples"] == 0
    assert summary["metrics"]["faithfulness"]["mean"] == 0.75
    assert {
        "manifest.json",
        "dataset.jsonl",
        "results.jsonl",
        "report.md",
        "scores.csv",
        "corpus.json",
        "summary.json",
        "ragas_evaluation.jsonl",
    } <= {p.name for p in directory.iterdir()}
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["dataset_reviewed"] is reviewed
    assert manifest["evaluation_mode"] == ("reviewed" if reviewed else "draft_preview")
    native = json.loads((directory / "ragas_evaluation.jsonl").read_text(encoding="utf-8"))
    assert native["user_input"] == item.question
    assert native["reference"] == item.reference
    assert native["retrieved_contexts"]
    assert (
        "ПРОБНЫЙ ПРОГОН" in (directory / "report.md").read_text(encoding="utf-8")
    ) is not reviewed
    assert "test-key" not in (directory / "manifest.json").read_text(encoding="utf-8")
    assert not Path(settings.database_url.removeprefix("sqlite:///")).exists()


def test_real_langfuse_runner_and_nested_traces(settings, provider, monkeypatch, tmp_path):
    exporter = InMemorySpanExporter()

    def deny_network(request):
        raise AssertionError(f"Unexpected HTTP request in isolated test: {request.url.path}")

    client = Langfuse(
        public_key="pk-test-" + uuid4().hex,
        secret_key="sk-test",
        base_url="http://localhost:39999",
        span_exporter=exporter,
        httpx_client=httpx.Client(transport=httpx.MockTransport(deny_network)),
    )
    telemetry = Telemetry(settings)
    telemetry.client = client
    scores, links = [], []
    monkeypatch.setattr(client, "create_dataset", lambda **kwargs: None)

    def create_item(**kwargs):
        return DatasetItem(
            id=kwargs["id"],
            status="ACTIVE",
            input=kwargs["input"],
            expected_output=kwargs["expected_output"],
            metadata=kwargs["metadata"],
            dataset_id="dataset-test",
            dataset_name=kwargs["dataset_name"],
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            media_references=[],
        )

    def create_link(**kwargs):
        links.append(kwargs)
        return SimpleNamespace(dataset_run_id="dataset-run-test")

    monkeypatch.setattr(client, "create_dataset_item", create_item)
    monkeypatch.setattr(client.api.dataset_run_items, "create", create_link)
    monkeypatch.setattr(client, "create_score", lambda **kwargs: scores.append(kwargs))
    monkeypatch.setattr(client, "_get_project_id", lambda: "project-test")
    monkeypatch.setattr("rag_lab.runtime.Telemetry", lambda *args: telemetry)
    monkeypatch.setattr("rag_lab.runtime.Models", lambda *args: provider)
    monkeypatch.setattr("rag_lab.experiments.Judge", FakeJudge)
    examples = [
        Example(
            id=f"q{i}",
            question=f"RAG {i}?",
            reference="6 недель",
            reference_sources=["courses.md"],
            reviewed=True,
        )
        for i in range(2)
    ]
    try:
        directory, summary = run_evaluation(settings, RagConfig(), examples, tmp_path / "runs")
        assert summary["failed_examples"] == 0
        assert len(links) == 2
        assert len({link["trace_id"] for link in links}) == 2
        assert (
            sum(
                score["name"] == "faithfulness" and score.get("trace_id") is not None
                for score in scores
            )
            == 2
        )
        assert any(
            score["name"] == "faithfulness" and score.get("dataset_run_id") == "dataset-run-test"
            for score in scores
        )
        spans = exporter.get_finished_spans()
        assert sum(span.name == "rag-answer" for span in spans) == 2
        assert all(span.parent for span in spans if span.name == "rag-answer")
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["langfuse_url"].endswith("/datasets/dataset-test/runs/dataset-run-test")
        calls_before = len(provider.chat_calls)
        receipt = publish_saved_run(client, directory)
        assert len(provider.chat_calls) == calls_before
        assert receipt["examples"] == 2
        assert receipt["item_scores"] == 10
        assert receipt["run_scores"] == 5
        assert json.loads((directory / "langfuse.json").read_text(encoding="utf-8")) == receipt
    finally:
        client.shutdown()


def test_langfuse_preserves_all_metrics_and_error_statuses():
    scores = {
        "faithfulness": 0.8,
        "context_precision": 0.9,
        "context_recall": 0.7,
        "answer_relevancy": 0.6,
        "source_recall": 1.0,
        "source_mrr": 1.0,
    }
    evaluations = item_evaluations(
        output={"scores": scores, "errors": {}, "skipped": {}, "answer": {"latency_ms": 123}}
    )
    exported = {score.name: score.value for score in evaluations}
    assert all(exported[name] == value for name, value in scores.items())
    assert exported["latency_ms"] == 123 and exported["evaluation_error"] == 0
    failures = item_evaluations(
        output={
            "scores": {},
            "errors": {"faithfulness": "timeout"},
            "skipped": {"context_recall": "unanswerable"},
        }
    )
    exported = {score.name: score.value for score in failures}
    assert "faithfulness" not in exported
    assert exported["faithfulness_status"] == "error"
    assert exported["context_recall_status"] == "skipped"

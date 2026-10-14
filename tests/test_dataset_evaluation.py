import json
from pathlib import Path

import pytest

from rag_lab.dataset import Example, read_dataset, validate_sources
from rag_lab.documents import load_documents
from rag_lab.evaluation import aggregate, evaluate_example, retrieval_scores
from rag_lab.experiments import grid_configs


def test_golden_dataset_and_grid():
    examples = read_dataset(Path("data/eval/golden.jsonl"))
    validate_sources(examples, load_documents(Path("data/knowledge")))
    assert len(examples) == 20
    assert {item.kind for item in examples} == {
        "single_hop",
        "multi_hop",
        "conversational",
        "unanswerable",
    }
    assert {item.id for item in examples if item.split == "dev"}.isdisjoint(
        {item.id for item in examples if item.split == "test"}
    )
    assert len(grid_configs(Path("configs/experiments.yaml"))) == 8


def test_unreviewed_and_duplicate_dataset_rejected(tmp_path):
    item = {"id": "1", "question": "q", "reference": "a", "reference_sources": ["a.md"]}
    path = tmp_path / "test.jsonl"
    path.write_text(json.dumps(item), encoding="utf-8")
    with pytest.raises(ValueError, match="Непроверенный"):
        read_dataset(path)
    item["reviewed"] = True
    path.write_text(json.dumps(item) + "\n" + json.dumps(item), encoding="utf-8")
    with pytest.raises(ValueError, match="Повтор"):
        read_dataset(path)


def test_metrics_do_not_double_count_sources_or_turn_errors_into_zero():
    assert retrieval_scores(["a", "a", "b"], ["a", "c"]) == {
        "source_recall": 0.5,
        "source_mrr": 1.0,
    }
    rows = [
        {
            "kind": "single_hop",
            "scores": {"faithfulness": 0.8},
            "errors": {},
            "skipped": {},
            "answer": {"latency_ms": 10},
        },
        {"kind": "single_hop", "scores": {}, "errors": {"faithfulness": "timeout"}, "skipped": {}},
    ]
    metric = aggregate(rows)["metrics"]["faithfulness"]
    assert metric == {"mean": 0.8, "valid": 1, "errors": 1, "skipped": 0}


def test_pipeline_errors_are_visible():
    class BrokenRag:
        def answer(self, *args):
            raise RuntimeError("offline")

    item = Example(id="1", question="q", reference="a", reference_sources=["a"])
    result = evaluate_example(BrokenRag(), None, item)
    assert "offline" in result["errors"]["pipeline"]
    assert result["scores"] == {}


@pytest.mark.parametrize("override", [False, True])
def test_cli_uses_config_from_environment_with_explicit_override(tmp_path, monkeypatch, override):
    from typer.testing import CliRunner

    from rag_lab.cli import app
    from rag_lab.config import RagConfig

    config = tmp_path / "deployed.yaml"
    config.write_text(RagConfig(name="deployed").model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("RAG_CONFIG", str(config))
    args = ["evaluate", "--dry-run"]
    if override:
        args += ["--config", "configs/baseline.yaml"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 0, result.output
    assert ("baseline" if override else "deployed") in result.output

"""Общие evaluators Langfuse и публикация ранее рассчитанных оценок без LLM."""

import json
import math
from pathlib import Path

from langfuse import Evaluation

from rag_lab.dataset import dataset_fingerprint, publish_dataset, read_dataset
from rag_lab.evaluation import aggregate
from rag_lab.index import write_json


def item_evaluations(*, output, **kwargs):
    result = [
        Evaluation(name=name, value=value, data_type="NUMERIC")
        for name, value in output["scores"].items()
        if value is not None and math.isfinite(value)
    ]
    for state, values in (("error", output["errors"]), ("skipped", output["skipped"])):
        result.extend(
            Evaluation(name=f"{name}_status", value=state, data_type="CATEGORICAL", comment=reason)
            for name, reason in values.items()
        )
    result.append(Evaluation(name="evaluation_error", value=float(bool(output["errors"]))))
    if "answer" in output:
        result.append(Evaluation(name="latency_ms", value=output["answer"]["latency_ms"]))
    return result


def run_evaluations(*, item_results, **kwargs):
    summary = aggregate([item.output for item in item_results])
    result = [
        Evaluation(
            name=name,
            value=metric["mean"],
            data_type="NUMERIC",
            comment=f"Mean of {metric['valid']} valid scores; errors={metric['errors']}, skipped={metric['skipped']}",
            metadata={
                "valid": metric["valid"],
                "errors": metric["errors"],
                "skipped": metric["skipped"],
            },
        )
        for name, metric in summary["metrics"].items()
        if metric["mean"] is not None
    ]
    result.append(Evaluation(name="failed_examples", value=summary["failed_examples"]))
    if summary["latency_ms_mean"] is not None:
        result.append(Evaluation(name="latency_ms", value=summary["latency_ms_mean"]))
    return result


def check_experiment_result(result, expected_count):
    if len(result.item_results) != expected_count or not result.dataset_run_id:
        raise RuntimeError("Langfuse не связал все примеры с прогоном")
    for item in result.item_results:
        expected = {score.name for score in item_evaluations(output=item.output)}
        actual = {score.name for score in item.evaluations}
        if not item.dataset_run_id or not item.trace_id or not expected <= actual:
            raise RuntimeError("Langfuse не записал все оценки примера")
    expected = {score.name for score in run_evaluations(item_results=result.item_results)}
    if not expected <= {score.name for score in result.run_evaluations}:
        raise RuntimeError("Langfuse не записал сводные оценки прогона")


def publish_saved_run(client, directory: Path, *, allow_unreviewed=False) -> dict:
    """Публикует frozen dataset и готовые ответы/scores; модели не вызываются."""
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    examples = read_dataset(directory / "dataset.jsonl", require_reviewed=not allow_unreviewed)
    if dataset_fingerprint(examples) != manifest["dataset_hash"]:
        raise ValueError("Датасет не соответствует манифесту прогона")
    rows = [
        json.loads(line)
        for line in (directory / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    lookup = {row["id"]: row for row in rows}
    if len(lookup) != len(rows) or set(lookup) != {item.id for item in examples}:
        raise ValueError("Результаты должны содержать ровно одну строку на каждый пример")
    for example in examples:
        row = lookup[example.id]
        if row["question"] != example.question or row["reference"] != example.reference:
            raise ValueError("Вопрос/эталон в результатах не соответствует датасету")
    dataset_name = "rag-" + manifest["dataset_hash"][:16]
    remote = publish_dataset(client, examples, dataset_name)
    run_name = manifest["run_name"] + "-import"

    def task(*, item, **kwargs):
        return lookup[item.metadata["id"]]

    result = client.run_experiment(
        name=manifest["config"]["name"],
        run_name=run_name,
        description="Imported RAG answers and Ragas scores; no model calls during import.",
        data=list(remote.values()),
        task=task,
        evaluators=[item_evaluations],
        run_evaluators=[run_evaluations],
        max_concurrency=1,
        metadata={
            "dataset_hash": manifest["dataset_hash"],
            "evaluation_mode": manifest.get("evaluation_mode", "unknown"),
            "judge_model": manifest["judge_model"],
            "config": json.dumps(manifest["config"], ensure_ascii=False),
            "imported_from": str(directory),
            "metric_retry_count": str(len(manifest.get("metric_retries", []))),
        },
    )
    check_experiment_result(result, len(examples))
    receipt = {
        "dataset_name": dataset_name,
        "run_name": run_name,
        "dataset_run_id": result.dataset_run_id,
        "url": result.dataset_run_url,
        "examples": len(examples),
        "item_scores": sum(len(item.evaluations) for item in result.item_results),
        "run_scores": len(result.run_evaluations),
        "trace_ids": [item.trace_id for item in result.item_results],
        "status": "submitted",
    }
    write_json(directory / "langfuse.json", receipt)
    manifest["langfuse_url"] = result.dataset_run_url
    manifest["langfuse_dataset"] = dataset_name
    write_json(manifest_path, manifest)
    return receipt

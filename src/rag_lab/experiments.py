"""Воспроизводимый прогон: конфигурация + frozen dataset + corpus + результаты.

LLM даже при temperature=0 не гарантирует побитовую воспроизводимость.
Манифест фиксирует входы, а повторные запуски позволяют оценить разброс.
"""

import asyncio
import csv
import itertools
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from uuid import uuid4

import yaml

from rag_lab.config import RagConfig, digest
from rag_lab.dataset import dataset_fingerprint, publish_dataset
from rag_lab.evaluation import Judge, aggregate, evaluate_example, to_ragas_sample
from rag_lab.index import write_json
from rag_lab.langfuse_export import check_experiment_result, item_evaluations, run_evaluations
from rag_lab.pipeline import PROMPT_VERSION, SYSTEM_PROMPT
from rag_lab.progress import progress
from rag_lab.runtime import Runtime


def grid_configs(path: Path) -> list[RagConfig]:
    grid = yaml.safe_load(path.read_text(encoding="utf-8"))
    expected = {"chunk_strategy", "chunk_size", "chunk_overlap", "embedding_model", "top_k"}
    if set(grid) != expected or any(not isinstance(v, list) or not v for v in grid.values()):
        raise ValueError(f"Сетка должна содержать непустые списки: {sorted(expected)}")
    configs = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid.keys(), values, strict=True))
        config = RagConfig(name="variant-" + digest(params)[:8], **params)
        if config not in configs:
            configs.append(config)
    return configs


def run_evaluation(settings, config, examples, output_root: Path) -> tuple[Path, dict]:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
    run_name = f"{config.name}-{run_id}"
    directory = output_root / run_name
    directory.mkdir(parents=True, exist_ok=False)
    progress("evaluation.start", run=run_name, examples=len(examples), output=directory)
    rows = []
    manifest = {
        "run_name": run_name,
        "status": "running",
        "config": config.model_dump(),
        "dataset_hash": dataset_fingerprint(examples),
        "examples": len(examples),
        "dataset_reviewed": all(item.reviewed for item in examples),
        "evaluation_mode": "reviewed"
        if all(item.reviewed for item in examples)
        else "draft_preview",
        "chat_model": settings.chat_model,
        "judge_model": settings.judge_model,
        "judge_max_output_tokens": settings.judge_max_output_tokens,
        "judge_embedding_model": settings.judge_embedding_model,
        "chat_endpoint_hash": digest(settings.openai_base_url),
        "judge_endpoint_hash": digest(settings.judge_base_url or settings.openai_base_url),
        "history_turns": settings.history_turns,
        "max_output_tokens": settings.max_output_tokens,
        "temperature": 0,
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": digest(SYSTEM_PROMPT),
        "python": platform.python_version(),
        "packages": {
            package: version(package)
            for package in ("ragas", "langfuse", "qdrant-client", "openai")
        },
    }
    write_json(directory / "manifest.json", manifest)
    (directory / "dataset.jsonl").write_text(
        "\n".join(item.model_dump_json() for item in examples) + "\n", encoding="utf-8"
    )
    try:
        with Runtime(settings, config) as runtime:
            progress("evaluation.index.start", config=config.name)
            index_manifest = runtime.index.build(runtime.provider)
            progress("evaluation.index.done", result=index_manifest)
            manifest["index"] = index_manifest
            write_json(
                directory / "corpus.json",
                [{"source": d.source, "text": d.text} for d in runtime.index.documents],
            )
            write_json(directory / "manifest.json", manifest)
            judge = Judge(settings)

            def process(item):
                progress("evaluation.item.start", id=item.id, total=len(examples))
                row = evaluate_example(runtime.rag, judge, item)
                rows.append(row)
                # JSONL сохраняется сразу, а не после окончания всего эксперимента.
                with (directory / "results.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                if "answer" in row:
                    with (directory / "ragas_evaluation.jsonl").open("a", encoding="utf-8") as file:
                        file.write(to_ragas_sample(item, row["answer"]).model_dump_json() + "\n")
                progress("evaluation.item.done", id=item.id, completed=len(rows), total=len(examples), scores=row["scores"], errors=row["errors"])
                return row

            client = runtime.telemetry.client
            if client:
                dataset_name = "rag-" + manifest["dataset_hash"][:16]
                remote = publish_dataset(client, examples, dataset_name)
                lookup = {item.id: item for item in examples}

                async def task(*, item, **kwargs):
                    # Langfuse вызывает task внутри своего event loop. Наш
                    # синхронный process использует asyncio.run для Ragas,
                    # поэтому запускаем его в worker thread. to_thread также
                    # переносит OTel context: дочерние spans сохраняют trace.
                    return await asyncio.to_thread(process, lookup[item.metadata["id"]])

                result = client.run_experiment(
                    name=config.name,
                    run_name=run_name,
                    data=list(remote.values()),
                    task=task,
                    evaluators=[item_evaluations],
                    run_evaluators=[run_evaluations],
                    # SQLite/local Qdrant и список rows не разделяем между потоками.
                    max_concurrency=1,
                    metadata={
                        "config": config.model_dump_json(),
                        "dataset_hash": manifest["dataset_hash"],
                        "corpus_hash": runtime.index.corpus_hash,
                        "judge_model": settings.judge_model,
                        "evaluation_mode": manifest["evaluation_mode"],
                        "judge_max_output_tokens": str(settings.judge_max_output_tokens),
                    },
                )
                # SDK может залогировать ошибку привязки dataset run и продолжить.
                # Не объявляем такую публикацию полностью успешной.
                check_experiment_result(result, len(examples))
                manifest["langfuse_url"] = result.dataset_run_url
                manifest["langfuse_dataset"] = dataset_name
                progress("evaluation.langfuse.done", dataset=dataset_name, run=run_name, url=result.dataset_run_url)
            else:
                progress("evaluation.langfuse.skipped", reason="Langfuse отключён; результаты будут сохранены локально")
                for item in examples:
                    process(item)
            if len(rows) != len(examples):
                raise RuntimeError("Не все примеры обработаны; проверьте ошибки experiment runner")
        summary = aggregate(rows)
        summary["dataset_reviewed"] = manifest["dataset_reviewed"]
        manifest["status"] = "completed_with_errors" if summary["failed_examples"] else "completed"
        write_json(directory / "summary.json", summary)
        write_report(directory, rows, summary)
        progress("evaluation.done", status=manifest["status"], summary=summary, report=directory / "report.md")
        return directory, summary
    except BaseException as error:
        manifest["status"] = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        progress("evaluation.error", run=run_name, error=manifest["error"], output=directory)
        raise
    finally:
        write_json(directory / "manifest.json", manifest)


def write_report(directory: Path, rows: list[dict], summary: dict):
    names = list(summary["metrics"])
    with (directory / "scores.csv").open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "kind", *names, "latency_ms", "errors"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    **row["scores"],
                    "latency_ms": row.get("answer", {}).get("latency_ms"),
                    "errors": json.dumps(row["errors"], ensure_ascii=False),
                }
            )
    lines = [
        "# Результат эксперимента",
        "",
        f"Примеров: {len(rows)}. С ошибками: {summary['failed_examples']}.",
        "Эталоны проверены."
        if summary.get("dataset_reviewed")
        else "ПРОБНЫЙ ПРОГОН: эталоны не проверены; оценки предварительные.",
        "",
        "| Метрика | Среднее | Валидных | Ошибок | Пропусков |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metric in summary["metrics"].items():
        value = f"{metric['mean']:.4f}" if metric["mean"] is not None else "—"
        lines.append(
            f"| {name} | {value} | {metric['valid']} | {metric['errors']} | {metric['skipped']} |"
        )
    lines += [
        "",
        f"Средняя задержка RAG: {summary['latency_ms_mean']} мс (без индексации и судьи).",
        "",
        "Сравнивайте средние только вместе с количеством валидных оценок. Полные ответы и ошибки: results.jsonl.",
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def compare_runs(paths: list[Path], output: Path):
    """Таблица без автоматического объявления «победителя» по одной метрике."""
    lines = [
        "# Сравнение конфигураций",
        "",
        "| Вариант | Чанки | Эмбеддер | Faithfulness | Precision | Recall | Relevancy | Ошибки | мс |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    hashes = set()
    for path in paths:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
        hashes.add(manifest["dataset_hash"])
        config = manifest["config"]
        metrics = []
        for name in ("faithfulness", "context_precision", "context_recall", "answer_relevancy"):
            metric = summary["metrics"][name]
            metrics.append(
                f"{metric['mean']:.3f} (n={metric['valid']})" if metric["mean"] is not None else "—"
            )
        lines.append(
            f"| {config['name']} | {config['chunk_strategy']} {config['chunk_size']}/{config['chunk_overlap']} | {config['embedding_model']} | {' | '.join(metrics)} | {summary['failed_examples']} | {summary['latency_ms_mean']} |"
        )
    if len(hashes) > 1:
        raise ValueError("Нельзя сравнивать прогоны с разными датасетами")
    lines += [
        "",
        "Размер/overlap указаны в символах. Меньшая задержка лучше; n — число валидных оценок.",
        "Не делайте статистических выводов по нескольким примерам. Посмотрите ошибки по категориям и повторите прогон.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

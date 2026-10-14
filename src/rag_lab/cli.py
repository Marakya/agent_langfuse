"""Команды индексации, чата и оценки. Запуск: uv run rag-chat --help."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from rag_lab.config import RagConfig, Settings
from rag_lab.dataset import (
    dataset_fingerprint,
    publish_dataset,
    read_dataset,
    validate_sources,
)
from rag_lab.documents import corpus_hash, load_documents, split_document
from rag_lab.experiments import compare_runs, grid_configs, run_evaluation
from rag_lab.index import write_json
from rag_lab.memory import Memory
from rag_lab.progress import progress
from rag_lab.runtime import Runtime
from rag_lab.telemetry import Telemetry

app = typer.Typer(no_args_is_help=True, help="RAG: индексация, чат, датасет, оценка.")
ConfigOption = Annotated[
    Path | None, typer.Option(help="YAML конфигурации RAG; по умолчанию RAG_CONFIG")
]


@app.command()
def ingest(config: ConfigOption = None, prepare_only: bool = False):
    """Загрузить документы, создать чанки и идемпотентно заполнить Qdrant."""
    settings = Settings()
    rag_config = RagConfig.read(config or settings.rag_config)
    if prepare_only:
        documents = load_documents(settings.documents_dir)
        chunks = [asdict(chunk) for doc in documents for chunk in split_document(doc, rag_config)]
        output = settings.index_dir / "prepared"
        write_json(
            output / "parents.json",
            {doc.parent_id: asdict(doc) for doc in documents if doc.parent_id},
        )
        write_json(output / "chunks.json", chunks)
        summary = {
            "status": "prepared_not_indexed",
            "corpus_hash": corpus_hash(documents),
            "config": rag_config.model_dump(),
            "documents": len({doc.source for doc in documents}),
            "parent_pages": sum(doc.page is not None for doc in documents),
            "chunks": len(chunks),
            "output": str(output),
        }
        write_json(output / "manifest.json", summary)
        typer.echo(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    with Runtime(settings, rag_config) as runtime:
        typer.echo(json.dumps(runtime.index.build(runtime.provider), ensure_ascii=False, indent=2))


@app.command()
def chat(config: ConfigOption = None, session: str | None = None):
    """Чат в терминале. Передайте --session UUID для продолжения после перезапуска."""
    settings = Settings()
    memory = Memory(settings.database_url)
    try:
        session_id = session or memory.create()
        typer.echo(f"Диалог: {session_id}. Для выхода: /exit")
        _, history = memory.history(session_id)
        for message in history:
            typer.echo(f"{message['role']}: {message['content']}")
        with Runtime(settings, RagConfig.read(config or settings.rag_config)) as runtime:
            while True:
                question = typer.prompt("Вы")
                if question.strip() == "/exit":
                    break
                answer = runtime.rag.chat(memory, session_id, question)
                typer.echo(answer.response)
                for n, context in enumerate(answer.contexts, 1):
                    page = f" · стр. PDF {context['page']}" if context.get("page") else ""
                    typer.echo(
                        f"[{n}] {context['source']}{page} ({context['start']}:{context['end']})"
                    )
    finally:
        memory.close()


@app.command("dataset-check")
def dataset_check(dataset: Path = Path("data/eval/golden.jsonl")):
    """Проверить JSONL, уникальность вопросов, наличие источников и точность цитат."""
    examples = read_dataset(dataset)
    validate_sources(examples, load_documents(Settings().documents_dir))
    typer.echo(f"OK: {len(examples)} примеров; SHA-256={dataset_fingerprint(examples)}")


@app.command("history")
def show_history(session: Annotated[str, typer.Option(help="ID диалога")]):
    """Показать сохранённую историю без запуска RAG и обращения к LLM."""
    progress("history.start", session_id=session)
    memory = Memory(Settings().database_url)
    try:
        revision, messages = memory.history(session)
        for message in messages:
            typer.echo(f"{message['role']}: {message['content']}")
        progress("history.done", session_id=session, messages=len(messages), revision=revision)
    except Exception as error:
        progress("history.error", session_id=session, error=str(error))
        raise typer.Exit(1) from error
    finally:
        memory.close()


@app.command("dataset-generate")
def dataset_generate(
    output: Path = Path("data/eval/draft.jsonl"),
    testset_size: Annotated[int, typer.Option(min=1, max=1000)] = 10,
    max_documents: Annotated[int, typer.Option(min=0, help="0 = все подходящие страницы")] = 0,
):
    """ПЛАТНЫЕ запросы: создать draft через Ragas TestsetGenerator."""
    from rag_lab.ragas_generation import generate_ragas_dataset

    settings = Settings()
    progress("generation.read.start", documents_dir=settings.documents_dir, output=output)
    try:
        documents = load_documents(settings.documents_dir)
        progress("generation.read.done", documents=len(documents))
        manifest = generate_ragas_dataset(
            documents, settings, output, testset_size, max_documents
        )
    except Exception as error:
        progress("generation.error", error=str(error))
        raise
    typer.echo(json.dumps(manifest, ensure_ascii=False, indent=2))
    typer.echo("Черновик Ragas: проверьте эталоны перед установкой reviewed=true.")


@app.command("dataset-push")
def dataset_push(dataset: Path = Path("data/eval/golden.jsonl"), allow_unreviewed: bool = False):
    """Опубликовать проверенный датасет в настроенный проект Langfuse."""
    settings = Settings()
    progress("dataset.read.start", path=dataset)
    examples = read_dataset(dataset, require_reviewed=not allow_unreviewed)
    progress("dataset.read.done", examples=len(examples))
    validate_sources(examples, load_documents(settings.documents_dir))
    progress("dataset.validation.done", examples=len(examples))
    telemetry = Telemetry(settings)
    if not telemetry.client:
        raise typer.BadParameter("Установите LANGFUSE_ENABLED=true и ключи проекта")
    name = "rag-" + dataset_fingerprint(examples)[:16]
    try:
        publish_dataset(telemetry.client, examples, name)
    except Exception as error:
        progress("dataset.publish.error", dataset=name, error=str(error))
        raise
    typer.echo(f"Langfuse dataset: {name}")


@app.command("evaluation-push")
def evaluation_push(run: Path, allow_unreviewed: bool = False):
    """Опубликовать сохранённый прогон и все метрики в Langfuse без вызовов LLM."""
    from rag_lab.langfuse_export import publish_saved_run

    telemetry = Telemetry(Settings())
    if not telemetry.client:
        raise typer.BadParameter("Установите LANGFUSE_ENABLED=true и ключи проекта")
    try:
        receipt = publish_saved_run(telemetry.client, run, allow_unreviewed=allow_unreviewed)
        typer.echo(json.dumps(receipt, ensure_ascii=False, indent=2))
    finally:
        telemetry.flush()


def select_examples(dataset, split, limit, allow_unreviewed=False):
    if split not in {"dev", "test"}:
        raise typer.BadParameter("split должен быть dev или test")
    examples = read_dataset(dataset, split=split, require_reviewed=not allow_unreviewed)
    validate_sources(examples, load_documents(Settings().documents_dir))
    return examples[:limit] if limit else examples


@app.command()
def evaluate(
    config: ConfigOption = None,
    dataset: Path = Path("data/eval/golden.jsonl"),
    split: str = "dev",
    limit: Annotated[
        int, typer.Option(min=0, help="0 = вся выбранная часть; по умолчанию 3 примера")
    ] = 3,
    output: Path = Path("artifacts/runs"),
    dry_run: bool = False,
    allow_unreviewed: Annotated[
        bool, typer.Option(help="Пробная оценка непроверенного draft")
    ] = False,
):
    """ПЛАТНЫЕ запросы: оценить одну конфигурацию; --dry-run показывает план."""
    progress("evaluation.read.start", path=dataset, split=split, limit=limit)
    examples = select_examples(dataset, split, limit, allow_unreviewed)
    progress("evaluation.read.done", examples=len(examples), split=split)
    settings = Settings()
    selected = RagConfig.read(config or settings.rag_config)
    typer.echo(f"Примеров: {len(examples)}, конфигурация: {selected.model_dump()}")
    if dry_run:
        return
    directory, summary = run_evaluation(settings, selected, examples, output)
    typer.echo(f"Результат: {directory / 'report.md'}")
    if summary["failed_examples"]:
        typer.echo(
            "Есть ошибки. Они сохранены в results.jsonl; не трактуйте их как нулевые оценки."
        )
        raise typer.Exit(1)


@app.command()
def experiment(
    grid: Path = Path("configs/experiments.yaml"),
    dataset: Path = Path("data/eval/golden.jsonl"),
    split: str = "dev",
    limit: Annotated[int, typer.Option(min=0)] = 3,
    max_variants: Annotated[int, typer.Option(min=1)] = 2,
    output: Path = Path("artifacts/experiments"),
    dry_run: bool = False,
):
    """ПЛАТНЫЕ запросы: одинаковые вопросы для всех вариантов сетки."""
    examples = select_examples(dataset, split, limit)
    variants = grid_configs(grid)[:max_variants]
    typer.echo(
        f"Вариантов: {len(variants)}; примеров: {len(examples)}; ответов RAG: {len(variants) * len(examples)}."
    )
    typer.echo("Каждая метрика может вызвать судью несколько раз; индексация также расходует API.")
    for config in variants:
        typer.echo(config.model_dump_json())
    if dry_run:
        return
    from datetime import datetime, timezone
    from uuid import uuid4

    # Отдельная папка каждого сравнения: прошлые выводы не перезаписываются.
    output = output / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:6]
    )
    paths, failures = [], 0
    for config in variants:
        directory, summary = run_evaluation(Settings(), config, examples, output)
        paths.append(directory)
        failures += summary["failed_examples"]
        compare_runs(paths, output / "comparison.md")
    typer.echo(f"Сравнение: {output / 'comparison.md'}")
    if failures:
        raise typer.Exit(1)

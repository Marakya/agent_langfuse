"""Генерация эталонов средствами Ragas, независимо от ответов нашего RAG."""

import asyncio
import json
import os
from importlib.metadata import version
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from rag_lab.config import Settings, digest
from rag_lab.dataset import Example, validate_sources
from rag_lab.documents import Document, corpus_hash
from rag_lab.progress import progress


def examples_from_testset(testset, documents: list[Document]) -> list[Example]:
    """Адаптер Testset -> формат CLI с проверяемой привязкой к источникам."""
    examples, questions = [], set()
    for sample in testset.samples:
        row = sample.eval_sample
        contexts = row.reference_contexts or []
        if not row.user_input or not row.reference or not contexts:
            raise ValueError("Ragas вернул пример без вопроса, эталона или контекста")
        sources = set()
        for context in contexts:
            matches = [doc.source for doc in documents if context.strip() and context in doc.text]
            if not matches:
                raise ValueError("Контекст Ragas не найден в исходных документах")
            sources.update(matches)
        key = row.user_input.casefold().strip()
        if key in questions:
            raise ValueError("Ragas вернул повторяющийся вопрос")
        questions.add(key)
        examples.append(
            Example(
                id=str(uuid5(NAMESPACE_URL, digest([sorted(sources), row.user_input]))),
                question=row.user_input,
                reference=row.reference,
                reference_sources=sorted(sources),
                reference_contexts=contexts,
                kind="single_hop",
                reviewed=False,
            )
        )
    if not examples:
        raise ValueError("Ragas вернул пустой датасет")
    validate_sources(examples, documents)
    return examples


def generate_ragas_dataset(
    documents: list[Document],
    settings: Settings,
    output: Path,
    testset_size: int = 10,
    max_documents: int = 0,
) -> dict:
    """Страницы -> NER Ragas -> сценарии -> вопросы и эталоны TestsetGenerator.

    Базовый набор single-hop проверяет ответы по одной странице. Граф строит
    Ragas; чанки поискового индекса и ответы оцениваемого бота не используются.
    """
    raw_path = output.with_suffix(".ragas.jsonl")
    manifest_path = output.with_suffix(".manifest.json")
    for path in (output, raw_path, manifest_path):
        if path.exists():
            raise ValueError(f"Файл уже существует: {path}. Выберите новое имя.")
    if testset_size < 1 or max_documents < 0:
        raise ValueError("testset_size должен быть > 0, max_documents >= 0")
    settings.require_llm()
    os.environ["RAGAS_DO_NOT_TRACK"] = str(settings.ragas_do_not_track).lower()
    import tiktoken
    from langchain_core.documents import Document as LCDocument
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings
    from ragas.run_config import RunConfig
    from ragas.testset import TestsetGenerator
    from ragas.testset.persona import Persona
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
    from ragas.testset.transforms.extractors.llm_based import NERExtractor

    #NERExtractor — извлекает сущности из текста, чтобы Ragas понимал, о чём документ.
    #SingleHopSpecificQuerySynthesizer — выбирает источник и параметры вопроса, затем генерирует вопрос и эталонный ответ.
    #TestsetGenerator — управляет всем процессом, от формирования графа до создания тестовых примеров.
    
    encoder = tiktoken.get_encoding("cl100k_base")
    eligible = [d for d in documents if len(encoder.encode(d.text, disallowed_special=())) > 100]
    selected = eligible[:max_documents] if max_documents else eligible
    if not selected:
        raise ValueError("Для генерации нужны страницы с текстом длиннее 100 токенов")
    progress("generation.pages.done", input_pages=len(documents), eligible_pages=len(eligible), selected_pages=len(selected))
    llm = ChatOpenAI(
        model=settings.chat_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        temperature=0,
        timeout=settings.request_timeout,
        max_retries=2,
    )
    embeddings = OpenAIEmbeddings(
        model=settings.judge_embedding_model,
        api_key=settings.judge_api_key or settings.openai_api_key,
        base_url=settings.judge_base_url or settings.openai_base_url,
        request_timeout=settings.request_timeout,
    )
    generator = TestsetGenerator.from_langchain(
        llm,
        embeddings,
        llm_context="Вопрос и эталонный ответ пиши по-русски. Ответ только по контексту документа.",
    )
    generator.persona_list = [
        Persona(
            name="Инженер-конструктор",
            role_description="Изучает правила оформления конструкторской документации по ГОСТ.",
        )
    ]
    synthesizer = SingleHopSpecificQuerySynthesizer(llm=generator.llm)
    progress("generation.prompts.start", language="russian", model=settings.chat_model)
    prompts = asyncio.run(synthesizer.adapt_prompts("russian", llm=generator.llm))
    synthesizer.set_prompts(**prompts)
    progress("generation.prompts.done", language="russian")
    progress("generation.ragas.start", requested_size=testset_size, pages=len(selected))
    testset = generator.generate_with_langchain_docs(
        [
            LCDocument(page_content=d.text, metadata={"source": d.source, "page": d.page})
            for d in selected
        ],
        testset_size=testset_size,
        transforms=[NERExtractor(llm=generator.llm)],
        query_distribution=[(synthesizer, 1.0)],
        run_config=RunConfig(timeout=settings.request_timeout, max_workers=2, max_retries=2),
        raise_exceptions=True,
    )
    examples = examples_from_testset(testset, selected)
    progress("generation.validation.done", examples=len(examples))
    manifest = {
        "generator": "ragas.testset.TestsetGenerator",
        "ragas_version": version("ragas"),
        "status": "draft_unreviewed",
        "requested_size": testset_size,
        "actual_size": len(examples),
        "query_distribution": {synthesizer.name: 1.0},
        "language": "russian",
        "model": settings.chat_model,
        "input_pages": len(documents),
        "eligible_pages": len(eligible),
        "selected_pages": [{"source": d.source, "page": d.page} for d in selected],
        "corpus_hash": corpus_hash(documents),
        "selected_corpus_hash": corpus_hash(selected),
        "dataset": str(output),
        "ragas_testset": str(raw_path),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    # Сохраняем родной Testset без подмены reference_contexts найденными чанками.
    with raw_path.open("x", encoding="utf-8") as target:
        for row in testset.to_list():
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    with output.open("x", encoding="utf-8") as target:
        for item in examples:
            target.write(item.model_dump_json() + "\n")
    with manifest_path.open("x", encoding="utf-8") as target:
        json.dump(manifest, target, ensure_ascii=False, indent=2)
    progress("generation.done", examples=len(examples), dataset=output, ragas_testset=raw_path, manifest=manifest_path)
    return manifest

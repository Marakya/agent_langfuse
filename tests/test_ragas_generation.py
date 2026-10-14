import json

import pytest
from ragas import EvaluationDataset, SingleTurnSample
from ragas.testset.synthesizers.testset_schema import Testset as RagasTestset
from ragas.testset.synthesizers.testset_schema import TestsetSample as RagasTestsetSample
from typer.testing import CliRunner

from rag_lab.cli import app
from rag_lab.dataset import Example, read_dataset
from rag_lab.documents import Document
from rag_lab.evaluation import to_ragas_sample
from rag_lab.ragas_generation import examples_from_testset, generate_ragas_dataset


def native_testset(context="Курс длится 6 недель."):
    return RagasTestset(
        samples=[
            RagasTestsetSample(
                eval_sample=SingleTurnSample(
                    user_input="Сколько длится курс?",
                    reference="6 недель.",
                    reference_contexts=[context],
                ),
                synthesizer_name="single_hop_specific_query_synthesizer",
            )
        ]
    )


def test_native_ragas_testset_keeps_reference_and_source():
    testset = native_testset()
    docs = [Document("book.pdf", "Курс длится 6 недель.", page=2)]
    examples = examples_from_testset(testset, docs)
    assert examples[0].reference == testset.samples[0].eval_sample.reference
    assert examples[0].reference_sources == ["book.pdf"]
    assert examples[0].reference_contexts == [docs[0].text]
    assert not examples[0].reviewed
    assert examples_from_testset(testset, docs)[0].id == examples[0].id


def test_ragas_generated_context_must_exist_in_corpus():
    with pytest.raises(ValueError, match="Контекст Ragas"):
        examples_from_testset(native_testset(), [Document("wrong.pdf", "Другая страница")])


def test_ragas_duplicate_questions_rejected():
    testset = native_testset()
    testset.samples.append(testset.samples[0])
    with pytest.raises(ValueError, match="повторяющийся"):
        examples_from_testset(testset, [Document("book.pdf", "Курс длится 6 недель.")])


def test_evaluation_uses_retrieved_context_not_reference(tmp_path):
    item = Example(
        id="q",
        question="Вопрос?",
        reference="Эталон",
        reference_sources=["book.pdf"],
        reference_contexts=["Эталонная страница"],
    )
    sample = to_ragas_sample(
        item, {"response": "Ответ бота", "contexts": [{"text": "Найденная страница"}]}
    )
    assert sample.retrieved_contexts == ["Найденная страница"]
    assert sample.reference_contexts == ["Эталонная страница"]
    output = tmp_path / "evaluation.jsonl"
    EvaluationDataset(samples=[sample]).to_jsonl(output)
    loaded = EvaluationDataset.from_jsonl(output)
    assert loaded.samples[0].response == "Ответ бота"


def test_generation_invokes_ragas_and_exports_native_testset(settings, tmp_path, monkeypatch):
    from ragas.testset import TestsetGenerator
    from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer

    text = "Курс длится 6 недель. " * 100
    documents = [Document("book.pdf", text, page=2)]
    calls = []

    async def adapt(self, *args, **kwargs):
        return self.get_prompts()

    def generate(self, docs, **kwargs):
        calls.append((docs, kwargs))
        return native_testset(text)

    monkeypatch.setattr(SingleHopSpecificQuerySynthesizer, "adapt_prompts", adapt)
    monkeypatch.setattr(TestsetGenerator, "generate_with_langchain_docs", generate)
    output = tmp_path / "draft.jsonl"
    manifest = generate_ragas_dataset(documents, settings, output, testset_size=1)
    assert calls[0][1]["testset_size"] == 1
    assert calls[0][0][0].metadata == {"source": "book.pdf", "page": 2}
    assert manifest["generator"] == "ragas.testset.TestsetGenerator"
    assert (
        json.loads(output.with_suffix(".ragas.jsonl").read_text(encoding="utf-8"))["reference"]
        == "6 недель."
    )
    assert read_dataset(output, require_reviewed=False)[0].reference_contexts == [text]
    with pytest.raises(ValueError, match="Непроверенный"):
        read_dataset(output)
    with pytest.raises(ValueError, match="уже существует"):
        generate_ragas_dataset(documents, settings, output)


def test_draft_evaluation_requires_explicit_flag(settings, tmp_path, monkeypatch):
    output = tmp_path / "draft.jsonl"
    item = Example(id="q", question="RAG?", reference="6 недель", reference_sources=["courses.md"])
    output.write_text(item.model_dump_json(), encoding="utf-8")
    monkeypatch.setenv("DOCUMENTS_DIR", str(settings.documents_dir))
    args = ["evaluate", "--dataset", str(output), "--dry-run"]
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "Непроверенный" in str(result.exception)
    assert CliRunner().invoke(app, args + ["--allow-unreviewed"]).exit_code == 0

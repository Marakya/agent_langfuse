"""Настоящий Ragas/Instructor/OpenAI SDK, подменён только HTTP-транспорт.

Тест проверяет вызовы всех четырёх алгоритмов, structured output и эмбеддинги.
Оценки ниже заданы транспортной заглушкой и не измеряют качество ответов RAG.
"""

import asyncio
import json
import re

import httpx
import openai

from rag_lab.dataset import Example
from rag_lab.evaluation import Judge


def schema_example(schema, definitions, field=""):
    """Создать валидный детерминированный ответ судьи по JSON-schema Instructor."""
    if "$ref" in schema:
        return schema_example(definitions[schema["$ref"].split("/")[-1]], definitions, field)
    if "anyOf" in schema:
        return schema_example(schema["anyOf"][0], definitions, field)
    kind = schema.get("type")
    if kind == "object":
        return {
            key: schema_example(value, definitions, key)
            for key, value in schema["properties"].items()
        }
    if kind == "array":
        return [schema_example(schema["items"], definitions, field)]
    if kind in {"integer", "number"}:
        return 0 if field == "noncommittal" else 1
    if kind == "boolean":
        return True
    if "enum" in schema:
        return schema["enum"][0]
    return "Курс RAG длится 6 недель."


def schema_from_messages(messages):
    # Instructor JSON mode помещает JSON-schema в текст system prompt.
    decoder = json.JSONDecoder()
    for message in messages:
        content = message.get("content") or ""
        for match in re.finditer(r"\{", content):
            try:
                value, _ = decoder.raw_decode(content[match.start() :])
            except ValueError:
                continue
            if isinstance(value, dict) and value.get("type") == "object" and "properties" in value:
                return value
    raise AssertionError("Instructor schema missing from mock request")


def test_real_ragas_algorithms_with_mock_http(settings, monkeypatch):
    requests = []

    def handle(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            return httpx.Response(
                200,
                json={
                    "object": "list",
                    "model": body["model"],
                    "data": [
                        {"object": "embedding", "index": i, "embedding": [1.0, 0.0, 0.0]}
                        for i in range(len(inputs))
                    ],
                    "usage": {"prompt_tokens": 3, "total_tokens": 3},
                },
            )
        if "tools" in body:
            function = body["tools"][0]["function"]
            schema = function["parameters"]
        else:
            function = None
            schema = (
                body["response_format"]["json_schema"]["schema"]
                if "json_schema" in body["response_format"]
                else schema_from_messages(body["messages"])
            )
        arguments = schema_example(schema, schema.get("$defs", {}))
        message = {"role": "assistant", "content": json.dumps(arguments)}
        if function:
            message = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-test",
                        "type": "function",
                        "function": {"name": function["name"], "arguments": json.dumps(arguments)},
                    }
                ],
            }
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "created": 1,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls" if function else "stop",
                        "message": message,
                    }
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
            },
        )

    original_init = openai.AsyncOpenAI.__init__

    def isolated_init(self, *args, **kwargs):
        kwargs["http_client"] = httpx.AsyncClient(transport=httpx.MockTransport(handle))
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(openai.AsyncOpenAI, "__init__", isolated_init)
    example = Example(
        id="test",
        question="Сколько длится RAG?",
        reference="6 недель",
        reference_sources=["courses.md"],
    )
    answer = {
        "response": "Курс RAG длится 6 недель.",
        "contexts": [{"text": "Курс RAG длится 6 недель."}],
    }
    scores, errors, skipped = asyncio.run(Judge(settings).score(example, answer))
    assert errors == {}, (errors, requests[:1])
    assert skipped == {}
    assert set(scores) == {
        "faithfulness",
        "context_precision",
        "context_recall",
        "answer_relevancy",
    }
    assert all(value > 0.99 for value in scores.values())
    assert any(path.endswith("/embeddings") for path, _ in requests)
    assert all(
        body.get("max_tokens") == settings.judge_max_output_tokens
        for path, body in requests
        if path.endswith("/chat/completions")
    )
    assert {body["model"] for path, body in requests if path.endswith("/embeddings")} == {
        settings.judge_embedding_model
    }

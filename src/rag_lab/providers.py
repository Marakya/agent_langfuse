"""Тонкая обёртка OpenAI-совместимого API: генерация и эмбеддинги разделены.

При смене эмбеддера меняется пространство векторов, даже если размерность
совпадает. Поэтому чужие векторы нельзя положить в существующую коллекцию.
"""

from openai import OpenAI

from rag_lab.config import Settings
from rag_lab.telemetry import Telemetry


class Models:
    def __init__(self, settings: Settings, telemetry: Telemetry):
        settings.require_llm()
        self.settings, self.telemetry = settings, telemetry
        self.client = OpenAI(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            timeout=settings.request_timeout,
            max_retries=2,
        )

    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        if not texts:
            return []
        vectors = []
        # Батчи ограничивают размер одного запроса к API эмбеддингов.
        for start in range(0, len(texts), 32):
            batch = texts[start : start + 32]
            with self.telemetry.span("embed", kind="embedding", model=model, input=batch) as span:
                response = self.client.embeddings.create(model=model, input=batch)
                ordered = sorted(response.data, key=lambda row: row.index)
                vectors.extend(row.embedding for row in ordered)
                span.update(
                    output={"vectors": len(ordered), "dimensions": len(ordered[0].embedding)},
                    usage_details={"input": response.usage.prompt_tokens},
                )
        if len(vectors) != len(texts):
            raise ValueError("API вернуло неверное количество векторов")
        return vectors

    def chat(self, messages: list[dict], *, purpose: str, json_mode=False) -> str:
        with self.telemetry.span(
            purpose,
            kind="generation",
            model=self.settings.chat_model,
            input=messages,
            model_parameters={"temperature": 0},
        ) as span:
            options = {"response_format": {"type": "json_object"}} if json_mode else {}
            response = self.client.chat.completions.create(
                model=self.settings.chat_model,
                messages=messages,
                temperature=0,
                max_tokens=self.settings.max_output_tokens,
                **options,
            )
            if response.choices[0].finish_reason == "length":
                raise ValueError("Ответ обрезан: увеличьте MAX_OUTPUT_TOKENS")
            content = response.choices[0].message.content
            if not content or not content.strip():
                raise ValueError("Модель вернула пустой ответ")
            usage = response.usage
            span.update(
                output=content,
                usage_details={"input": usage.prompt_tokens, "output": usage.completion_tokens}
                if usage
                else None,
            )
            return content.strip()

    def close(self):
        self.client.close()

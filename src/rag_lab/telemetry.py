"""Langfuse включается явно. Без ключей тот же pipeline работает локально."""

from contextlib import contextmanager

from rag_lab.config import Settings


class NullObservation:
    trace_id = None

    def update(self, **kwargs):
        pass


class Telemetry:
    def __init__(self, settings: Settings):
        self.client = None
        if settings.langfuse_enabled:
            if not settings.langfuse_public_key or not settings.langfuse_secret_key:
                raise ValueError("Для LANGFUSE_ENABLED=true нужны оба ключа Langfuse")
            from langfuse import Langfuse

            self.client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                base_url=settings.langfuse_base_url,
            )

    @contextmanager
    def span(self, name: str, *, kind="span", session_id=None, **kwargs):
        if self.client is None:
            yield NullObservation()
            return
        from langfuse import propagate_attributes

        with propagate_attributes(**({"session_id": session_id} if session_id else {})):
            with self.client.start_as_current_observation(
                name=name, as_type=kind, **kwargs
            ) as observation:
                yield observation

    def flush(self):
        if self.client:
            self.client.flush()

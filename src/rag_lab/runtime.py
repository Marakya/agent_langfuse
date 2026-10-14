"""Сборка зависимостей: клиенты создаются один раз и закрываются явно."""

from rag_lab.config import RagConfig, Settings
from rag_lab.index import Index, connect_qdrant
from rag_lab.pipeline import Rag
from rag_lab.providers import Models
from rag_lab.telemetry import Telemetry


class Runtime:
    def __init__(self, settings: Settings, config: RagConfig):
        self.telemetry = Telemetry(settings)
        self.provider = Models(settings, self.telemetry)
        self.qdrant = connect_qdrant(settings)
        self.index = Index(settings, config, self.qdrant)
        self.rag = Rag(settings, self.index, self.provider, self.telemetry)

    def close(self):
        self.telemetry.flush()
        self.provider.close()
        self.qdrant.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

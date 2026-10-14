"""Загрузка текстовых документов и разбиение на фрагменты с перекрытием."""

import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_lab.config import RagConfig, digest


@dataclass(frozen=True)
class Document:
    source: str  # Относительный путь — стабильный ID документа.
    text: str
    page: int | None = None  # Физический номер страницы PDF, начиная с 1.

    @property
    def parent_id(self) -> str | None:
        return digest({"source": self.source, "page": self.page}) if self.page else None


@dataclass(frozen=True)
class Chunk:
    source: str
    text: str
    start: int
    end: int
    page: int | None = None
    parent_id: str | None = None
    token_count: int | None = None


def load_documents(directory: Path) -> list[Document]:
    """Загрузить Markdown/TXT и PDF, где каждая страница — отдельный parent."""
    result = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() == ".pdf":
            source = path.relative_to(directory).as_posix()
            pages = []
            for number, page in enumerate(PdfReader(path).pages, 1):
                text = (page.extract_text() or "").replace("\r\n", "\n").strip()
                # Склеиваем только явно отмеченные мягким дефисом переносы.
                text = text.replace("\u00ad\n", "").replace("\u00ad", "")
                if text:
                    pages.append(Document(source, text, number))
                else:
                    warnings.warn(
                        f"{source}: страница {number} без текста; для скана нужен OCR", stacklevel=2
                    )
            if not pages:
                raise ValueError(f"{source}: PDF не содержит извлекаемого текста; нужен OCR")
            result.extend(pages)
        if path.is_file() and path.suffix.lower() in {".md", ".txt"}:
            text = path.read_text(encoding="utf-8-sig").replace("\r\n", "\n").strip()
            if text:
                result.append(Document(path.relative_to(directory).as_posix(), text))
    if not result:
        raise ValueError(f"В {directory} нет непустых .md/.txt/.pdf документов")
    return result


def corpus_hash(documents: list[Document]) -> str:
    return digest([asdict(doc) for doc in documents])


def split_document(doc: Document, config: RagConfig) -> list[Chunk]:
    """Две стратегии с одинаковым верхним ограничением размера.

    fixed: окно строго по символам; может разорвать слово/предложение.
    paragraph: ищем ближайшую границу абзаца, строки или предложения во
    второй половине окна. Если границы нет, используем фиксированный срез.
    overlap оставляет хвост прошлого чанка в следующем: на границе меньше
    потерь, но растут индекс, повторы в контексте и стоимость эмбеддингов.
    """
    if config.chunk_strategy == "recursive_tokens":
        return split_document_by_tokens(doc, config)
    chunks, start = [], 0
    while start < len(doc.text):
        end = min(start + config.chunk_size, len(doc.text))
        if config.chunk_strategy == "paragraph" and end < len(doc.text):
            lower = start + max(config.chunk_size // 2, config.chunk_overlap + 1)
            for separator in ("\n\n", "\n", ". ", " "):
                boundary = doc.text.rfind(separator, lower, end)
                if boundary >= lower:
                    end = boundary + len(separator)
                    break
        text = doc.text[start:end]
        if text.strip():
            chunks.append(Chunk(doc.source, text, start, end, doc.page, doc.parent_id))
        if end == len(doc.text):
            break
        # Даже при необычных параметрах окно обязано двигаться вперёд.
        start = max(start + 1, end - config.chunk_overlap)
    return chunks


def split_document_by_tokens(doc: Document, config: RagConfig) -> list[Chunk]:
    """Рекурсивные границы по тексту, размер/overlap по токенам tiktoken.

    start/end остаются символьными координатами в родительской странице.
    Overlap — целевой максимум: зависит от найденных разделителей.
    """
    encoder = tiktoken.get_encoding(config.token_encoding)
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name=config.token_encoding,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
        strip_whitespace=False,
        disallowed_special=(),
    )
    chunks = []
    previous_start, previous_end = -1, 0
    for text in splitter.split_text(doc.text):
        # LangChain add_start_index вычитает overlap из символьного offset;
        # при токенном overlap это разные единицы. Восстанавливаем по тексту.
        lower = max(previous_start + 1, previous_end - len(text) + 1, 0)
        start = doc.text.find(text, lower)
        while 0 <= start < previous_end:
            overlap = doc.text[start:previous_end]
            if len(encoder.encode(overlap, disallowed_special=())) <= config.chunk_overlap:
                break
            start = doc.text.find(text, start + 1)
        if start < 0:
            raise ValueError(f"Не найдены координаты чанка: {doc.source}, страница {doc.page}")
        end = start + len(text)
        previous_start, previous_end = start, end
        if not text.strip():
            continue
        token_count = len(encoder.encode(text, disallowed_special=()))
        if token_count > config.chunk_size:
            raise ValueError(f"Чанк превышает лимит {config.chunk_size} токенов")
        chunks.append(Chunk(doc.source, text, start, end, doc.page, doc.parent_id, token_count))
    return chunks

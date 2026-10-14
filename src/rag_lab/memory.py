"""Долговременное хранение диалогов в SQLite/PostgreSQL через SQLAlchemy.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    JSON,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    select,
    update,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    title: Mapped[str] = mapped_column(String(120))
    revision: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id"), index=True)
    role: Mapped[str] = mapped_column(String(16))
    content: Mapped[str] = mapped_column(Text)
    details: Mapped[dict] = mapped_column(JSON, default=dict)


class ConcurrentTurnError(RuntimeError):
    pass


class Memory:
    def __init__(self, url: str):
        parsed = make_url(url)
        if parsed.get_backend_name() == "sqlite":
            if parsed.database and parsed.database != ":memory:":
                Path(parsed.database).parent.mkdir(parents=True, exist_ok=True)
            self.engine = create_engine(
                url, connect_args={"check_same_thread": False, "timeout": 30}
            )
        else:
            self.engine = create_engine(url, pool_pre_ping=True)
        Base.metadata.create_all(self.engine)

    def create(self, title="Новый диалог") -> str:
        session_id = str(uuid4())
        with Session(self.engine) as db, db.begin():
            db.add(
                Conversation(
                    id=session_id, title=title, created_at=datetime.now(timezone.utc)
                )
            )
        return session_id

    def conversations(self) -> list[dict]:
        with Session(self.engine) as db:
            rows = db.scalars(select(Conversation).order_by(Conversation.created_at.desc())).all()
            return [{"id": row.id, "title": row.title} for row in rows]

    def history(self, session_id: str, turns: int | None = None) -> tuple[int, list[dict]]:
        with Session(self.engine) as db:
            conversation = db.get(Conversation, session_id)
            if conversation is None:
                raise ValueError("Диалог не найден")
            query = (
                select(Message)
                .where(Message.conversation_id == session_id)
                .order_by(Message.id.desc())
            )
            if turns is not None:
                query = query.limit(turns * 2)
            rows = list(reversed(db.scalars(query).all()))
            return conversation.revision, [
                {"role": row.role, "content": row.content, "details": row.details} for row in rows
            ]

    def save_turn(self, session_id: str, revision: int, question: str, answer: str, details: dict):
        """Сохраняем пару атомарно: ошибка LLM не оставляет одинокий user turn.

        Optimistic lock не позволяет двум вкладкам незаметно записать ответы,
        сгенерированные по одной устаревшей истории. Сеть вызывается вне SQL-
        транзакции; конфликтующий запрос предлагается повторить в интерфейсе.
        """
        with Session(self.engine) as db, db.begin():
            result = db.execute(
                update(Conversation)
                .where(Conversation.id == session_id, Conversation.revision == revision)
                .values(
                    revision=revision + 1, **({"title": question[:120]} if revision == 0 else {})
                )
            )
            if result.rowcount != 1:
                raise ConcurrentTurnError("Диалог изменён другой вкладкой. Повторите вопрос.")
            db.add_all(
                [
                    Message(conversation_id=session_id, role="user", content=question, details={}),
                    Message(
                        conversation_id=session_id,
                        role="assistant",
                        content=answer,
                        details=details,
                    ),
                ]
            )

    def close(self):
        self.engine.dispose()

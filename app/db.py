"""Task 5: database trail.

Every prediction is stored as one row that answers: what was decided, when, by which
model, and how confidently. DATABASE_URL picks the database:
    sqlite:///instance/visionguard.db                     (local default)
    postgresql+psycopg2://user:pass@db:5432/visionguard   (Docker Compose)
"""
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import DateTime, Float, Integer, String, create_engine, func, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
    # the input
    filename: Mapped[str] = mapped_column(String(255))
    file_sha256: Mapped[str] = mapped_column(String(64), index=True)
    file_size: Mapped[int] = mapped_column(Integer)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    # the decision
    label: Mapped[str] = mapped_column(String(20))
    prob_ai: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float)
    threshold: Mapped[float] = mapped_column(Float)
    # which model made it, and how fast
    model_name: Mapped[str] = mapped_column(String(100))
    model_version: Mapped[str] = mapped_column(String(50))
    latency_ms: Mapped[float] = mapped_column(Float)


class Database:
    def __init__(self, url: str):
        if url.startswith("sqlite:///"):
            Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
        self.url = url
        self.engine = create_engine(url, pool_pre_ping=True)
        self.Session = sessionmaker(self.engine, expire_on_commit=False)

    def create_tables(self) -> None:
        Base.metadata.create_all(self.engine)

    def add(self, **fields) -> PredictionRecord:
        with self.Session.begin() as s:
            rec = PredictionRecord(**fields)
            s.add(rec)
        return rec

    def get(self, record_id: int) -> PredictionRecord | None:
        with self.Session() as s:
            return s.get(PredictionRecord, record_id)

    def recent(self, limit: int = 100) -> list[PredictionRecord]:
        with self.Session() as s:
            q = select(PredictionRecord).order_by(PredictionRecord.id.desc()).limit(limit)
            return list(s.scalars(q))

    def count(self) -> int:
        with self.Session() as s:
            return s.scalar(select(func.count(PredictionRecord.id))) or 0

    def ping(self) -> bool:
        try:
            with self.engine.connect() as c:
                c.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import List, Optional

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
    text,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    Session,
    mapped_column,
    relationship,
    sessionmaker,
)


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


def _get_default_db_url() -> str:
    """Return the default SQLite URL for experiments.db at project root.

    デフォルトではリポジトリ直下に `experiments.sqlite` を作成する。
    必要なら環境変数 `EXPERIMENT_DB_URL` で上書き可能。
    """
    env_url = Path
    # 環境変数で上書きできるようにする
    import os

    url = os.getenv("EXPERIMENT_DB_URL")
    if url:
        return url

    project_root = Path(__file__).resolve().parent.parent
    db_path = project_root / "experiments.sqlite"
    return f"sqlite:///{db_path}"


engine = create_engine(_get_default_db_url(), future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)


class Experiment(Base):
    """1 回の `kernel_h.py` 実行を表す実験テーブル。"""

    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, nullable=False
    )

    dataset_name: Mapped[str] = mapped_column(String(255), nullable=False)
    constraint_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    kernel_method: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    tags: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    params_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    n_clusters_start: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    n_clusters_end: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    comet_experiment_key: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )

    results: Mapped[List["ClusterResult"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ClusterResult(Base):
    """各 n_clusters ごとのクラスタリング評価結果。"""

    __tablename__ = "cluster_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )

    n_clusters: Mapped[int] = mapped_column(Integer, nullable=False)

    silhouette_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    ch_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    db_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    ARI: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    NMI: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    FMI: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    purity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    entropy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    experiment: Mapped[Experiment] = relationship(back_populates="results")


class ClusterSummary(Base):
    """各実験ごとのクラスタ数推定結果の要約."""

    __tablename__ = "cluster_summaries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    experiment_id: Mapped[int] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )

    # 真のクラスタ数（ラベル数）
    n_clusters_true: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 外的評価（ARI）に基づく推定クラスタ数
    n_clusters_hat_ari: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    # 誤差 |K_hat - K_true|
    n_clusters_error_abs: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # K_hat_ari のときの外的評価値
    ari_at_hat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    nmi_at_hat: Mapped[Optional[float]] = mapped_column(Float, nullable=True)


def init_db() -> None:
    """テーブルが存在しなければ作成する。"""

    Base.metadata.create_all(bind=engine)

    # 既存DBに列追加（軽量マイグレーション）
    # SQLite は CREATE TABLE IF NOT EXISTS では列追加されないため、必要な列だけ ALTER する
    with engine.begin() as conn:
        cols = conn.execute(text("PRAGMA table_info('cluster_results')")).fetchall()
        existing = {row[1] for row in cols}  # row[1] = name

        if "purity" not in existing:
            conn.execute(text("ALTER TABLE cluster_results ADD COLUMN purity FLOAT"))
        if "entropy" not in existing:
            conn.execute(text("ALTER TABLE cluster_results ADD COLUMN entropy FLOAT"))


def get_session() -> Session:
    """新しい DB セッションを返す。

    呼び出し側で必ず close() すること。
    """

    # ensure tables exist
    init_db()
    return SessionLocal()


from datetime import datetime
from typing import Any, Optional, Dict
from sqlalchemy import String, Float, ForeignKey, Enum, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin, generate_uuid
from backend.models.base_model import JSONType, JobStatusType
from backend.schemas.enums import JobStatus


class JobModel(Base, TimestampMixin):
    """Research job records database model."""
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    dataset_id: Mapped[str] = mapped_column(String(36), ForeignKey("datasets.id"), nullable=False, index=True)
    status: Mapped[Any] = mapped_column(
        JobStatusType(),
        nullable=False,
        default=JobStatus.QUEUED,
        index=True
    )
    objective: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    mission_brief: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)
    progress_pct: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    # Relationships
    dataset = relationship("DatasetModel", back_populates="jobs")
    experiments = relationship("ExperimentModel", back_populates="job")
    knowledge_entries = relationship("KnowledgeEntryModel", back_populates="job")
    report = relationship("ReportModel", back_populates="job", uselist=False)

from typing import Any, Optional, Dict
from sqlalchemy import String, Integer, BigInteger, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.base import Base, TimestampMixin, generate_uuid
from backend.models.base_model import JSONType


class DatasetModel(Base, TimestampMixin):
    """Uploaded dataset metadata database model."""
    __tablename__ = "datasets"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    owner_id: Mapped[Optional[str]] = mapped_column(String(36), ForeignKey("users.id"), nullable=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(1024), nullable=False)
    file_size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    row_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    column_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    checksum: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    mission_brief: Mapped[Optional[str]] = mapped_column(String(2048), nullable=True)
    semantic_profile: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSONType, nullable=True)

    # Relationships
    owner = relationship("UserModel", back_populates="datasets")
    jobs = relationship("JobModel", back_populates="dataset")

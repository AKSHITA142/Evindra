from typing import Optional, List
from sqlalchemy.orm import Session
from backend.models.dataset import DatasetModel
from backend.repositories.base import BaseRepository


class DatasetRepository(BaseRepository[DatasetModel]):
    """Repository handling database operations for DatasetModel."""

    def __init__(self, session: Session):
        super().__init__(DatasetModel, session)

    def get_by_checksum(self, checksum: str) -> Optional[DatasetModel]:
        """Find dataset by SHA256 checksum to detect existing uploads."""
        return self.session.query(DatasetModel).filter(DatasetModel.checksum == checksum).first()

    def list(self, skip: int = 0, limit: int = 100) -> List[DatasetModel]:
        """List datasets ordered by most recently created first."""
        return (
            self.session.query(DatasetModel)
            .order_by(DatasetModel.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

    def list_by_owner(self, owner_id: str, skip: int = 0, limit: int = 100) -> List[DatasetModel]:
        """List datasets belonging to a specific owner."""
        return (
            self.session.query(DatasetModel)
            .filter(DatasetModel.owner_id == owner_id)
            .order_by(DatasetModel.created_at.desc())
            .offset(skip)
            .limit(limit)
            .all()
        )

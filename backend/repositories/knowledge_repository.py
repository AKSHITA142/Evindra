from typing import List, Optional
from sqlalchemy.orm import Session
from backend.models.knowledge import KnowledgeEntryModel
from backend.repositories.base import BaseRepository


class KnowledgeRepository(BaseRepository[KnowledgeEntryModel]):
    """Repository handling database operations for KnowledgeEntryModel."""

    def __init__(self, session: Session):
        super().__init__(KnowledgeEntryModel, session)

    def create(self, instance: Optional[KnowledgeEntryModel] = None, **kwargs) -> KnowledgeEntryModel:
        """Create and persist a new KnowledgeEntryModel, supporting both model instance and keyword args."""
        if instance is None:
            instance = KnowledgeEntryModel(**kwargs)
        elif kwargs:
            for k, v in kwargs.items():
                setattr(instance, k, v)
        return super().create(instance)

    def list_by_job(self, job_id: str) -> List[KnowledgeEntryModel]:
        """List knowledge findings accumulated for a research session/job."""
        return (
            self.session.query(KnowledgeEntryModel)
            .filter(KnowledgeEntryModel.job_id == job_id)
            .all()
        )

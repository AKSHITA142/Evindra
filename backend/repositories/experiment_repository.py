from typing import Optional, List
from sqlalchemy.orm import Session
from backend.models.experiment import ExperimentModel
from backend.repositories.base import BaseRepository


class ExperimentRepository(BaseRepository[ExperimentModel]):
    """Repository handling database operations for ExperimentModel."""

    def __init__(self, session: Session):
        super().__init__(ExperimentModel, session)

    def list_by_job(self, job_id: str, skip: int = 0, limit: int = 500) -> List[ExperimentModel]:
        """List experiments associated with a job (indexed lookup pattern)."""
        return (
            self.session.query(ExperimentModel)
            .filter(ExperimentModel.job_id == job_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def get_by_code(self, job_id: str, experiment_id_code: str) -> Optional[ExperimentModel]:
        """Get experiment by job ID and experiment code (e.g. 'EXP_001')."""
        return (
            self.session.query(ExperimentModel)
            .filter(ExperimentModel.job_id == job_id, ExperimentModel.experiment_id_code == experiment_id_code)
            .first()
        )

    def create_batch(self, experiments: List[ExperimentModel]) -> List[ExperimentModel]:
        """Bulk insert experiment models."""
        self.session.add_all(experiments)
        self.session.commit()
        for exp in experiments:
            self.session.refresh(exp)
        return experiments

from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from backend.models.job import JobModel
from backend.repositories.base import BaseRepository


class JobRepository(BaseRepository[JobModel]):
    """Repository handling database operations for JobModel."""

    def __init__(self, session: Session):
        super().__init__(JobModel, session)

    def list_by_dataset(self, dataset_id: str, skip: int = 0, limit: int = 100) -> List[JobModel]:
        """List jobs for a specific dataset."""
        return (
            self.session.query(JobModel)
            .filter(JobModel.dataset_id == dataset_id)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def list_by_status(self, status: str, skip: int = 0, limit: int = 100) -> List[JobModel]:
        """List jobs filtered by status (e.g. 'queued', 'running', 'completed')."""
        return (
            self.session.query(JobModel)
            .filter(JobModel.status == status)
            .offset(skip)
            .limit(limit)
            .all()
        )

    def update_status(
        self,
        job_id: str,
        status: Any,
        progress_pct: Optional[float] = None,
        error_message: Optional[str] = None
    ) -> Optional[JobModel]:
        """Update status, progress percentage, and optional error message of a job."""
        job = self.get_by_id(job_id)
        if not job:
            return None
        
        status_val = status.value if hasattr(status, "value") else str(status)
        job.status = status_val
        if job.started_at is None and status_val.lower() != "queued":
            from datetime import datetime, timezone
            job.started_at = datetime.now(timezone.utc)

        if progress_pct is not None:
            job.progress_pct = progress_pct
        if error_message is not None:
            job.error_message = error_message

        self.session.commit()
        self.session.refresh(job)
        return job

    def set_mission_brief(self, job_id: str, mission_brief: Dict[str, Any]) -> Optional[JobModel]:
        """Attach generated mission brief JSON to job."""
        job = self.get_by_id(job_id)
        if not job:
            return None
        
        job.mission_brief = mission_brief
        self.session.commit()
        self.session.refresh(job)
        return job

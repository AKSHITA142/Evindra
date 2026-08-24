from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func

from backend.database.connection import get_db
from backend.models.job import JobModel
from backend.models.experiment import ExperimentModel
from backend.schemas.response import SuccessResponse
from backend.schemas.enums import JobStatus

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])


@router.get("", response_model=SuccessResponse)
def get_dashboard(db: Session = Depends(get_db)):
    """
    Returns aggregated dashboard metrics:
    - total_jobs, completed_jobs, total_experiments
    - recent_jobs (latest 10 jobs)
    """
    total_jobs = db.query(func.count(JobModel.id)).scalar() or 0
    completed_jobs = (
        db.query(func.count(JobModel.id))
        .filter(JobModel.status == JobStatus.COMPLETED.value)
        .scalar()
        or 0
    )
    total_experiments = db.query(func.count(ExperimentModel.id)).scalar() or 0

    # Aggregate status counts across all jobs in the database
    status_counts_raw = (
        db.query(JobModel.status, func.count(JobModel.id))
        .group_by(JobModel.status)
        .all()
    )
    status_counts = {
        (s.value if hasattr(s, "value") else str(s)): count
        for s, count in status_counts_raw
    }

    # Fetch the latest 15 research jobs (ordered by creation time descending)
    recent_job_rows = (
        db.query(JobModel)
        .order_by(JobModel.created_at.desc())
        .limit(15)
        .all()
    )

    # Map granular backend statuses to pipeline stages
    _STATUS_TO_STAGE = {
        "queued": None, "profiling": "profiling", "understanding": "understanding",
        "planning": "planning", "executing": "executing", "evaluating": "evaluating",
        "directing": "decision", "reporting": "reporting",
        "completed": None, "failed": None, "cancelled": None,
    }

    recent_jobs = [
        {
            "job_id": job.id,
            "dataset_id": job.dataset_id,
            "mission": job.objective or "Automated ML Research",
            "status": job.status.value if hasattr(job.status, "value") else str(job.status),
            "current_stage": _STATUS_TO_STAGE.get(job.status.value if hasattr(job.status, "value") else str(job.status)),
            "progress_percent": job.progress_pct,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "error_message": job.error_message,
        }
        for job in recent_job_rows
    ]

    return SuccessResponse(
        data={
            "total_jobs": total_jobs,
            "completed_jobs": completed_jobs,
            "status_counts": status_counts,
            "recent_jobs": recent_jobs,
            "total_experiments": total_experiments,
        },
        message="Dashboard metrics retrieved successfully.",
    )

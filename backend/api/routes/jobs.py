from typing import Optional
from fastapi import APIRouter, Depends, BackgroundTasks, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.services.job_service import JobService
from backend.schemas.response import SuccessResponse

router = APIRouter(prefix="/jobs", tags=["Research Jobs"])


# ── Mapping helpers ────────────────────────────────────────────────────

# Map granular backend statuses to the broader PipelineStage the frontend expects.
_STATUS_TO_STAGE = {
    "queued": None,
    "profiling": "profiling",
    "understanding": "understanding",
    "planning": "planning",
    "executing": "executing",
    "evaluating": "evaluating",
    "directing": "decision",
    "reporting": "reporting",
    "completed": None,
    "failed": None,
    "cancelled": None,
}


def _job_to_frontend(job_record) -> dict:
    """Converts a JobModel row into the dictionary shape the frontend `Job` type expects."""
    return {
        "job_id": job_record.id,
        "dataset_id": job_record.dataset_id,
        "status": job_record.status,
        "mission": job_record.objective or "Automated ML Research",
        "current_stage": _STATUS_TO_STAGE.get(job_record.status),
        "progress_percent": job_record.progress_pct,
        "created_at": job_record.created_at.isoformat() if job_record.created_at else None,
        "updated_at": job_record.updated_at.isoformat() if job_record.updated_at else None,
        "error_message": job_record.error_message,
        # Keep original field names for backwards-compatibility
        "progress_pct": job_record.progress_pct,
        "objective": job_record.objective,
    }


# ── Request models ─────────────────────────────────────────────────────

class StartJobRequest(BaseModel):
    dataset_id: str
    user_goal: Optional[str] = None
    # Accept `mission` as an alias so the frontend can send either field name
    mission: Optional[str] = None
    task_type: Optional[str] = "general"


# ── Route handlers ─────────────────────────────────────────────────────

@router.post("/start", response_model=SuccessResponse, status_code=status.HTTP_202_ACCEPTED)
def start_research_job(
    payload: StartJobRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Triggers an automated research job asynchronously for a dataset.
    Returns immediately with job_id and queued status.
    """
    # Accept either `user_goal` or `mission` from the frontend
    effective_goal = payload.user_goal or payload.mission

    service = JobService(db)
    job_record = service.start_job(
        dataset_id=payload.dataset_id,
        user_goal=effective_goal,
        task_type=payload.task_type or "general",
        background_tasks=background_tasks,
    )

    return SuccessResponse(
        data=_job_to_frontend(job_record),
        message="Research job queued and started in background worker.",
    )


@router.get("/{job_id}", response_model=SuccessResponse)
def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """
    Queries real-time status and progress percentage for a research job.
    """
    service = JobService(db)
    job_record = service.get_job(job_id)

    return SuccessResponse(
        data=_job_to_frontend(job_record),
        message="Job status retrieved successfully.",
    )


@router.get("/{job_id}/logs", response_model=SuccessResponse)
def get_job_logs(job_id: str, db: Session = Depends(get_db)):
    """
    Retrieves full chronological execution logs and audit trail for a job.
    """
    service = JobService(db)
    logs = service.get_job_logs(job_id)

    return SuccessResponse(
        data=logs,
        message="Job execution logs retrieved successfully.",
    )


@router.post("/{job_id}/cancel", response_model=SuccessResponse)
def cancel_job(job_id: str, db: Session = Depends(get_db)):
    """
    Requests cancellation of a running or queued research job.
    """
    service = JobService(db)
    job_record = service.cancel_job(job_id)

    return SuccessResponse(
        data={
            "job_id": job_record.id,
            "status": job_record.status,
        },
        message="Job cancellation requested successfully.",
    )

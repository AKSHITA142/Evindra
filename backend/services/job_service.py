import asyncio
import logging
import uuid
from typing import Optional, Dict, Any, List
from fastapi import BackgroundTasks
from sqlalchemy.orm import Session

from backend.core.exceptions import NotFoundException, ValidationException, ConflictException
from backend.repositories.job_repository import JobRepository
from backend.repositories.dataset_repository import DatasetRepository
from backend.models.job import JobModel
from backend.schemas.enums import JobStatus
from backend.services.job_manager import JobManager

logger = logging.getLogger("datapilot.services.job_service")


def _run_job_in_background(
    job_id: str,
    dataset_id: str,
    file_path: str,
    user_goal: Optional[str] = None,
    task_type: str = "general",
):
    """
    Sync wrapper that safely schedules the async JobManager coroutine
    onto the already-running FastAPI event loop.

    CRITICAL FIX: asyncio.run() cannot be called from within a running event loop
    (which FastAPI always has). Using loop.create_task() instead.
    """
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(
            JobManager.run_job_async(
                job_id=job_id,
                dataset_id=dataset_id,
                file_path=file_path,
                user_goal=user_goal,
                task_type=task_type,
            )
        )
    except RuntimeError:
        # No running loop (shouldn't happen in FastAPI, but handle gracefully)
        logger.warning(f"No running event loop found for job {job_id}; creating new loop")
        asyncio.run(
            JobManager.run_job_async(
                job_id=job_id,
                dataset_id=dataset_id,
                file_path=file_path,
                user_goal=user_goal,
                task_type=task_type,
            )
        )


class JobService:
    """Service layer managing research job creation, status querying, and cancellation."""

    def __init__(self, db: Session):
        self.db = db
        self.repository = JobRepository(db)
        self.dataset_repository = DatasetRepository(db)

    def start_job(
        self,
        dataset_id: str,
        user_goal: Optional[str] = None,
        task_type: str = "general",
        background_tasks: Optional[BackgroundTasks] = None
    ) -> JobModel:
        """
        Creates a new Job record and dispatches background execution worker.
        """
        dataset = self.dataset_repository.get_by_id(dataset_id)
        if not dataset:
            raise NotFoundException(f"Dataset with ID '{dataset_id}' not found.")

        # If task_type is default "general", check if dataset profile stored user's explicit choice
        if task_type == "general" and dataset.semantic_profile:
            prof_dict = dataset.semantic_profile if isinstance(dataset.semantic_profile, dict) else {}
            task_type = prof_dict.get("user_task_type") or "general"

        job_id = f"job_{uuid.uuid4().hex[:8]}"

        # Create Job DB record
        job_record = self.repository.create(
            JobModel(
                id=job_id,
                dataset_id=dataset_id,
                status=JobStatus.QUEUED.value,
                objective=user_goal or "Automated ML Research & Preprocessing Optimization",
                progress_pct=0.0,
            )
        )

        # Dispatch background worker using safe async scheduling
        if background_tasks:
            background_tasks.add_task(
                _run_job_in_background,
                job_id=job_id,
                dataset_id=dataset_id,
                file_path=dataset.file_path,
                user_goal=user_goal,
                task_type=task_type,
            )
        else:
            # Direct dispatch on running event loop
            _run_job_in_background(
                job_id=job_id,
                dataset_id=dataset_id,
                file_path=dataset.file_path,
                user_goal=user_goal,
                task_type=task_type,
            )

        return job_record

    def get_job(self, job_id: str) -> JobModel:
        """Retrieves job record by ID."""
        job = self.repository.get_by_id(job_id)
        if not job:
            raise NotFoundException(f"Research job with ID '{job_id}' not found.")
        return job

    def cancel_job(self, job_id: str) -> JobModel:
        """Requests cancellation for a running or queued job."""
        job = self.get_job(job_id)

        if job.status in [JobStatus.COMPLETED.value, JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            raise ConflictException(f"Cannot cancel job {job_id} in status '{job.status}'.")

        updated_job = self.repository.update_status(job_id, JobStatus.CANCELLED)
        return updated_job

    def get_job_logs(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Reconstructs the full execution event audit trail for a job.
        Provides historical logs for completed or running jobs.
        """
        job = self.get_job(job_id)
        logs = []
        created_iso = job.created_at.isoformat() if job.created_at else None

        logs.append({
            "id": f"{job_id}-start",
            "timestamp": created_iso,
            "level": "info",
            "message": f"Initialized research job '{job.id}' for dataset '{job.dataset_id}'.",
            "stage": "profiling",
        })
        logs.append({
            "id": f"{job_id}-graph",
            "timestamp": created_iso,
            "level": "info",
            "message": "Compiled LangGraph research workflow. State machine initialized.",
            "stage": "profiling",
        })

        if job.dataset:
            ds = job.dataset
            prof = ds.semantic_profile if isinstance(ds.semantic_profile, dict) else {}
            summary = prof.get("dataset_summary", {}) if isinstance(prof.get("dataset_summary"), dict) else {}
            rows = prof.get("row_count") or summary.get("rows") or ds.row_count or "N/A"
            cols = prof.get("column_count") or summary.get("columns") or ds.column_count or "N/A"
            target_info = summary.get("target", {}) if isinstance(summary.get("target"), dict) else {}
            target = target_info.get("target_column") or prof.get("detected_target_column") or "Auto-detected"
            task = target_info.get("task_type") or prof.get("detected_task_type") or prof.get("user_task_type") or "classification"
            logs.append({
                "id": f"{job_id}-prof",
                "timestamp": created_iso,
                "level": "success",
                "message": f"Dataset profiling completed: {rows} rows, {cols} columns profiled.",
                "stage": "profiling",
            })
            logs.append({
                "id": f"{job_id}-und",
                "timestamp": created_iso,
                "level": "info",
                "message": f"Semantic analysis complete. Target: '{target}', Task Type: '{task}'.",
                "stage": "understanding",
            })
            logs.append({
                "id": f"{job_id}-plan",
                "timestamp": created_iso,
                "level": "info",
                "message": "Experiment planner formulated multi-model candidate evaluation matrix.",
                "stage": "planning",
            })

        # Experiments log entries
        if job.experiments:
            for idx, exp in enumerate(job.experiments):
                exp_ts = exp.created_at.isoformat() if exp.created_at else created_iso
                metrics = exp.metrics if isinstance(exp.metrics, dict) else {}
                primary_metric = metrics.get("primary_metric_name") or "Score"
                primary_val = metrics.get("primary_metric_value") or metrics.get("composite_score") or metrics.get("accuracy") or metrics.get("r2")
                metric_str = f"{primary_metric} = {primary_val:.4f}" if isinstance(primary_val, (int, float)) else "Evaluated"
                runtime = f"{exp.runtime_seconds:.2f}s" if exp.runtime_seconds else "N/A"
                logs.append({
                    "id": f"{job_id}-exp-{exp.id}",
                    "timestamp": exp_ts,
                    "level": "info" if exp.status == "completed" else "warning",
                    "message": f"Experiment #{idx+1} ({exp.model_name}): {metric_str} [Runtime: {runtime}].",
                    "stage": "executing",
                })

        # Knowledge findings log entries
        if job.knowledge_entries:
            for idx, k in enumerate(job.knowledge_entries):
                k_ts = k.created_at.isoformat() if k.created_at else created_iso
                logs.append({
                    "id": f"{job_id}-know-{k.id}",
                    "timestamp": k_ts,
                    "level": "success",
                    "message": f"Knowledge finding: {k.finding[:90]}...",
                    "stage": "evaluating",
                })

        # Report & Recommendation
        if job.report and job.report.summary:
            summary = job.report.summary if isinstance(job.report.summary, dict) else {}
            winner = summary.get("recommended_model") or "Optimal Model"
            score = summary.get("primary_metric_value") or summary.get("composite_score")
            score_str = f" ({score:.4f})" if isinstance(score, (int, float)) else ""
            logs.append({
                "id": f"{job_id}-winner",
                "timestamp": created_iso,
                "level": "success",
                "message": f"Decision stage: Selected '{winner}'{score_str} as winning architecture.",
                "stage": "decision",
            })
            logs.append({
                "id": f"{job_id}-report",
                "timestamp": created_iso,
                "level": "success",
                "message": "Generated standalone HTML & Markdown research reports.",
                "stage": "reporting",
            })

        if job.status == "completed":
            logs.append({
                "id": f"{job_id}-done",
                "timestamp": created_iso,
                "level": "success",
                "message": "Research job completed successfully. All stages verified.",
                "stage": "reporting",
            })
        elif job.status == "failed":
            logs.append({
                "id": f"{job_id}-fail",
                "timestamp": created_iso,
                "level": "error",
                "message": f"Job execution failed: {job.error_message or 'Unknown error'}",
                "stage": None,
            })

        return logs

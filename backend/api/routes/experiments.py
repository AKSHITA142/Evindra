import math
from typing import List
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.database.connection import get_db
from backend.services.experiment_service import ExperimentService
from backend.schemas.response import SuccessResponse

router = APIRouter(prefix="/experiments", tags=["Experiments Explorer"])


def _sanitize_json(obj):
    """Recursively replaces NaN and Inf float values with None for valid JSON serialization."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_json(item) for item in obj]
    return obj


def _experiment_to_frontend(exp, job_id: str) -> dict:
    """
    Converts an ExperimentModel row into the flattened dictionary shape
    the frontend `ExperimentResult` type expects.

    Frontend needs top-level metric fields (accuracy, f1_score, composite_score, etc.)
    instead of a nested `metrics: {}` object.
    """
    metrics = exp.metrics or {}
    inner_metrics = metrics.get("metrics", metrics) if isinstance(metrics, dict) else {}

    from backend.api.routes.reports import compute_composite_and_confidence
    composite_score, confidence_score, primary_metric_name = compute_composite_and_confidence(inner_metrics)
    p_name = inner_metrics.get("primary_metric_name") or (metrics.get("primary_metric_name") if isinstance(metrics, dict) else None)
    if p_name:
        primary_metric_name = p_name

    raw_primary = (metrics.get("primary_metric") if isinstance(metrics, dict) else None)
    if raw_primary is None and isinstance(inner_metrics, dict):
        raw_primary = inner_metrics.get("primary_metric")
    if raw_primary is None and isinstance(inner_metrics, dict):
        raw_primary = inner_metrics.get("rmse") if "rmse" in inner_metrics else (inner_metrics.get("f1_score") if "f1_score" in inner_metrics else inner_metrics.get("f1"))

    if raw_primary is not None and isinstance(raw_primary, (int, float)) and not math.isnan(raw_primary) and not math.isinf(raw_primary):
        primary_metric_value = float(raw_primary)
    else:
        primary_metric_value = 0.0

    # Build pipeline name from operations or model name
    pipeline_ops = exp.pipeline or {}
    operations = pipeline_ops.get("operations", []) if isinstance(pipeline_ops, dict) else []
    if operations:
        step_names = [op.get("method", op.get("type", "step")) for op in operations if isinstance(op, dict)]
        pipeline_name = " → ".join(step_names) + f" → {exp.model_name}"
    else:
        pipeline_name = exp.model_name or "Unknown Pipeline"

    f1_val = inner_metrics.get("f1_score") if isinstance(inner_metrics, dict) and inner_metrics.get("f1_score") is not None else (inner_metrics.get("f1") if isinstance(inner_metrics, dict) else None)

    raw_dict = {
        "experiment_id": exp.experiment_id_code,
        "job_id": job_id,
        "pipeline_name": pipeline_name,
        "model_name": exp.model_name,
        "model_type": exp.model_name,
        "status": exp.status or "completed",
        "primary_metric_name": primary_metric_name,
        "primary_metric_value": primary_metric_value,
        "primary_metric_rationale": inner_metrics.get("primary_metric_rationale") if isinstance(inner_metrics, dict) else None,
        "composite_score": composite_score,
        # Flatten individual metrics to top-level
        "accuracy": inner_metrics.get("accuracy") if isinstance(inner_metrics, dict) else None,
        "precision": inner_metrics.get("precision") if isinstance(inner_metrics, dict) else None,
        "recall": inner_metrics.get("recall") if isinstance(inner_metrics, dict) else None,
        "f1_score": f1_val,
        "roc_auc": inner_metrics.get("roc_auc") if isinstance(inner_metrics, dict) else None,
        "rmse": inner_metrics.get("rmse") if isinstance(inner_metrics, dict) else None,
        "mae": inner_metrics.get("mae") if isinstance(inner_metrics, dict) else None,
        "r2": inner_metrics.get("r2") if isinstance(inner_metrics, dict) else None,
        "runtime_seconds": exp.runtime_seconds,
        "feature_importance": (exp.artifact_paths or {}).get("feature_importance") if isinstance(exp.artifact_paths, dict) else None,
        "pipeline_steps": [op.get("method", "") for op in operations if isinstance(op, dict)] if operations else None,
        "error_message": (metrics.get("error") if isinstance(metrics, dict) else None),
        "created_at": exp.created_at.isoformat() if exp.created_at else None,
        "completed_at": None,
        # Keep original nested format for backwards-compatibility
        "pipeline": exp.pipeline,
        "metrics": exp.metrics,
        "artifact_paths": exp.artifact_paths,
    }
    return _sanitize_json(raw_dict)


@router.get("/{job_id}", response_model=SuccessResponse)
def list_job_experiments(job_id: str, db: Session = Depends(get_db)):
    """
    Lists all executed experiment results and metrics for a research job.
    """
    service = ExperimentService(db)
    experiments = service.list_experiments(job_id)

    exp_data = [_experiment_to_frontend(exp, job_id) for exp in experiments]

    return SuccessResponse(
        data=exp_data,
        message=f"Retrieved {len(exp_data)} experiments for job '{job_id}'.",
    )


@router.get("/detail/{experiment_id}", response_model=SuccessResponse)
def get_experiment_detail(experiment_id: str, db: Session = Depends(get_db)):
    """
    Retrieves a single experiment by its database ID.
    """
    service = ExperimentService(db)
    # Search across all jobs for this experiment DB id
    from backend.repositories.experiment_repository import ExperimentRepository
    repo = ExperimentRepository(db)
    exp = repo.get_by_id(experiment_id)
    if not exp:
        from backend.core.exceptions import NotFoundException
        raise NotFoundException(f"Experiment '{experiment_id}' not found.")

    return SuccessResponse(
        data=_experiment_to_frontend(exp, exp.job_id),
        message="Experiment detail retrieved successfully.",
    )

import uuid
from typing import Any, Dict, List, Optional
from pydantic import Field

from backend.schemas.base import BaseSchema


class FinalValidationReport(BaseSchema):
    """
    Final Holdout Validation Report for winning pipeline (Phase 16).
    """
    report_id: str = Field(default_factory=lambda: f"hval_{uuid.uuid4().hex[:10]}")
    pipeline_id: str
    pipeline_name: str = "Pipeline"
    model_family: str = "UNKNOWN"
    primary_metric: str = "accuracy"
    cv_score: float = 0.0
    holdout_score: float = 0.0
    difference: float = 0.0
    generalization_assessment: str = "GOOD"  # GOOD, MILD_OVERFITTING, SEVERE_OVERFITTING, SUSPICIOUS
    holdout_metrics: Dict[str, float] = Field(default_factory=dict)
    confusion_matrix: List[List[int]] = Field(default_factory=list)
    residual_analysis: Dict[str, float] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    leakage_checks: Dict[str, str] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

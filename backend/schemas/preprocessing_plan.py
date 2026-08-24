import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import Field

from backend.schemas.base import BaseSchema
from backend.schemas.decision import DecisionDomain, DecisionSource, DecisionResult


class PreprocessingStep(BaseSchema):
    """
    A single executable preprocessing step in a PreprocessingPlan, linked back to its decision source.
    """
    step_id: str = Field(default_factory=lambda: f"step_{uuid.uuid4().hex[:10]}")
    step_number: int
    domain: DecisionDomain
    action: str
    columns: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    decision_id: str
    decision_source: DecisionSource
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str = ""
    requires_validation: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PreprocessingPlan(BaseSchema):
    """
    Unified Preprocessing Plan aggregating column-level and dataset-level decisions into an ordered DAG.
    """
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:12]}")
    dataset_name: str = "dataset"
    target_column: Optional[str] = None
    problem_type: str = "general_tabular"
    model_family: str = "general"
    steps: List[PreprocessingStep] = Field(default_factory=list)
    decisions_summary: Dict[str, Any] = Field(default_factory=dict)
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic dictionary representation."""
        return self.model_dump(mode="json")

    def to_experiment_operations(self) -> List[Dict[str, Any]]:
        """Converts steps into legacy ExperimentOperation dictionary specs for execution engine compatibility."""
        ops = []
        for s in self.steps:
            ops.append({
                "type": s.domain.value if hasattr(s.domain, "value") else str(s.domain),
                "method": s.action,
                "params": {"columns": s.columns, **s.params},
            })
        return ops


class PlanValidationResult(BaseSchema):
    """
    Deterministic validation result produced by PlanValidator for a PreprocessingPlan.
    """
    is_valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    validated_step_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanExecutionResult(BaseSchema):
    """
    Result of safely executing a PreprocessingPlan on a DataFrame.
    """
    plan_id: str
    status: str = "SUCCESS"  # SUCCESS, FAILED
    initial_shape: List[int] = Field(default_factory=list)
    final_shape: List[int] = Field(default_factory=list)
    executed_steps_count: int = 0
    step_logs: List[Dict[str, Any]] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)



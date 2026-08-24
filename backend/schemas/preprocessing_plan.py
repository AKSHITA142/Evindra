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
    stage: str = "PREPROCESSING"  # DATA_INGESTION, TARGET_SEPARATION, LEAKAGE_REMOVAL, etc.
    domain: DecisionDomain
    action: str
    columns: List[str] = Field(default_factory=list)
    params: Dict[str, Any] = Field(default_factory=dict)
    dependencies: List[str] = Field(default_factory=list)
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
    dataset_id: str = "dataset_001"
    dataset_name: str = "dataset"
    target: Optional[str] = None
    target_column: Optional[str] = None
    problem_type: str = "general_tabular"
    model_family: str = "general"
    steps: List[PreprocessingStep] = Field(default_factory=list)
    operations: List[Dict[str, Any]] = Field(default_factory=list)
    dependencies: Dict[str, List[str]] = Field(default_factory=dict)
    expected_input_schema: Dict[str, str] = Field(default_factory=dict)
    expected_output_schema: Dict[str, str] = Field(default_factory=dict)
    decisions_summary: Dict[str, Any] = Field(default_factory=dict)
    overall_confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = ""
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def model_post_init(self, __context: Any) -> None:
        """Ensure alias field compatibility between target and target_column, dataset_id and dataset_name."""
        if not self.target and self.target_column:
            self.target = self.target_column
        elif not self.target_column and self.target:
            self.target_column = self.target
        if not self.dataset_id and self.dataset_name:
            self.dataset_id = self.dataset_name
        elif not self.dataset_name and self.dataset_id:
            self.dataset_name = self.dataset_id

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
    valid: bool = True
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    checks: Dict[str, str] = Field(default_factory=dict)
    severity: str = "CLEAN"  # CLEAN, WARNING, CRITICAL
    recommended_action: str = "PROCEED"  # PROCEED, REVISE, REJECT
    validated_step_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        """Sync valid and is_valid fields."""
        if not self.is_valid or not self.valid:
            self.is_valid = False
            self.valid = False


class PlanExecutionResult(BaseSchema):
    """
    Result of safely executing a PreprocessingPlan on a DataFrame.
    """
    plan_id: str
    status: str = "SUCCESS"  # SUCCESS, FAILED
    initial_shape: List[int] = Field(default_factory=list)
    final_shape: List[int] = Field(default_factory=list)
    train_shape: List[int] = Field(default_factory=list)
    test_shape: List[int] = Field(default_factory=list)
    executed_steps_count: int = 0
    step_logs: List[Dict[str, Any]] = Field(default_factory=list)
    execution_trace: List[Dict[str, Any]] = Field(default_factory=list)
    transformation_mapping: Dict[str, List[str]] = Field(default_factory=dict)
    fitted_pipeline_info: Dict[str, Any] = Field(default_factory=dict)
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)



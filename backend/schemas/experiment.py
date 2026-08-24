import uuid
from typing import Any, Dict, List, Optional
from pydantic import Field, BaseModel

from backend.schemas.base import BaseSchema


class ExperimentOperation(BaseSchema):
    type: str
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)


class ExperimentSpec(BaseSchema):
    experiment_id: str
    dataset_name: str
    operations: List[ExperimentOperation] = Field(default_factory=list)


class ExperimentPlan(BaseSchema):
    plan_id: str
    experiments: List[ExperimentSpec] = Field(default_factory=list)


class PipelineDefinition(BaseSchema):
    pipeline_id: str
    steps: List[Dict[str, Any]] = Field(default_factory=list)


class MetricsResult(BaseSchema):
    metrics: Dict[str, float] = Field(default_factory=dict)


class Artifacts(BaseSchema):
    models: List[str] = Field(default_factory=list)


class ExperimentResult(BaseSchema):
    experiment_id: str
    status: str = "SUCCESS"
    metrics: Dict[str, float] = Field(default_factory=dict)


class PipelineEvaluationResult(BaseSchema):
    """
    Evaluation results for a single candidate pipeline across CV folds (Phase 14).
    """
    evaluation_id: str = Field(default_factory=lambda: f"eval_{uuid.uuid4().hex[:10]}")
    pipeline_id: str
    pipeline_name: str = "Pipeline"
    model_family: str = "UNKNOWN"
    status: str = "SUCCESS"  # SUCCESS, FAILED
    primary_metric: str = "accuracy"
    primary_score: float = 0.0
    mean_metrics: Dict[str, float] = Field(default_factory=dict)
    std_metrics: Dict[str, float] = Field(default_factory=dict)
    fold_scores: List[Dict[str, float]] = Field(default_factory=list)
    training_time_seconds: float = 0.0
    prediction_time_seconds: float = 0.0
    feature_count: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")


class ExperimentRunReport(BaseSchema):
    """
    Comprehensive report summarizing an experiment run across multiple candidate pipelines.
    """
    run_id: str = Field(default_factory=lambda: f"exprun_{uuid.uuid4().hex[:10]}")
    dataset_name: str = "dataset"
    problem_type: str = "classification"
    primary_metric: str = "roc_auc"
    best_pipeline_id: Optional[str] = None
    best_primary_score: float = 0.0
    total_pipelines_evaluated: int = 0
    successful_evaluations: int = 0
    failed_evaluations: int = 0
    evaluation_results: List[PipelineEvaluationResult] = Field(default_factory=list)
    execution_time_seconds: float = 0.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

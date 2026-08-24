import uuid
from typing import Any, Dict, List, Optional
from pydantic import Field, BaseModel

from backend.schemas.base import BaseSchema


class ExperimentOperation(BaseSchema):
    type: str
    method: str
    params: Dict[str, Any] = Field(default_factory=dict)


class PipelineDefinition(BaseSchema):
    """Pipeline definition used by both the LangGraph evaluation layer and the new engine."""
    pipeline_id: str = Field(default_factory=lambda: f"pipe_{uuid.uuid4().hex[:8]}")
    operations: List[ExperimentOperation] = Field(default_factory=list)
    model_name: str = ""
    steps: List[Dict[str, Any]] = Field(default_factory=list)


class MetricsResult(BaseSchema):
    """Experiment metrics result — supports both structured and flat-dict usage."""
    primary_metric: float = 0.0
    primary_metric_name: str = ""
    primary_metric_rationale: str = ""
    metrics: Dict[str, float] = Field(default_factory=dict)
    cv_scores: List[float] = Field(default_factory=list)


class ExperimentResult(BaseSchema):
    """Single experiment result record used by the evaluation engine."""
    experiment_id: str
    pipeline: Optional[PipelineDefinition] = None
    model: str = ""
    metrics: Optional[MetricsResult] = None
    runtime: float = 0.0
    status: str = "completed"


class ExperimentSpec(BaseSchema):
    """Experiment specification used by the StrategyPlannerAgent."""
    experiment_id: str
    dataset_name: str = "dataset"
    operations: List[ExperimentOperation] = Field(default_factory=list)
    model_name: str = ""
    params: Dict[str, Any] = Field(default_factory=dict)


class ExperimentPlan(BaseSchema):
    """Experiment plan produced by the StrategyPlannerAgent."""
    plan_id: str = Field(default_factory=lambda: f"plan_{uuid.uuid4().hex[:8]}")
    experiments: List[ExperimentSpec] = Field(default_factory=list)


class Artifacts(BaseSchema):
    models: List[str] = Field(default_factory=list)


class PipelineEvaluationResult(BaseSchema):
    """
    Evaluation results for a single candidate pipeline across CV folds (Phase 14).
    """
    evaluation_id: str = Field(default_factory=lambda: f"eval_{uuid.uuid4().hex[:10]}")
    pipeline_id: str
    pipeline_name: str = "Pipeline"
    model_family: str = "UNKNOWN"
    status: str = "SUCCESS"
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
        return self.model_dump(mode="json")

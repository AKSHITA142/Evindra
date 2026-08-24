import uuid
from typing import Any, Dict, List, Optional
from pydantic import Field

from backend.schemas.base import BaseSchema
from backend.schemas.preprocessing_plan import PreprocessingPlan


class PipelineCandidate(BaseSchema):
    """
    A single valid preprocessing + feature + model candidate pipeline (Phase 13).
    """
    pipeline_id: str = Field(default_factory=lambda: f"pipe_{uuid.uuid4().hex[:10]}")
    name: str
    description: str
    preprocessing_plan: PreprocessingPlan
    feature_engineering_plan: Dict[str, Any] = Field(default_factory=dict)
    feature_selection_plan: Dict[str, Any] = Field(default_factory=dict)
    model_spec: Dict[str, Any] = Field(default_factory=dict, alias="model_config")
    estimated_cost: str = "LOW"  # LOW, MEDIUM, HIGH
    rank_score: float = 1.0
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def model_config_dict(self) -> Dict[str, Any]:
        """Accessor for model configuration dictionary."""
        return self.model_spec

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")


class PipelineCandidateSet(BaseSchema):
    """
    Collection of generated pipeline candidates for a dataset.
    """
    set_id: str = Field(default_factory=lambda: f"pset_{uuid.uuid4().hex[:10]}")
    dataset_name: str = "dataset"
    problem_type: str = "classification"
    target_column: Optional[str] = None
    total_candidates: int = 0
    pipelines: List[PipelineCandidate] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

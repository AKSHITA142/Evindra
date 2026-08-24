import uuid
from typing import Any, Dict, List, Optional
from pydantic import Field

from backend.schemas.base import BaseSchema
from backend.schemas.experiment import PipelineEvaluationResult


class BestPipelineResult(BaseSchema):
    """
    Deterministic final pipeline selection result (Phase 15).
    """
    selection_id: str = Field(default_factory=lambda: f"sel_{uuid.uuid4().hex[:10]}")
    winner_pipeline_id: str
    winner_pipeline_name: str = "Winner Pipeline"
    winner_model_family: str = "UNKNOWN"
    metric: str = "accuracy"
    score: float = 0.0
    confidence: float = 0.90
    selection_reason: str = "Selected based on optimal performance-complexity tradeoff."
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    tradeoffs: Dict[str, Any] = Field(default_factory=dict)
    winner_evaluation: Optional[PipelineEvaluationResult] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

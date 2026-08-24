from typing import Dict, List, Any
from pydantic import Field
from backend.schemas.base import BaseSchema
from backend.schemas.experiment import PipelineDefinition


class FinalRecommendation(BaseSchema):
    """Final, human-readable recommendation produced by the Report Generator."""
    winning_experiment_id: str
    pipeline: PipelineDefinition
    model: str
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    final_metrics: Dict[str, float] = Field(default_factory=dict)
    summary: str
    key_findings: List[str] = Field(default_factory=list)
    exported_artifacts: Dict[str, str] = Field(default_factory=dict)

import uuid
from typing import Any, Dict, List, Optional
from pydantic import Field

from backend.schemas.base import BaseSchema


class CandidateFeature(BaseSchema):
    """
    Metadata and provenance for a single generated candidate feature.
    """
    feature_id: str = Field(default_factory=lambda: f"feat_{uuid.uuid4().hex[:10]}")
    feature_name: str
    source_columns: List[str] = Field(default_factory=list)
    operation: str
    reason: str
    domain: str = "feature_engineering"
    leakage_status: str = "LEAKAGE_FREE"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CandidateFeatureSet(BaseSchema):
    """
    Collection of generated candidate features with complete provenance metadata.
    """
    set_id: str = Field(default_factory=lambda: f"fset_{uuid.uuid4().hex[:10]}")
    dataset_name: str = "dataset"
    target_column: Optional[str] = None
    total_candidates_generated: int = 0
    candidates: List[CandidateFeature] = Field(default_factory=list)
    generated_feature_names: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

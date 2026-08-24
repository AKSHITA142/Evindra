import uuid
from typing import Any, Dict, List, Optional
from pydantic import Field

from backend.schemas.base import BaseSchema


class FeatureRemovalDetail(BaseSchema):
    """
    Detail for a single feature removed during selection.
    """
    feature_name: str
    reason_removed: str
    method: str
    score: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeatureSelectionReport(BaseSchema):
    """
    Comprehensive report produced by FeatureSelector (Phase 12).
    """
    report_id: str = Field(default_factory=lambda: f"fsrep_{uuid.uuid4().hex[:10]}")
    dataset_name: str = "dataset"
    target_column: Optional[str] = None
    initial_feature_count: int = 0
    selected_feature_count: int = 0
    removed_feature_count: int = 0
    selected_features: List[str] = Field(default_factory=list)
    removed_features: List[FeatureRemovalDetail] = Field(default_factory=list)
    feature_scores: Dict[str, float] = Field(default_factory=dict)
    fold_stability: Dict[str, float] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")

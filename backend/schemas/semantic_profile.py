from typing import Any, Dict, List, Optional
from pydantic import Field
from backend.schemas.base import BaseSchema, ConfidenceScoredModel
from backend.schemas.enums import ColumnType, SeverityLevel


class ColumnProfile(BaseSchema):
    """Profile statistics for a single column."""
    name: str
    type: ColumnType = ColumnType.UNKNOWN
    missing_count: int = 0
    missing_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    distinct_count: int = 0
    skewness: Optional[float] = None
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    sample_values: List[Any] = Field(default_factory=list)
    encoding_recommendation: Optional[str] = None
    scaling_recommendation: Optional[str] = None


class QualityIssue(ConfidenceScoredModel):
    """Quality issue identified in the dataset by the Profiling Engine."""
    problem: str
    severity: SeverityLevel = SeverityLevel.MEDIUM
    description: Optional[str] = None
    affected_columns: List[str] = Field(default_factory=list)


class ResourceProfile(BaseSchema):
    """Hardware and execution hints computed for dataset size."""
    execution_mode: str = "standard"
    use_lazy_loading: bool = False
    recommended_workers: int = 1
    memory_mb: float = 0.0


class SemanticProfile(BaseSchema):
    """Structured, LLM-consumable output of the Profiling Engine."""
    dataset_summary: Dict[str, Any] = Field(
        default_factory=dict,
        description="General summary metrics (rows, cols, memory, etc.)"
    )
    column_profiles: List[ColumnProfile] = Field(default_factory=list)
    quality_issues: List[QualityIssue] = Field(default_factory=list)
    resource_profile: Optional[ResourceProfile] = None
    recommendation_context: Dict[str, Any] = Field(default_factory=dict)

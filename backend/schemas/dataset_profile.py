from typing import Any, Dict, List, Optional
from pydantic import Field, ConfigDict
from backend.schemas.base import BaseSchema, ConfidenceScoredModel
from backend.schemas.semantic_profile import (
    SemanticProfile,
    ColumnProfile,
    QualityIssue,
    ResourceProfile,
)
from backend.schemas.enums import ColumnType, SeverityLevel


class ColumnProfileExtended(ColumnProfile):
    """
    Unified deterministic column profile extending ColumnProfile with deep statistics,
    cardinality metrics, likelihood scores, and semantic roles as defined in Evindra Phase 2 architecture.
    """
    dtype: str = "object"
    normalized_dtype: str = "unknown"  # "numeric", "categorical", "datetime", "text", "binary", "unknown"
    row_count: int = 0
    missing_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    unique_count: int = 0
    unique_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    cardinality: int = 0
    numeric_statistics: Optional[Dict[str, Any]] = None
    median: Optional[float] = None
    quantiles: Optional[Dict[str, float]] = None
    iqr: Optional[float] = None
    kurtosis: Optional[float] = None
    outlier_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    categorical_distribution: Optional[Dict[str, float]] = None
    rare_category_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    constant_status: bool = False
    near_constant_status: bool = False
    duplicate_column_relationship: List[str] = Field(default_factory=list)
    identifier_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    datetime_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    text_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    ordinal_likelihood: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_role_hints: List[str] = Field(default_factory=list)


class TargetCandidateInfo(BaseSchema):
    """Candidate column details for target detection."""
    column: str
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    evidence: List[str] = Field(default_factory=list)
    task_type_suitability: Optional[str] = None


class DatasetProfile(SemanticProfile):
    """
    Unified, deterministic, JSON-serializable DatasetProfile for Evindra Decision Engine.
    Fully compatible with legacy SemanticProfile while serving as the authoritative Phase 2 data contract.
    """
    dataset_name: str = "dataset"
    rows: int = 0
    columns: int = 0
    row_count: int = 0
    column_count: int = 0
    memory_estimate_mb: float = 0.0
    memory_usage: float = 0.0
    dataset_wide_missingness: float = 0.0
    global_missingness: float = 0.0
    duplicate_rows: int = 0
    duplicate_row_count: int = 0
    numeric_count: int = 0
    categorical_count: int = 0
    datetime_count: int = 0
    text_count: int = 0
    numeric_columns: List[str] = Field(default_factory=list)
    categorical_columns: List[str] = Field(default_factory=list)
    datetime_columns: List[str] = Field(default_factory=list)
    text_columns: List[str] = Field(default_factory=list)
    target_column: Optional[str] = None
    target_candidate_list: List[Dict[str, Any]] = Field(default_factory=list)
    target_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    class_distribution: Optional[Dict[str, float]] = None
    imbalance_ratio: Optional[float] = None
    feature_target_relationships: Dict[str, Any] = Field(default_factory=dict)
    problem_type_candidates: List[str] = Field(default_factory=list)
    dataset_fingerprint: Optional[str] = None
    detailed_column_profiles: List[ColumnProfileExtended] = Field(default_factory=list)

    def model_post_init(self, __context: Any) -> None:
        """Sync convenience alias fields if not explicitly passed."""
        if not self.row_count and self.rows:
            self.row_count = self.rows
        elif not self.rows and self.row_count:
            self.rows = self.row_count

        if not self.column_count and self.columns:
            self.column_count = self.columns
        elif not self.columns and self.column_count:
            self.columns = self.column_count

        if not self.memory_usage and self.memory_estimate_mb:
            self.memory_usage = self.memory_estimate_mb
        elif not self.memory_estimate_mb and self.memory_usage:
            self.memory_estimate_mb = self.memory_usage

        if not self.global_missingness and self.dataset_wide_missingness:
            self.global_missingness = self.dataset_wide_missingness
        elif not self.dataset_wide_missingness and self.global_missingness:
            self.dataset_wide_missingness = self.global_missingness

        if not self.duplicate_row_count and self.duplicate_rows:
            self.duplicate_row_count = self.duplicate_rows
        elif not self.duplicate_rows and self.duplicate_row_count:
            self.duplicate_rows = self.duplicate_row_count

        if not self.target_candidates and self.target_candidate_list:
            self.target_candidates = self.target_candidate_list
        elif not self.target_candidate_list and self.target_candidates:
            self.target_candidate_list = self.target_candidates

    def to_dict(self) -> Dict[str, Any]:
        """Returns a deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")


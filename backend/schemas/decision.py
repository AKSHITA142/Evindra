import uuid
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import Field, ConfigDict
from backend.schemas.base import BaseSchema


class DecisionDomain(str, Enum):
    COLUMN_INTELLIGENCE = "column_intelligence"
    TARGET_DETECTION = "target_detection"
    LEAKAGE_DETECTION = "leakage_detection"
    MISSING_VALUE_STRATEGY = "missing_value_strategy"
    ENCODING_STRATEGY = "encoding_strategy"
    SCALING_TRANSFORMATION = "scaling_transformation"
    OUTLIER_HANDLING = "outlier_handling"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_SELECTION = "feature_selection"
    PIPELINE_STRATEGY = "pipeline_strategy"


class DecisionSource(str, Enum):
    RULE = "rule"
    RAG = "rag"
    LLM = "llm"
    USER = "user"
    SAFETY_DEFAULT = "safety_default"


class ValidationStatus(str, Enum):
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"


class DecisionRequest(BaseSchema):
    """
    Standardized internal request for a decision across any decision-producing layer.
    """
    request_id: str = Field(default_factory=lambda: f"req_{uuid.uuid4().hex[:12]}")
    domain: DecisionDomain
    column_name: Optional[str] = None
    target_column: Optional[str] = None
    feature_name: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    dataset_profile: Dict[str, Any] = Field(default_factory=dict)


class DecisionResult(BaseSchema):
    """
    Standardized internal decision object used by ALL decision-producing layers
    (RuleEngine, RAG, LLM, User Fallback).
    """
    decision_id: str = Field(default_factory=lambda: f"dec_{uuid.uuid4().hex[:12]}")
    domain: DecisionDomain
    decision: str
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    reasoning: str = ""
    evidence: List[Any] = Field(default_factory=list)
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    source: DecisionSource = DecisionSource.RULE
    requires_validation: bool = True
    validation_status: ValidationStatus = ValidationStatus.PENDING
    warnings: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    from pydantic import field_validator

    @field_validator("domain", mode="before")
    @classmethod
    def validate_domain(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            for member in DecisionDomain:
                if member.value.lower() == v_lower or member.name.lower() == v_lower:
                    return member
        return v

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            for member in DecisionSource:
                if member.value.lower() == v_lower or member.name.lower() == v_lower:
                    return member
        return v

    @field_validator("validation_status", mode="before")
    @classmethod
    def validate_validation_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_lower = v.lower().strip()
            for member in ValidationStatus:
                if member.value.lower() == v_lower or member.name.lower() == v_lower:
                    return member
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Returns deterministic JSON-serializable dictionary representation."""
        return self.model_dump(mode="json")


class UserFallbackRequest(BaseSchema):
    """
    Structured fallback prompt sent when Rule, RAG, and LLM confidence are below thresholds.
    """
    request_id: str = Field(default_factory=lambda: f"ufb_{uuid.uuid4().hex[:12]}")
    domain: DecisionDomain
    recommended_decision: str
    confidence: float
    reasoning: str
    alternatives: List[Dict[str, Any]] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    column_name: Optional[str] = None
    default_option: str


class UserFallbackResponse(BaseSchema):
    """
    User response/override selection for a UserFallbackRequest.
    """
    request_id: str
    selected_decision: str
    user_notes: Optional[str] = None
    overridden: bool = False
    rejected: bool = False
    alternative_decision: Optional[str] = None

import pytest
from pydantic import ValidationError

from backend.schemas.decision import (
    DecisionDomain,
    DecisionSource,
    ValidationStatus,
    DecisionRequest,
    DecisionResult,
)


def test_decision_result_structure_and_serialization():
    """Verify DecisionResult contains all mandatory fields and serializes to JSON dict."""
    result = DecisionResult(
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        decision="median_imputation",
        confidence=0.92,
        reasoning="Numeric feature with moderate right-skew and < 5% missing values",
        evidence=["distribution_skew=1.45", "missing_ratio=0.03"],
        alternatives=[{"strategy": "mean_imputation", "confidence": 0.70}],
        source=DecisionSource.RULE,
        requires_validation=True,
        validation_status=ValidationStatus.PENDING,
        warnings=["Check for extreme outliers post-imputation"],
    )

    assert result.decision_id.startswith("dec_")
    assert result.domain == DecisionDomain.MISSING_VALUE_STRATEGY
    assert result.decision == "median_imputation"
    assert result.confidence == 0.92
    assert result.source == DecisionSource.RULE

    # JSON Dict Serialization
    serialized = result.to_dict()
    assert isinstance(serialized, dict)
    assert serialized["decision_id"] == result.decision_id
    assert serialized["domain"] == "missing_value_strategy"
    assert serialized["source"] == "rule"
    assert serialized["validation_status"] == "pending"


def test_supported_decision_domains():
    """Verify all 10 mandated decision domains are supported."""
    domains = [
        DecisionDomain.COLUMN_INTELLIGENCE,
        DecisionDomain.TARGET_DETECTION,
        DecisionDomain.LEAKAGE_DETECTION,
        DecisionDomain.MISSING_VALUE_STRATEGY,
        DecisionDomain.ENCODING_STRATEGY,
        DecisionDomain.SCALING_TRANSFORMATION,
        DecisionDomain.OUTLIER_HANDLING,
        DecisionDomain.FEATURE_ENGINEERING,
        DecisionDomain.FEATURE_SELECTION,
        DecisionDomain.PIPELINE_STRATEGY,
    ]
    assert len(domains) == 10

    for domain in domains:
        res = DecisionResult(
            domain=domain,
            decision="test_action",
            confidence=0.85,
        )
        assert res.domain == domain


def test_confidence_validation():
    """Verify confidence score is bounded between 0.0 and 1.0."""
    with pytest.raises(ValidationError):
        DecisionResult(
            domain=DecisionDomain.ENCODING_STRATEGY,
            decision="one_hot",
            confidence=1.5,  # Out of range
        )

    with pytest.raises(ValidationError):
        DecisionResult(
            domain=DecisionDomain.ENCODING_STRATEGY,
            decision="one_hot",
            confidence=-0.1,  # Out of range
        )


def test_decision_request():
    """Verify DecisionRequest structure."""
    req = DecisionRequest(
        domain=DecisionDomain.ENCODING_STRATEGY,
        column_name="city",
        context={"cardinality": 4},
        dataset_profile={"rows": 500},
    )
    assert req.request_id.startswith("req_")
    assert req.domain == DecisionDomain.ENCODING_STRATEGY
    assert req.column_name == "city"

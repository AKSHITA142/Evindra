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


def test_invalid_source_validation():
    """Verify invalid source string raises ValidationError."""
    with pytest.raises(ValidationError):
        DecisionResult(
            domain=DecisionDomain.ENCODING_STRATEGY,
            decision="one_hot",
            confidence=0.8,
            source="invalid_source_value",
        )


def test_invalid_domain_validation():
    """Verify invalid domain string raises ValidationError."""
    with pytest.raises(ValidationError):
        DecisionResult(
            domain="invalid_quantum_domain",
            decision="one_hot",
            confidence=0.8,
        )


def test_missing_required_fields_validation():
    """Verify missing required fields (domain, decision, confidence) raises ValidationError."""
    with pytest.raises(ValidationError):
        DecisionResult(
            domain=DecisionDomain.ENCODING_STRATEGY,
            # Missing decision and confidence
        )


def test_serialization_and_deserialization():
    """Verify round-trip serialization and deserialization of DecisionResult."""
    original = DecisionResult(
        domain="scaling_transformation",
        decision="standard_scaler",
        confidence=0.95,
        reasoning="Normally distributed numeric feature",
        evidence=[{"mean": 0.0, "std": 1.0}],
        alternatives=[{"strategy": "minmax_scaler"}],
        source="rule",
    )

    json_str = original.model_dump_json()
    reconstructed = DecisionResult.model_validate_json(json_str)

    assert reconstructed.decision_id == original.decision_id
    assert reconstructed.domain == DecisionDomain.SCALING_TRANSFORMATION
    assert reconstructed.decision == "standard_scaler"
    assert reconstructed.confidence == 0.95
    assert reconstructed.source == DecisionSource.RULE


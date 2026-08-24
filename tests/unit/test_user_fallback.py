import pytest
from unittest.mock import MagicMock

from backend.engine.decision_orchestrator import DecisionOrchestrator
from backend.engine.user_fallback import UserFallbackHandler
from backend.core.confidence_policy import ConfidencePolicy, DomainConfidenceThresholds
from backend.schemas.dataset_profile import ColumnProfileExtended
from backend.schemas.decision import (
    DecisionDomain,
    DecisionSource,
    DecisionResult,
    UserFallbackRequest,
    UserFallbackResponse,
)
from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService
from backend.services.rag.decision_service import LLMDecisionService, PreprocessingRecommendation


def test_user_fallback_handler_request_creation():
    """Verify UserFallbackHandler constructs a valid UserFallbackRequest from a DecisionResult."""
    handler = UserFallbackHandler()

    rule_res = DecisionResult(
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        decision="IMPUTE_MEAN",
        confidence=0.50,
        reasoning="Low confidence numeric imputation",
        alternatives=[{"strategy": "IMPUTE_MEDIAN"}],
        warnings=["Low confidence rule"],
    )

    req = handler.create_fallback_request(rule_res, column_name="income")

    assert req.domain == DecisionDomain.MISSING_VALUE_STRATEGY
    assert req.recommended_decision == "IMPUTE_MEAN"
    assert req.column_name == "income"
    assert req.default_option == "IMPUTE_MEAN"
    assert len(req.alternatives) == 1


def test_user_fallback_explicit_response_resolution():
    """Verify resolve_user_fallback correctly applies explicit user override."""
    handler = UserFallbackHandler()

    rule_res = DecisionResult(
        domain=DecisionDomain.ENCODING_STRATEGY,
        decision="ONE_HOT_ENCODING",
        confidence=0.60,
        reasoning="High cardinality warning",
    )

    user_resp = UserFallbackResponse(
        request_id="ufb_123",
        selected_decision="TARGET_ENCODING",
        user_notes="Prefer target encoding to prevent dimensional explosion.",
        overridden=True,
    )

    res = handler.resolve_user_fallback(rule_res, user_response=user_resp)

    assert res.source == DecisionSource.USER
    assert res.decision == "TARGET_ENCODING"
    assert res.confidence == 1.0
    assert "TARGET_ENCODING" in res.reasoning
    assert res.metadata["user_overridden"] is True


def test_user_fallback_manual_selection_string():
    """Verify resolve_user_fallback correctly applies manual selection string."""
    handler = UserFallbackHandler()

    rule_res = DecisionResult(
        domain=DecisionDomain.SCALING_TRANSFORMATION,
        decision="STANDARD_SCALER",
        confidence=0.55,
        reasoning="Skewed distribution detected",
    )

    res = handler.resolve_user_fallback(rule_res, user_selection="ROBUST_SCALER")

    assert res.source == DecisionSource.USER
    assert res.decision == "ROBUST_SCALER"
    assert res.confidence == 1.0


def test_normal_user_decision():
    """Verify normal explicit user choice returns DecisionResult with source=USER."""
    handler = UserFallbackHandler()
    rule_res = DecisionResult(
        domain=DecisionDomain.ENCODING_STRATEGY,
        decision="ONE_HOT_ENCODING",
        confidence=0.60,
        reasoning="Moderate cardinality",
    )
    user_resp = UserFallbackResponse(
        request_id="ufb_001",
        selected_decision="ORDINAL_ENCODING",
        user_notes="Categories are ordinal: High School < Bachelor < Master < PhD",
        overridden=True,
    )
    res = handler.resolve_user_fallback(rule_res, user_response=user_resp)

    assert res.source == DecisionSource.USER
    assert res.decision == "ORDINAL_ENCODING"
    assert res.confidence == 1.0
    assert res.metadata["audit_record"]["event"] == "USER_DECISION_CAPTURED"


def test_user_rejection():
    """Verify explicit user rejection of recommended option switches decision to alternative choice."""
    handler = UserFallbackHandler()
    rule_res = DecisionResult(
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        decision="IMPUTE_MEAN",
        confidence=0.50,
        reasoning="Low confidence",
        alternatives=[{"strategy": "IMPUTE_MEDIAN"}],
    )
    user_resp = UserFallbackResponse(
        request_id="ufb_002",
        selected_decision="IMPUTE_MEAN",
        rejected=True,
        alternative_decision="IMPUTE_MEDIAN",
        user_notes="Mean is sensitive to outliers",
    )
    res = handler.resolve_user_fallback(rule_res, user_response=user_resp)

    assert res.source == DecisionSource.USER
    assert res.decision == "IMPUTE_MEDIAN"
    assert res.metadata["user_rejected"] is True


def test_invalid_user_choice():
    """Verify invalid user choice generates warning log while capturing selection."""
    handler = UserFallbackHandler()
    rule_res = DecisionResult(
        domain=DecisionDomain.SCALING_TRANSFORMATION,
        decision="STANDARD_SCALER",
        confidence=0.55,
        alternatives=[{"strategy": "ROBUST_SCALER"}],
    )
    res = handler.resolve_user_fallback(rule_res, user_selection="CUSTOM_UNSUPPORTED_SCALER")

    assert res.source == DecisionSource.USER
    assert res.decision == "CUSTOM_UNSUPPORTED_SCALER"
    assert any("not among initial recommendations" in w for w in res.warnings)


def test_timeout_fallback():
    """Verify when user interaction times out, system applies safety_default."""
    handler = UserFallbackHandler()
    rule_res = DecisionResult(
        domain=DecisionDomain.OUTLIER_HANDLING,
        decision="CLIP_IQR",
        confidence=0.60,
    )
    # Auto approve safety default on timeout
    res = handler.resolve_user_fallback(rule_res, auto_approve_default=True)

    assert res.source == DecisionSource.SAFETY_DEFAULT
    assert res.decision == "CLIP_IQR"
    assert res.metadata["audit_record"]["event"] == "SAFETY_DEFAULT_APPLIED"


def test_missing_input_awaiting_approval():
    """Verify missing input when auto_approve_default=False returns pending status."""
    handler = UserFallbackHandler()
    rule_res = DecisionResult(
        domain=DecisionDomain.PIPELINE_STRATEGY,
        decision="XGBOOST_BASELINE",
        confidence=0.50,
    )
    res = handler.resolve_user_fallback(rule_res, auto_approve_default=False)

    assert res.source == DecisionSource.USER
    assert res.metadata["awaiting_user_input"] is True
    assert any("Awaiting explicit user input" in w for w in res.warnings)


def test_low_confidence_escalation():
    """Verify end-to-end Orchestrator low confidence escalates to User Fallback layer."""
    strict_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.99, llm_accept=0.99)
        }
    )
    mock_rag = MagicMock(spec=HybridRetrievalService)
    mock_rag.retrieve_relevant_scenarios.return_value = []
    mock_llm = MagicMock(spec=LLMDecisionService)
    mock_llm.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="IMPUTE_MEDIAN",
        confidence_score=0.70,
        reasoning="Uncertain recommendation.",
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=strict_policy,
        hybrid_retrieval_service=mock_rag,
        llm_decision_service=mock_llm,
    )
    col_profile = ColumnProfileExtended(name="income", normalized_dtype="categorical", missing_ratio=0.15)
    res = orchestrator.evaluate_decision(
        DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile, user_selection="IMPUTE_ZERO"
    )

    assert res.source == DecisionSource.USER
    assert res.decision == "IMPUTE_ZERO"


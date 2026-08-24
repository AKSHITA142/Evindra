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


def test_user_fallback_auto_approve_default():
    """Verify resolve_user_fallback auto-approves default option when non-interactive."""
    handler = UserFallbackHandler()

    rule_res = DecisionResult(
        domain=DecisionDomain.OUTLIER_HANDLING,
        decision="CLIP_IQR",
        confidence=0.65,
        reasoning="Outlier ratio 0.08",
    )

    res = handler.resolve_user_fallback(rule_res, auto_approve_default=True)

    assert res.source == DecisionSource.USER
    assert res.decision == "CLIP_IQR"
    assert res.metadata["auto_approved"] is True
    assert any("auto-approved" in w for w in res.warnings)


def test_orchestrator_end_to_end_user_fallback_escalation():
    """Verify Orchestrator escalates Rule -> RAG -> LLM -> USER when all preceding layers have low confidence."""
    # Policy requiring 0.99 for all layers to force escalation to USER
    strict_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(
                rule_strong=0.99,
                rag_strong=0.99,
                llm_accept=0.99,
            )
        }
    )

    mock_rag = MagicMock(spec=HybridRetrievalService)
    mock_rag.retrieve_relevant_scenarios.return_value = [
        {"scenario_id": "scen_1", "similarity_score": 0.50, "final_score": 0.50}
    ]

    mock_llm = MagicMock(spec=LLMDecisionService)
    mock_llm.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="IMPUTE_MEDIAN_ROBUST",
        confidence_score=0.75,
        reasoning="Moderate confidence LLM recommendation.",
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=strict_policy,
        hybrid_retrieval_service=mock_rag,
        llm_decision_service=mock_llm,
    )

    col_profile = ColumnProfileExtended(name="income", normalized_dtype="numeric", missing_ratio=0.10)

    # With user_selection provided
    res = orchestrator.evaluate_decision(
        DecisionDomain.MISSING_VALUE_STRATEGY,
        col_profile=col_profile,
        user_selection="IMPUTE_ZERO",
    )

    assert res.source == DecisionSource.USER
    assert res.decision == "IMPUTE_ZERO"
    assert res.confidence == 1.0

    # Without user_selection provided (auto-approves default LLM decision)
    res_auto = orchestrator.evaluate_decision(
        DecisionDomain.MISSING_VALUE_STRATEGY,
        col_profile=col_profile,
        auto_approve_default=True,
    )

    assert res_auto.source == DecisionSource.USER
    assert res_auto.decision == "IMPUTE_MEDIAN_ROBUST"
    assert res_auto.metadata["auto_approved"] is True

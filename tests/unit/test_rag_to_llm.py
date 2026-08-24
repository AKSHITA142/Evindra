import pytest
from unittest.mock import MagicMock

from backend.engine.decision_orchestrator import DecisionOrchestrator
from backend.core.confidence_policy import ConfidencePolicy, DomainConfidenceThresholds
from backend.schemas.dataset_profile import ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionSource, DecisionResult
from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService
from backend.services.rag.decision_service import LLMDecisionService, PreprocessingRecommendation


def test_rag_low_confidence_escalates_to_llm_and_accepts_llm():
    """Verify when Rule and RAG confidence are low, decision escalates to LLM and accepts LLM recommendation."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(
                rule_strong=0.99,
                rag_strong=0.95,
                llm_accept=0.85,
                llm_review=0.70,
            )
        }
    )

    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    # RAG returns low score 0.50 (< 0.95 rag_strong)
    mock_rag_service.retrieve_relevant_scenarios.return_value = [
        {
            "scenario_id": "scen_low",
            "domain": "missing_value_strategy",
            "similarity_score": 0.50,
            "final_score": 0.50,
            "metadata": {"recommendation": "IMPUTE_MODE"},
        }
    ]

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="IMPUTE_KNN",
        confidence_score=0.88,
        reasoning="KNN imputation models multi-feature correlation effectively for this dataset.",
        evidence_scenarios=["scen_low"],
        alternative_strategies=[{"strategy": "IMPUTE_MEDIAN", "pros": "fast", "cons": "ignores correlation"}],
        risk_analysis=["Computations scale quadratically with sample size"],
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )

    col_profile = ColumnProfileExtended(name="income", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    # Both Rule (0.95 < 0.99) and RAG (0.50 < 0.95) failed threshold -> LLM called
    mock_llm_service.generate_preprocessing_recommendation.assert_called_once()

    assert res.source == DecisionSource.LLM
    assert res.decision == "IMPUTE_KNN"
    assert res.confidence == 0.88
    assert res.reasoning.startswith("KNN imputation")
    assert res.metadata["escalate_to_user"] is False


def test_llm_low_confidence_flags_user_escalation():
    """Verify when LLM returns low confidence (< 0.70), DecisionResult marks escalate_to_user=True."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(
                rule_strong=0.99,
                rag_strong=0.95,
                llm_accept=0.85,
                llm_review=0.70,
            )
        }
    )

    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    # LLM returns low confidence 0.62 (< 0.70)
    mock_llm_service.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="AMBIGUOUS_IMPUTE",
        confidence_score=0.62,
        reasoning="Unclear feature distribution semantics.",
        evidence_scenarios=[],
        risk_analysis=["High risk of domain mismatch"],
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )

    col_profile = ColumnProfileExtended(name="ambiguous_col", normalized_dtype="categorical", missing_ratio=0.30)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    assert res.source in (DecisionSource.LLM, DecisionSource.USER)
    assert res.confidence == 0.62
    assert any("User Fallback" in w or "auto-approved" in w for w in res.warnings)


def test_llm_unavailable_graceful_fallback():
    """Verify when LLM throws an exception or is offline, Orchestrator falls back safely without crashing."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95)
        }
    )

    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.side_effect = RuntimeError("Gemini API rate limit exceeded")

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )

    col_profile = ColumnProfileExtended(name="income", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    # Pipeline does NOT crash; returns user fallback result
    assert res.source == DecisionSource.USER
    assert res.metadata["auto_approved"] is True

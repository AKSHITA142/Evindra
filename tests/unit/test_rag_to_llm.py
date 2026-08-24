import pytest
from unittest.mock import MagicMock

from backend.engine.decision_orchestrator import DecisionOrchestrator
from backend.core.confidence_policy import ConfidencePolicy, DomainConfidenceThresholds
from backend.schemas.dataset_profile import ColumnProfileExtended, DatasetProfile
from backend.schemas.decision import DecisionDomain, DecisionSource, DecisionResult
from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService
from backend.services.rag.decision_service import LLMDecisionService, PreprocessingRecommendation


def test_valid_llm_response():
    """Verify valid LLM response is accepted as DecisionResult with source=LLM."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95, llm_accept=0.85)
        }
    )
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="IMPUTE_KNN",
        confidence_score=0.88,
        reasoning="KNN imputation models multi-feature correlation effectively.",
        evidence_scenarios=["scen_101"],
        alternative_strategies=[{"strategy": "IMPUTE_MEDIAN"}],
        risk_analysis=["Quadratic complexity"],
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )
    col_profile = ColumnProfileExtended(name="income", normalized_dtype="categorical", missing_ratio=0.15)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    assert res.source == DecisionSource.LLM
    assert res.decision == "IMPUTE_KNN"
    assert res.confidence == 0.88


def test_malformed_llm_response():
    """Verify malformed JSON from LLM is handled safely via fallback."""
    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.side_effect = ValueError("Invalid JSON token")

    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95)
        }
    )
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )
    col_profile = ColumnProfileExtended(name="income", normalized_dtype="categorical", missing_ratio=0.15)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    # Malformed response handled without crashing; returns user/safety fallback
    assert res.source in (DecisionSource.USER, DecisionSource.SAFETY_DEFAULT)


def test_invalid_decision():
    """Verify invalid/empty decision from LLM triggers low confidence or fallback warning."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95, llm_accept=0.85)
        }
    )
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="",
        confidence_score=0.40,
        reasoning="Unable to select valid decision.",
        evidence_scenarios=[],
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )
    col_profile = ColumnProfileExtended(name="income", normalized_dtype="categorical", missing_ratio=0.15)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    # Escalates to user fallback due to low confidence (0.40 < 0.85)
    assert res.source in (DecisionSource.USER, DecisionSource.SAFETY_DEFAULT)


def test_low_confidence():
    """Verify LLM confidence below threshold triggers User Fallback escalation."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95, llm_accept=0.85)
        }
    )
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="IMPUTE_MEDIAN",
        confidence_score=0.55,
        reasoning="Low confidence LLM assessment.",
        evidence_scenarios=[],
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )
    col_profile = ColumnProfileExtended(name="income", normalized_dtype="categorical", missing_ratio=0.15)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    assert res.source in (DecisionSource.USER, DecisionSource.SAFETY_DEFAULT)


def test_unavailable_llm():
    """Verify offline or throwing LLM service falls back gracefully."""
    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.side_effect = RuntimeError("API key invalid or offline")

    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95)
        }
    )
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )
    col_profile = ColumnProfileExtended(name="income", normalized_dtype="categorical", missing_ratio=0.15)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    assert res.source in (DecisionSource.USER, DecisionSource.SAFETY_DEFAULT)


def test_hallucinated_column():
    """Verify LLM attempting to reference non-existent/hallucinated column is handled safely."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95)
        }
    )
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="IMPUTE_MEAN",
        confidence_score=0.90,
        reasoning="Impute hallucinated_ghost_col.",
        evidence_scenarios=[],
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )
    col_profile = ColumnProfileExtended(name="income", normalized_dtype="categorical", missing_ratio=0.15)

    dataset_prof = DatasetProfile(
        dataset_name="test",
        rows=10,
        columns=2,
        detailed_column_profiles=[col_profile, ColumnProfileExtended(name="age", normalized_dtype="numeric")],
    )

    res = orchestrator.evaluate_decision(
        DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile, dataset_profile=dataset_prof
    )
    assert res.decision is not None


def test_unsupported_operation():
    """Verify LLM choosing unsupported operation is flagged or safely resolved."""
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.ENCODING_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99, rag_strong=0.95, llm_accept=0.85)
        }
    )
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = []

    mock_llm_service = MagicMock(spec=LLMDecisionService)
    mock_llm_service.generate_preprocessing_recommendation.return_value = PreprocessingRecommendation(
        primary_recommendation="MAGIC_UNSUPPORTED_ENCODING",
        confidence_score=0.60,
        reasoning="Unsupported operation recommended by LLM.",
        evidence_scenarios=[],
        risk_analysis=["Unsupported action in pipeline validator"],
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
        llm_decision_service=mock_llm_service,
    )
    col_profile = ColumnProfileExtended(name="city", normalized_dtype="categorical", cardinality=100)
    res = orchestrator.evaluate_decision(DecisionDomain.ENCODING_STRATEGY, col_profile=col_profile)

    assert res.source in (DecisionSource.USER, DecisionSource.SAFETY_DEFAULT)



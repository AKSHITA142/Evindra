import pytest
from unittest.mock import MagicMock

from backend.engine.decision_orchestrator import DecisionOrchestrator
from backend.engine.rule_engine import RuleEngine
from backend.core.confidence_policy import ConfidencePolicy, DomainConfidenceThresholds
from backend.schemas.dataset_profile import ColumnProfileExtended, DatasetProfile
from backend.schemas.decision import DecisionDomain, DecisionSource, DecisionResult
from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService


def test_high_confidence_rule_bypasses_rag():
    """Verify high-confidence rule decision is accepted immediately without querying RAG."""
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    orchestrator = DecisionOrchestrator(hybrid_retrieval_service=mock_rag_service)

    # Numeric symmetric missing -> Rule confidence 0.95 (High)
    col_sym = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_sym)

    assert res.source == DecisionSource.RULE
    assert res.decision == "IMPUTE_MEAN"
    assert res.confidence == 0.95
    # RAG service should NOT have been called
    mock_rag_service.retrieve_relevant_scenarios.assert_not_called()


def test_low_confidence_rule_escalates_to_rag_and_accepts_rag():
    """Verify low-confidence rule decision escalates to RAG and accepts RAG scenario evidence."""
    # Policy requiring 0.98 for rule_strong, so 0.75 forces RAG escalation
    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.98, rule_acceptable=0.70, rag_strong=0.85)
        }
    )

    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.return_value = [
        {
            "scenario_id": "scen_102",
            "domain": "missing_value_strategy",
            "similarity_score": 1.0,
            "semantic_score": 1.0,
            "structured_score": 1.0,
            "validation_score": 1.0,
            "final_score": 1.0,
            "metadata": {"recommendation": "IMPUTE_MEDIAN_ROBUST"},
            "retrieval_text": "Similar dataset with missing values",
        }
    ]

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
    )

    col_profile = ColumnProfileExtended(name="income", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    # Rule gave 0.95 < 0.98, so it queried RAG
    mock_rag_service.retrieve_relevant_scenarios.assert_called_once()

    # RAG returned 0.89 >= 0.85 (rag_strong) -> Decision accepted from RAG
    assert res.source == DecisionSource.RAG
    assert res.decision == "IMPUTE_MEDIAN_ROBUST"
    assert res.confidence >= 0.85
    assert len(res.evidence) == 1
    assert res.metadata["scenario_id"] == "scen_102"


def test_rag_unavailable_graceful_fallback():
    """Verify Orchestrator gracefully falls back if RAG service throws an exception or is offline."""
    mock_rag_service = MagicMock(spec=HybridRetrievalService)
    mock_rag_service.retrieve_relevant_scenarios.side_effect = RuntimeError("Supabase connection timeout")

    custom_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(rule_strong=0.99)
        }
    )

    orchestrator = DecisionOrchestrator(
        confidence_policy=custom_policy,
        hybrid_retrieval_service=mock_rag_service,
    )

    col_profile = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1)
    res = orchestrator.evaluate_decision(DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col_profile)

    # Should not raise error, falls back safely across decision hierarchy
    assert res.source in (DecisionSource.RULE, DecisionSource.LLM, DecisionSource.USER)
    assert res.decision is not None

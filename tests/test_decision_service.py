import os
import sys
import pytest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Cloud-dependency guard — see module docstring in test_context_builder.py
try:
    import google.genai  # noqa: F401  # pyrefly: ignore[missing-import]
    _google_genai_available = True
except ImportError:
    _google_genai_available = False

_SKIP_REASON = "google-genai cloud package not installed (pip install google-genai)"

from backend.services.rag.reranker_service import retrieve_and_rerank_scenarios
from backend.services.rag.context_builder import build_rag_evidence_package
from backend.services.rag.decision_service import (
    LLMDecisionService,
    PreprocessingRecommendation,
    generate_preprocessing_recommendation,
)


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_end_to_end_rag_decision_missing_values():
    """
    Tests full end-to-end RAG pipeline for Missing Value Imputation.
    """
    dataset_profile = {
        "dataset_name": "housing_prices_sample.csv",
        "target_column": "SalePrice",
        "problem_type": "regression",
        "target_feature": "LotFrontage",
        "feature_dtype": "numeric",
        "issue_description": "Numeric column LotFrontage has 17.7% missing values requiring imputation.",
    }
    query_text = (
        "Numeric feature with a high percentage of missing values in a regression dataset. "
        "Need to determine an appropriate imputation strategy."
    )
    context = {
        "domain": "missing_value_strategy",
        "scenario_type": "missing_value",
        "problem_type": "regression",
        "column_type": "numeric",
    }

    # Step 1: Vector Retrieval + Hybrid Search + Reranking
    reranked_scenarios = retrieve_and_rerank_scenarios(query_text, context=context, top_k=3)
    assert len(reranked_scenarios) == 3

    # Step 2: Context Builder
    evidence_package = build_rag_evidence_package(dataset_profile, reranked_scenarios)
    assert evidence_package.total_evidence_count == 3

    # Step 3: LLM Decision Layer
    service = LLMDecisionService()
    recommendation = service.generate_preprocessing_recommendation(dataset_profile, evidence_package)

    assert isinstance(recommendation, PreprocessingRecommendation)
    assert recommendation.primary_recommendation is not None
    assert 0.0 <= recommendation.confidence_score <= 1.0
    assert len(recommendation.evidence_scenarios) > 0

    print("\n==================================================")
    print("  TEST 1: END-TO-END RAG DECISION — MISSING VALUES")
    print("==================================================")
    print(f"Primary Recommendation: {recommendation.primary_recommendation}")
    print(f"Confidence Score: {recommendation.confidence_score:.2f}")
    print(f"Evidence Scenarios Cited: {recommendation.evidence_scenarios}")
    print(f"Reasoning:\n{recommendation.reasoning}\n")
    print(f"Alternative Strategies: {recommendation.alternative_strategies}")
    print(f"Risk Analysis: {recommendation.risk_analysis}\n")

    # Assert cited scenario IDs are present in original reranked scenario list
    retrieved_ids = [sc["scenario_id"] for sc in reranked_scenarios]
    for cited_id in recommendation.evidence_scenarios:
        assert cited_id in retrieved_ids, f"Cited ID '{cited_id}' not found in retrieved scenario pool {retrieved_ids}"


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_end_to_end_rag_decision_categorical_encoding():
    """
    Tests full end-to-end RAG pipeline for Categorical Encoding.
    """
    dataset_profile = {
        "dataset_name": "telco_customer_churn.csv",
        "target_column": "Churn",
        "problem_type": "binary_classification",
        "target_feature": "Contract",
        "feature_dtype": "categorical",
        "issue_description": "Categorical feature Contract with low cardinality needing optimal encoding strategy.",
    }
    query_text = (
        "Categorical feature with low cardinality in a classification dataset. "
        "Need an appropriate encoding strategy."
    )
    context = {
        "domain": "encoding_strategy",
        "scenario_type": "categorical_encoding",
        "problem_type": "binary_classification",
        "column_type": "categorical",
    }

    reranked_scenarios = retrieve_and_rerank_scenarios(query_text, context=context, top_k=3)
    evidence_package = build_rag_evidence_package(dataset_profile, reranked_scenarios)

    recommendation = generate_preprocessing_recommendation(dataset_profile, evidence_package)

    assert isinstance(recommendation, PreprocessingRecommendation)
    assert recommendation.primary_recommendation is not None
    assert 0.0 <= recommendation.confidence_score <= 1.0

    print("\n==================================================")
    print("  TEST 2: END-TO-END RAG DECISION — CATEGORICAL ENCODING")
    print("==================================================")
    print(f"Primary Recommendation: {recommendation.primary_recommendation}")
    print(f"Confidence Score: {recommendation.confidence_score:.2f}")
    print(f"Evidence Scenarios Cited: {recommendation.evidence_scenarios}")
    print(f"Reasoning:\n{recommendation.reasoning}\n")


if __name__ == "__main__":
    pytest.main(["-s", __file__])

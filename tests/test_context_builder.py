import os
import sys
import pytest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# ---------------------------------------------------------------------------
# Cloud-dependency guard — skip entire module when google-genai is absent.
# These tests exercise the live RAG embedding stack and require the
# google-genai package plus valid cloud credentials at runtime.
# They are intentionally excluded from CI/offline runs.
# ---------------------------------------------------------------------------
try:
    import google.genai  # noqa: F401  # pyrefly: ignore[missing-import]
    _google_genai_available = True
except ImportError:
    _google_genai_available = False

_SKIP_REASON = "google-genai cloud package not installed (pip install google-genai)"

from backend.services.rag.reranker_service import retrieve_and_rerank_scenarios
from backend.services.rag.context_builder import RAGContextBuilder, build_rag_evidence_package


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_context_builder_missing_values():
    """
    Tests RAG Context Builder for Missing Value Imputation dataset profile.
    """
    dataset_profile = {
        "dataset_name": "house_prices.csv",
        "target_column": "SalePrice",
        "problem_type": "regression",
        "target_feature": "LotFrontage",
        "feature_dtype": "numeric",
        "issue_description": "Numeric column LotFrontage has 17.7% missing values.",
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

    # Step 1: Retrieve and rerank scenarios
    scenarios = retrieve_and_rerank_scenarios(query_text, context=context, top_k=3)

    # Step 2: Build RAG evidence package
    builder = RAGContextBuilder()
    pkg = builder.build_evidence_package(dataset_profile, scenarios)

    assert pkg.total_evidence_count == 3
    assert pkg.dataset_summary["dataset_name"] == "house_prices.csv"
    assert pkg.dataset_summary["target_feature"] == "LotFrontage"

    print("\n==================================================")
    print("  TEST 1: CONTEXT BUILDER — MISSING VALUES")
    print("==================================================")
    print("PROMPT CONTEXT STRING PREVIEW:\n")
    print(pkg.prompt_context_str)

    # Assertions on evidence package structure
    for ev in pkg.evidence_items:
        assert "scenario_id" in ev
        assert "historical_decision" in ev
        assert "recommended_action" in ev
        assert "validation_status" in ev
        assert "rank_explanation" in ev
        # Scenario ID must appear in prompt text
        assert ev["scenario_id"] in pkg.prompt_context_str

    assert "SECTION 1: CURRENT DATASET FACTS" in pkg.prompt_context_str
    assert "SECTION 2: HISTORICAL EVIDENTIAL SCENARIOS" in pkg.prompt_context_str


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_context_builder_categorical_encoding():
    """
    Tests RAG Context Builder for Categorical Encoding dataset profile.
    """
    dataset_profile = {
        "dataset_name": "customer_churn.csv",
        "target_column": "Churn",
        "problem_type": "binary_classification",
        "target_feature": "PaymentMethod",
        "feature_dtype": "categorical",
        "issue_description": "Categorical feature PaymentMethod with low cardinality needing encoding.",
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

    scenarios = retrieve_and_rerank_scenarios(query_text, context=context, top_k=3)
    pkg = build_rag_evidence_package(dataset_profile, scenarios)

    assert pkg.total_evidence_count == 3

    print("\n==================================================")
    print("  TEST 2: CONTEXT BUILDER — CATEGORICAL ENCODING")
    print("==================================================")
    print("PROMPT CONTEXT STRING PREVIEW:\n")
    print(pkg.prompt_context_str)

    assert "customer_churn.csv" in pkg.prompt_context_str
    assert "PaymentMethod" in pkg.prompt_context_str


if __name__ == "__main__":
    pytest.main(["-s", __file__])

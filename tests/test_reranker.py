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

from backend.services.rag.reranker_service import (
    ScenarioRerankerService,
    retrieve_and_rerank_scenarios,
)


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_reranker_missing_values():
    """
    Tests multi-signal reranking for Missing Value Imputation query.
    """
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
    results = retrieve_and_rerank_scenarios(query_text, context=context, top_k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    print("\n==================================================")
    print("  TEST 1: RERANKER — MISSING VALUES QUERY")
    print("==================================================")
    print(f"Query: '{query_text}'")
    print(f"Context: {context}\n")

    prev_final_score = 1.0
    for i, res in enumerate(results, 1):
        assert "scenario_id" in res
        assert "semantic_score" in res
        assert "structured_score" in res
        assert "validation_score" in res
        assert "final_score" in res
        assert "rank_explanation" in res
        assert isinstance(res["rank_explanation"], str)

        final_s = res["final_score"]
        assert final_s <= prev_final_score, f"Results not sorted descending: {final_s} > {prev_final_score}"
        prev_final_score = final_s

        print(f"[{i}] Scenario ID: {res['scenario_id']}")
        print(f"    Domain: {res['domain']} | Type: {res['scenario_type']}")
        print(
            f"    Semantic: {res['semantic_score']:.4f} | Structured: {res['structured_score']:.4f} | "
            f"Validation: {res['validation_score']:.4f} | Final: {final_s:.4f}"
        )
        print(f"    Explanation: {res['rank_explanation']}")
        print(f"    Text: {res['retrieval_text'].replace('\n', ' ')[:90]}...\n")


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_reranker_categorical_encoding():
    """
    Tests multi-signal reranking for Categorical Encoding query.
    """
    query_text = (
        "Categorical feature with low cardinality in a classification dataset. "
        "Need an appropriate encoding strategy."
    )
    context = {
        "domain": "encoding_strategy",
        "scenario_type": "categorical_encoding",
        "problem_type": "multiclass_classification",
        "column_type": "categorical",
    }
    results = retrieve_and_rerank_scenarios(query_text, context=context, top_k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    print("\n==================================================")
    print("  TEST 2: RERANKER — CATEGORICAL ENCODING QUERY")
    print("==================================================")
    print(f"Query: '{query_text}'")
    print(f"Context: {context}\n")

    prev_final_score = 1.0
    for i, res in enumerate(results, 1):
        final_s = res["final_score"]
        assert final_s <= prev_final_score, f"Results not sorted descending: {final_s} > {prev_final_score}"
        prev_final_score = final_s

        print(f"[{i}] Scenario ID: {res['scenario_id']}")
        print(f"    Domain: {res['domain']} | Type: {res['scenario_type']}")
        print(
            f"    Semantic: {res['semantic_score']:.4f} | Structured: {res['structured_score']:.4f} | "
            f"Validation: {res['validation_score']:.4f} | Final: {final_s:.4f}"
        )
        print(f"    Explanation: {res['rank_explanation']}\n")


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_reranker_custom_weights():
    """
    Tests reranking with custom signal weights prioritizing validation quality.
    """
    query_text = "Handling highly imbalanced target column in classification."
    custom_weights = {"semantic": 0.2, "structured": 0.2, "validation": 0.6}
    results = retrieve_and_rerank_scenarios(query_text, context=None, top_k=5, weights=custom_weights)

    assert len(results) == 5
    for res in results:
        assert res["validation_score"] >= 0.0
        assert res["final_score"] > 0.0


if __name__ == "__main__":
    pytest.main(["-s", __file__])

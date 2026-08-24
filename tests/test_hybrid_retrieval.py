import os
import sys
import pytest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

# Cloud-dependency guard — see module docstring in test_context_builder.py
try:
    import google.genai  # noqa: F401
    _google_genai_available = True
except ImportError:
    _google_genai_available = False

_SKIP_REASON = "google-genai cloud package not installed (pip install google-genai)"

from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService, retrieve_relevant_scenarios


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_hybrid_missing_values():
    """
    Tests Hybrid Retrieval with structured context for Missing Value Imputation.
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
    service = HybridRetrievalService()
    results = service.retrieve_relevant_scenarios(query_text, context=context, top_k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    print("\n==================================================")
    print("  TEST 1: HYBRID RETRIEVAL — MISSING VALUES")
    print("==================================================")
    print(f"Query: '{query_text}'")
    print(f"Context: {context}\n")

    prev_final_score = 1.0
    for i, res in enumerate(results, 1):
        assert "scenario_id" in res
        assert "semantic_score" in res
        assert "structured_score" in res
        assert "final_score" in res

        sem_s = res["semantic_score"]
        struct_s = res["structured_score"]
        final_s = res["final_score"]

        assert final_s <= prev_final_score, f"Results not sorted descending: {final_s} > {prev_final_score}"
        prev_final_score = final_s

        print(f"[{i}] Scenario ID: {res['scenario_id']}")
        print(f"    Domain: {res['domain']} | Type: {res['scenario_type']}")
        print(f"    Semantic: {sem_s:.4f} | Structured: {struct_s:.4f} | Final Hybrid: {final_s:.4f}")
        print(f"    Text: {res['retrieval_text'].replace('\n', ' ')[:90]}...\n")

    # Top result should have a high structured score due to exact domain/type/problem match
    assert results[0]["structured_score"] >= 0.75, f"Expected high structured match score, got {results[0]['structured_score']}"


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_hybrid_categorical_encoding():
    """
    Tests Hybrid Retrieval with structured context for Categorical Encoding.
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
    service = HybridRetrievalService()
    results = service.retrieve_relevant_scenarios(query_text, context=context, top_k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    print("\n==================================================")
    print("  TEST 2: HYBRID RETRIEVAL — CATEGORICAL ENCODING")
    print("==================================================")
    print(f"Query: '{query_text}'")
    print(f"Context: {context}\n")

    prev_final_score = 1.0
    for i, res in enumerate(results, 1):
        sem_s = res["semantic_score"]
        struct_s = res["structured_score"]
        final_s = res["final_score"]

        assert final_s <= prev_final_score, f"Results not sorted descending: {final_s} > {prev_final_score}"
        prev_final_score = final_s

        print(f"[{i}] Scenario ID: {res['scenario_id']}")
        print(f"    Domain: {res['domain']} | Type: {res['scenario_type']}")
        print(f"    Semantic: {sem_s:.4f} | Structured: {struct_s:.4f} | Final Hybrid: {final_s:.4f}")
        print(f"    Text: {res['retrieval_text'].replace('\n', ' ')[:90]}...\n")

    assert results[0]["structured_score"] >= 0.75, f"Expected high structured match score, got {results[0]['structured_score']}"


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_hybrid_empty_context_graceful_fallback():
    """
    Tests Hybrid Retrieval with empty / None context.
    Verifies that structured_score defaults to neutral (0.5) and results remain valid.
    """
    query_text = "Handling highly imbalanced target column in classification."
    results = retrieve_relevant_scenarios(query_text, context=None, top_k=5)

    assert len(results) == 5
    for res in results:
        assert res["structured_score"] == 0.5, f"Expected neutral 0.5 structured score for empty context, got {res['structured_score']}"
        assert res["final_score"] > 0.0


if __name__ == "__main__":
    pytest.main(["-s", __file__])

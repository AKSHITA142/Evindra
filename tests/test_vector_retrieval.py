import os
import sys
import pytest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.services.rag.retrieval_service import VectorRetrievalService, search_similar_scenarios
from backend.services.rag.embedding_service import EXPECTED_DIMENSION


def test_missing_values_retrieval():
    """
    Tests vector retrieval for missing value imputation scenario query.
    Verifies that returned scenarios match missing value strategy domains.
    """
    query_text = (
        "Numeric feature with a high percentage of missing values in a regression dataset. "
        "Need to determine an appropriate imputation strategy."
    )
    service = VectorRetrievalService()
    results = service.search_similar_scenarios(query_text, top_k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    print("\n==================================================")
    print("  TEST 1: MISSING VALUE IMPUTATION QUERY RESULTS")
    print("==================================================")
    print(f"Query: '{query_text}'\n")

    prev_score = 1.0
    for i, res in enumerate(results, 1):
        assert "scenario_id" in res
        assert "domain" in res
        assert "scenario_type" in res
        assert "retrieval_text" in res
        assert "metadata" in res
        assert "similarity_score" in res
        assert isinstance(res["similarity_score"], float)

        score = res["similarity_score"]
        assert score <= prev_score, f"Results not sorted descending: {score} > {prev_score}"
        prev_score = score

        print(f"[{i}] Scenario ID: {res['scenario_id']}")
        print(f"    Domain: {res['domain']} | Type: {res['scenario_type']}")
        print(f"    Similarity Score: {score:.6f}")
        print(f"    Text: {res['retrieval_text'].replace('\n', ' ')[:100]}...\n")

    # Verify domain relevance
    top_domain = results[0]["domain"]
    top_type = results[0]["scenario_type"]
    assert top_domain in ("missing_value_strategy", "column_intelligence", "pipeline_strategy")
    assert "missing" in top_type or "imput" in results[0]["retrieval_text"].lower() or "missing" in results[0]["retrieval_text"].lower()


def test_categorical_encoding_retrieval():
    """
    Tests vector retrieval for categorical encoding scenario query.
    Verifies that returned scenarios match categorical encoding / column role domains.
    """
    query_text = (
        "Categorical feature with low cardinality in a classification dataset. "
        "Need an appropriate encoding strategy."
    )
    service = VectorRetrievalService()
    results = service.search_similar_scenarios(query_text, top_k=5)

    assert len(results) == 5, f"Expected 5 results, got {len(results)}"

    print("\n==================================================")
    print("  TEST 2: CATEGORICAL ENCODING QUERY RESULTS")
    print("==================================================")
    print(f"Query: '{query_text}'\n")

    prev_score = 1.0
    for i, res in enumerate(results, 1):
        score = res["similarity_score"]
        assert score <= prev_score, f"Results not sorted descending: {score} > {prev_score}"
        prev_score = score

        print(f"[{i}] Scenario ID: {res['scenario_id']}")
        print(f"    Domain: {res['domain']} | Type: {res['scenario_type']}")
        print(f"    Similarity Score: {score:.6f}")
        print(f"    Text: {res['retrieval_text'].replace('\n', ' ')[:100]}...\n")

    # Verify domain relevance
    top_type = results[0]["scenario_type"]
    top_domain = results[0]["domain"]
    assert top_domain in ("column_intelligence", "feature_encoding", "pipeline_strategy") or "encoding" in top_type or "role" in top_type or "cardinality" in results[0]["retrieval_text"].lower() or "categorical" in results[0]["retrieval_text"].lower()


if __name__ == "__main__":
    pytest.main(["-s", __file__])

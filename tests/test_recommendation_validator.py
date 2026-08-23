import os
import sys
import pytest

# Ensure workspace root is in sys.path
WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

from backend.services.rag.reranker_service import retrieve_and_rerank_scenarios
from backend.services.rag.context_builder import build_rag_evidence_package
from backend.services.rag.decision_service import (
    LLMDecisionService,
    PreprocessingRecommendation,
    generate_preprocessing_recommendation,
)
from backend.services.rag.recommendation_validator import (
    RecommendationValidatorService,
    RecommendationValidationReport,
    validate_recommendation,
)


def test_validate_valid_recommendation():
    """
    Tests validation of a valid, evidence-grounded recommendation for Missing Value Imputation.
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

    # Step 1: Retrieval + Reranking + Context Builder
    reranked = retrieve_and_rerank_scenarios(query_text, context=context, top_k=3)
    pkg = build_rag_evidence_package(dataset_profile, reranked)

    # Step 2: LLM Decision Layer
    rec = generate_preprocessing_recommendation(dataset_profile, pkg)

    # Step 3: Recommendation Validation Layer
    report = validate_recommendation(dataset_profile, rec, pkg)

    assert isinstance(report, RecommendationValidationReport)
    assert report.validation_status in ("PASSED", "WARNING")
    assert len(report.checks_performed) == 4

    print("\n==================================================")
    print("  TEST 1: VALIDATION REPORT — VALID RECOMMENDATION")
    print("==================================================")
    print(f"Validation Status: {report.validation_status}")
    print("Checks Performed:")
    for chk in report.checks_performed:
        print(f"  - [{ 'PASS' if chk['passed'] else 'FAIL' }] {chk['check']}: {chk['details']}")
    print(f"\nExperimental Metrics: {report.experimental_validation}")
    print(f"Executable Pipeline Spec:\n{report.executable_pipeline_spec}\n")

    assert report.experimental_validation["executed"] is True
    assert report.experimental_validation["improvement"] > 0.0
    assert report.executable_pipeline_spec["target_column"] == "LotFrontage"


def test_validate_type_mismatch_recommendation():
    """
    Tests validation of an incompatible recommendation (e.g. Robust Scaling applied to a Categorical column).
    """
    dataset_profile = {
        "dataset_name": "customer_churn.csv",
        "target_column": "Churn",
        "problem_type": "binary_classification",
        "target_feature": "PaymentMethod",
        "feature_dtype": "categorical",
        "issue_description": "Categorical feature PaymentMethod with low cardinality.",
    }
    reranked = retrieve_and_rerank_scenarios("Categorical feature encoding", context=None, top_k=3)
    pkg = build_rag_evidence_package(dataset_profile, reranked)

    # Incompatible recommendation object
    incompatible_rec = PreprocessingRecommendation(
        primary_recommendation="ROBUST_SCALING",  # Scaling is for numeric columns, not categorical
        confidence_score=0.90,
        reasoning="Testing invalid type compatibility check",
        evidence_scenarios=[reranked[0]["scenario_id"]],
        alternative_strategies=[],
        risk_analysis=[],
    )

    validator = RecommendationValidatorService()
    report = validator.validate_recommendation(dataset_profile, incompatible_rec, pkg)

    assert report.validation_status == "FAILED"
    type_chk = next(c for c in report.checks_performed if c["check"] == "type_compatibility")
    assert type_chk["passed"] is False

    print("\n==================================================")
    print("  TEST 2: VALIDATION REPORT — TYPE MISMATCH CHECK")
    print("==================================================")
    print(f"Validation Status: {report.validation_status} (Correctly failed on type mismatch!)")
    print(f"Failed Check Details: {type_chk['details']}\n")


if __name__ == "__main__":
    pytest.main(["-s", __file__])

import os
import sys
import json
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

from backend.services.rag.pipeline_planner import (
    EvindraPipelinePlanner,
    EvindraPreprocessingPlan,
    generate_evindra_preprocessing_plan,
)


@pytest.mark.skipif(not _google_genai_available, reason=_SKIP_REASON)
def test_full_evindra_pipeline_planner():
    """
    Tests complete end-to-end Evindra RAG Pipeline Planner across a multi-issue dataset.
    """
    dataset_profile = {
        "dataset_name": "house_prices_full.csv",
        "target_column": "SalePrice",
        "problem_type": "regression",
    }
    column_issues = [
        {
            "column": "LotFrontage",
            "issue_type": "missing_value",
            "domain": "missing_value_strategy",
            "column_type": "numeric",
            "severity": "high",
            "query_text": "Numeric feature LotFrontage has 17.7% missing values requiring an imputation strategy.",
        },
        {
            "column": "Neighborhood",
            "issue_type": "categorical_encoding",
            "domain": "encoding_strategy",
            "column_type": "categorical",
            "severity": "medium",
            "query_text": "Categorical feature Neighborhood with 25 unique categories requiring optimal encoding.",
        },
        {
            "column": "Electrical",
            "issue_type": "missing_value",
            "domain": "missing_value_strategy",
            "column_type": "categorical",
            "severity": "low",
            "query_text": "Categorical feature Electrical has rare missing values requiring mode or constant imputation.",
        },
    ]

    planner = EvindraPipelinePlanner()
    plan = planner.generate_evindra_preprocessing_plan(dataset_profile, column_issues, top_k_evidence=3)

    assert isinstance(plan, EvindraPreprocessingPlan)
    assert plan.dataset_name == "house_prices_full.csv"
    assert plan.problem_type == "regression"
    assert len(plan.steps) == 3

    print("\n==================================================")
    print("  TEST: EVINDRA END-TO-END PREPROCESSING PLAN")
    print("==================================================")
    print(f"Plan ID: {plan.plan_id}")
    print(f"Created At: {plan.created_at}")
    print(f"Pipeline Validation: {plan.pipeline_validation}\n")

    for step in plan.steps:
        assert step.step_id >= 1
        assert step.target_column in ("LotFrontage", "Neighborhood", "Electrical")
        assert step.recommended_action is not None
        assert step.validation_status in ("PASSED", "WARNING")
        assert len(step.evidence_scenario_ids) > 0

        print(f"--- [Step {step.step_id}] Target: {step.target_column} ({step.issue_type}) ---")
        print(f"    Action: {step.recommended_action} (Confidence: {step.confidence_score:.2f})")
        print(f"    Validation Status: {step.validation_status}")
        print(f"    Cited Scenarios: {step.evidence_scenario_ids}")
        print(f"    Reasoning: {step.reasoning[:120]}...")
        print(f"    Pipeline Spec: {step.transformation_spec}\n")

    # JSON export assertion
    json_output = plan.to_json_str()
    assert "plan_id" in json_output
    assert "pipeline_validation" in json_output
    print("Full JSON Plan Output Length:", len(json_output))


if __name__ == "__main__":
    pytest.main(["-s", __file__])

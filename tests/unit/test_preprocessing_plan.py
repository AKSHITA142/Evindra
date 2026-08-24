import pytest
from unittest.mock import MagicMock

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionSource, DecisionResult
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.engine.preprocessing_plan_builder import PreprocessingPlanBuilder
from backend.engine.decision_orchestrator import DecisionOrchestrator


def test_preprocessing_plan_schema():
    """Verify PreprocessingPlan schema, serialization, and legacy operations converter."""
    step1 = PreprocessingStep(
        step_number=1,
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        action="IMPUTE_MEDIAN",
        columns=["age"],
        decision_id="dec_001",
        decision_source=DecisionSource.RULE,
        confidence=0.95,
        reasoning="Symmetric numeric distribution.",
    )
    step2 = PreprocessingStep(
        step_number=2,
        domain=DecisionDomain.ENCODING_STRATEGY,
        action="ONE_HOT_ENCODING",
        columns=["gender"],
        decision_id="dec_002",
        decision_source=DecisionSource.RAG,
        confidence=0.88,
        reasoning="Low cardinality binary category.",
    )

    plan = PreprocessingPlan(
        dataset_name="titanic",
        target_column="survived",
        problem_type="binary_classification",
        steps=[step1, step2],
        overall_confidence=0.915,
    )

    p_dict = plan.to_dict()
    assert p_dict["dataset_name"] == "titanic"
    assert len(p_dict["steps"]) == 2

    ops = plan.to_experiment_operations()
    assert len(ops) == 2
    assert ops[0]["type"] == "missing_value_strategy"
    assert ops[0]["method"] == "IMPUTE_MEDIAN"
    assert ops[0]["params"]["columns"] == ["age"]


def test_preprocessing_plan_builder_full_pipeline():
    """Verify PreprocessingPlanBuilder generates ordered plan from DatasetProfile."""
    col_age = ColumnProfileExtended(
        name="age",
        normalized_dtype="numeric",
        missing_ratio=0.15,
        skewness=0.1,
        outlier_ratio=0.03,
    )
    col_gender = ColumnProfileExtended(
        name="gender",
        normalized_dtype="categorical",
        distinct_count=2,
        missing_ratio=0.0,
    )
    col_income = ColumnProfileExtended(
        name="income",
        normalized_dtype="numeric",
        skewness=2.5,
        missing_ratio=0.0,
    )

    dataset_prof = DatasetProfile(
        dataset_name="customer_churn",
        rows=1000,
        columns=3,
        detailed_column_profiles=[col_age, col_gender, col_income],
        target_column="churn",
        problem_type="binary_classification",
    )

    builder = PreprocessingPlanBuilder()
    plan = builder.build_plan(dataset_prof, model_family="tree")

    assert isinstance(plan, PreprocessingPlan)
    assert plan.dataset_name == "customer_churn"
    assert len(plan.steps) >= 3

    step_domains = [s.domain for s in plan.steps]
    assert DecisionDomain.MISSING_VALUE_STRATEGY in step_domains
    assert DecisionDomain.ENCODING_STRATEGY in step_domains
    assert DecisionDomain.SCALING_TRANSFORMATION in step_domains

    summary = plan.decisions_summary
    assert summary["total_decisions"] > 0
    assert "sources_breakdown" in summary
    assert plan.overall_confidence > 0.0

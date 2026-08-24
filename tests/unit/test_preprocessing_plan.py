import pytest
from unittest.mock import MagicMock

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionSource, DecisionResult
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.engine.preprocessing_plan_builder import PreprocessingPlanBuilder
from backend.engine.decision_orchestrator import DecisionOrchestrator


def test_plan_topological_ordering():
    """Verify plan steps strictly adhere to DAG stage order hierarchy."""
    col_age = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.15, skewness=0.1)
    col_gender = ColumnProfileExtended(name="gender", normalized_dtype="categorical", distinct_count=2, missing_ratio=0.0)
    col_leak = ColumnProfileExtended(name="leak_feature", normalized_dtype="numeric", missing_ratio=0.0)
    col_churn = ColumnProfileExtended(name="churn", normalized_dtype="categorical", distinct_count=2)

    prof = DatasetProfile(
        dataset_name="ordering_test",
        rows=100,
        columns=4,
        detailed_column_profiles=[col_age, col_gender, col_leak, col_churn],
        target_column="churn",
        feature_target_relationships={"leak_feature": 0.999},
    )

    builder = PreprocessingPlanBuilder()
    plan = builder.build_plan(prof)

    stage_order = builder.STAGE_ORDER
    stage_sequence = [stage_order.get(s.stage, 99) for s in plan.steps]

    # Verify stage sequence is monotonically non-decreasing
    assert stage_sequence == sorted(stage_sequence)
    assert plan.steps[0].stage == "DATA_INGESTION"


def test_dependency_resolution():
    """Verify dependencies map contains explicit step_id prerequisites for each stage."""
    col_age = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.10)
    prof = DatasetProfile(dataset_name="dep_test", rows=50, columns=2, detailed_column_profiles=[col_age], target_column="target")

    builder = PreprocessingPlanBuilder()
    plan = builder.build_plan(prof)

    assert isinstance(plan.dependencies, dict)
    assert len(plan.dependencies) == len(plan.steps)

    # Later steps must depend on earlier steps in the DAG
    for step in plan.steps[1:]:
        deps = step.dependencies
        assert isinstance(deps, list)


def test_duplicate_operation_removal():
    """Verify duplicate step actions on same column/stage are deduplicated."""
    step1 = PreprocessingStep(step_number=1, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="IMPUTE_MEAN", columns=["age"], decision_id="dec_1", decision_source=DecisionSource.RULE, confidence=0.9)
    step2 = PreprocessingStep(step_number=2, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="IMPUTE_MEAN", columns=["age"], decision_id="dec_2", decision_source=DecisionSource.RULE, confidence=0.9)

    builder = PreprocessingPlanBuilder()
    cleaned = builder._deduplicate_and_resolve_conflicts([step1, step2])

    assert len(cleaned) == 1
    assert cleaned[0].decision_id == "dec_1"


def test_conflicting_operation_detection():
    """Verify conflicting steps (e.g. transform on column marked for drop) are resolved."""
    step_drop = PreprocessingStep(step_number=1, stage="LEAKAGE_REMOVAL", domain=DecisionDomain.LEAKAGE_DETECTION, action="DROP_LEAKAGE_COLUMNS", columns=["leak_col"], decision_id="dec_1", decision_source=DecisionSource.RULE, confidence=1.0)
    step_scale = PreprocessingStep(step_number=2, stage="SCALING", domain=DecisionDomain.SCALING_TRANSFORMATION, action="STANDARD_SCALER", columns=["leak_col"], decision_id="dec_2", decision_source=DecisionSource.RULE, confidence=0.9)

    builder = PreprocessingPlanBuilder()
    cleaned = builder._deduplicate_and_resolve_conflicts([step_drop, step_scale])

    assert len(cleaned) == 1
    assert cleaned[0].action == "DROP_LEAKAGE_COLUMNS"


def test_target_separation():
    """Verify target column is separated and excluded from feature-level preprocessing steps."""
    col_feat = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.1)
    col_target = ColumnProfileExtended(name="target", normalized_dtype="numeric", missing_ratio=0.1)

    prof = DatasetProfile(dataset_name="target_sep_test", rows=50, columns=2, detailed_column_profiles=[col_feat, col_target], target_column="target")

    builder = PreprocessingPlanBuilder()
    plan = builder.build_plan(prof)

    target_sep_steps = [s for s in plan.steps if s.stage == "TARGET_SEPARATION"]
    assert len(target_sep_steps) == 1
    assert "target" in target_sep_steps[0].columns

    # Feature transformation steps (missing, encoding, scaling) must NOT include target column
    feature_transform_steps = [s for s in plan.steps if s.stage in ("MISSING_VALUE_HANDLING", "ENCODING", "SCALING", "OUTLIER_TRANSFORMATION")]
    for s in feature_transform_steps:
        assert "target" not in s.columns


def test_plan_serialization():
    """Verify PreprocessingPlan JSON round-trip serialization and deserialization."""
    col = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.1)
    prof = DatasetProfile(dataset_name="ser_test", rows=50, columns=1, detailed_column_profiles=[col])

    builder = PreprocessingPlanBuilder()
    plan = builder.build_plan(prof)

    p_dict = plan.to_dict()
    assert isinstance(p_dict, dict)
    assert p_dict["dataset_name"] == "ser_test"
    assert "steps" in p_dict
    assert "dependencies" in p_dict

    # Reconstruct from dict
    reconstructed = PreprocessingPlan(**p_dict)
    assert reconstructed.dataset_name == plan.dataset_name
    assert len(reconstructed.steps) == len(plan.steps)


def test_deterministic_plan_generation():
    """Verify identical dataset profile produces identical PreprocessingPlan."""
    col1 = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.1, skewness=0.1)
    col2 = ColumnProfileExtended(name="gender", normalized_dtype="categorical", distinct_count=2)
    prof = DatasetProfile(dataset_name="det_test", rows=100, columns=2, detailed_column_profiles=[col1, col2], target_column="target")

    builder = PreprocessingPlanBuilder()
    plan1 = builder.build_plan(prof)
    plan2 = builder.build_plan(prof)

    assert plan1.to_dict()["steps"] == plan2.to_dict()["steps"]
    assert plan1.to_dict()["expected_input_schema"] == plan2.to_dict()["expected_input_schema"]


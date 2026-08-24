import pytest

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionSource
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan, PlanValidationResult
from backend.engine.preprocessing_plan_builder import PreprocessingPlanBuilder
from backend.engine.plan_validator import PlanValidator


def test_valid_preprocessing_plan_passes_validation():
    """Verify a clean PreprocessingPlan passes PlanValidator cleanly."""
    step1 = PreprocessingStep(
        step_number=1,
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        action="IMPUTE_MEDIAN",
        columns=["age"],
        decision_id="dec_101",
        decision_source=DecisionSource.RULE,
        confidence=0.95,
    )
    step2 = PreprocessingStep(
        step_number=2,
        domain=DecisionDomain.SCALING_TRANSFORMATION,
        action="STANDARD_SCALER",
        columns=["income"],
        decision_id="dec_102",
        decision_source=DecisionSource.RULE,
        confidence=0.90,
    )

    plan = PreprocessingPlan(
        dataset_name="census",
        target_column="income_bracket",
        steps=[step1, step2],
    )

    validator = PlanValidator()
    result = validator.validate_plan(plan)

    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.validated_step_count == 2


def test_target_leakage_detection():
    """Verify target column transformation triggers target leakage validation error."""
    step_leak = PreprocessingStep(
        step_number=1,
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        action="IMPUTE_MEAN",
        columns=["target_col"],  # Target column passed as feature column to impute!
        decision_id="dec_leak",
        decision_source=DecisionSource.RULE,
        confidence=0.80,
    )

    plan = PreprocessingPlan(
        dataset_name="leakage_test",
        target_column="target_col",
        steps=[step_leak],
    )

    validator = PlanValidator()
    result = validator.validate_plan(plan)

    assert result.is_valid is False
    assert any("Target Leakage Error" in err for err in result.errors)


def test_unsupported_action_detection():
    """Verify unsupported action triggers validation error."""
    step_bad = PreprocessingStep(
        step_number=1,
        domain=DecisionDomain.ENCODING_STRATEGY,
        action="MAGIC_QUANTUM_ENCODE",
        columns=["city"],
        decision_id="dec_bad",
        decision_source=DecisionSource.RULE,
        confidence=0.50,
    )

    plan = PreprocessingPlan(dataset_name="bad_action", steps=[step_bad])

    validator = PlanValidator()
    result = validator.validate_plan(plan)

    assert result.is_valid is False
    assert any("Unsupported or invalid action" in err for err in result.errors)


def test_dropped_column_contradiction_detection():
    """Verify referencing a dropped column in a subsequent step triggers contradiction error."""
    step_drop = PreprocessingStep(
        step_number=1,
        domain=DecisionDomain.FEATURE_SELECTION,
        action="DROP_COLUMNS",
        columns=["old_feature"],
        decision_id="dec_drop",
        decision_source=DecisionSource.RULE,
        confidence=0.99,
    )
    step_scale = PreprocessingStep(
        step_number=2,
        domain=DecisionDomain.SCALING_TRANSFORMATION,
        action="STANDARD_SCALER",
        columns=["old_feature"],  # Old feature was dropped in step 1!
        decision_id="dec_scale",
        decision_source=DecisionSource.RULE,
        confidence=0.90,
    )

    plan = PreprocessingPlan(dataset_name="contradiction_test", steps=[step_drop, step_scale])

    validator = PlanValidator()
    result = validator.validate_plan(plan)

    assert result.is_valid is False
    assert any("Contradiction Error" in err for err in result.errors)


def test_adversarial_target_derived_feature_leakage():
    """Verify adversarial feature deriving from target column name is rejected as CRITICAL leakage."""
    step = PreprocessingStep(
        step_number=1,
        stage="FEATURE_ENGINEERING",
        domain=DecisionDomain.FEATURE_ENGINEERING,
        action="CREATE_INTERACTION",
        columns=["price_target_ratio"],
        decision_id="dec_adv_1",
        decision_source=DecisionSource.LLM,
        confidence=0.85,
    )
    plan = PreprocessingPlan(dataset_name="adv_test", target_column="price", steps=[step])

    validator = PlanValidator()
    res = validator.validate_plan(plan)

    assert res.is_valid is False
    assert res.severity == "CRITICAL"
    assert res.recommended_action == "REJECT"
    assert any("Target Leakage Error" in err for err in res.errors)


def test_adversarial_preprocessing_fitted_before_split():
    """Verify scaling/fitting operations placed before train/test split step trigger CRITICAL leakage error."""
    step_fit = PreprocessingStep(
        step_number=1,
        stage="SCALING",
        domain=DecisionDomain.SCALING_TRANSFORMATION,
        action="STANDARD_SCALER",
        columns=["feature1"],
        decision_id="dec_fit",
        decision_source=DecisionSource.RULE,
        confidence=0.95,
    )
    step_split = PreprocessingStep(
        step_number=2,
        stage="TRAIN_TEST_SPLIT",
        domain=DecisionDomain.PIPELINE_STRATEGY,
        action="STRATIFIED_TRAIN_TEST_SPLIT",
        columns=["feature1"],
        decision_id="dec_split",
        decision_source=DecisionSource.RULE,
        confidence=1.0,
    )
    plan = PreprocessingPlan(dataset_name="fit_before_split", target_column="label", steps=[step_fit, step_split])

    validator = PlanValidator()
    res = validator.validate_plan(plan)

    assert res.is_valid is False
    assert res.severity == "CRITICAL"
    assert any("fitted before train/test split" in err for err in res.errors)


def test_adversarial_target_encoding_on_full_dataset():
    """Verify target encoding without out-of-fold parameter on full dataset triggers CRITICAL leakage error."""
    step_te = PreprocessingStep(
        step_number=1,
        stage="ENCODING",
        domain=DecisionDomain.ENCODING_STRATEGY,
        action="TARGET_ENCODING",
        columns=["category_code"],
        params={"out_of_fold": False},
        decision_id="dec_te",
        decision_source=DecisionSource.LLM,
        confidence=0.90,
    )
    plan = PreprocessingPlan(dataset_name="full_te", target_column="label", steps=[step_te])

    validator = PlanValidator()
    res = validator.validate_plan(plan)

    assert res.is_valid is False
    assert res.severity == "CRITICAL"
    assert any("Target Encoding Leakage Error" in err for err in res.errors)


def test_adversarial_future_timestamp_leakage():
    """Verify temporal lookup of future timestamp operations is rejected."""
    step_future = PreprocessingStep(
        step_number=1,
        stage="FEATURE_ENGINEERING",
        domain=DecisionDomain.FEATURE_ENGINEERING,
        action="LOOKUP_FUTURE_TIMESTAMP_VALUE",
        columns=["event_time"],
        params={"future_window_hours": 24},
        decision_id="dec_time",
        decision_source=DecisionSource.LLM,
        confidence=0.85,
    )
    col_dt = ColumnProfileExtended(name="event_time", normalized_dtype="datetime")
    prof = DatasetProfile(dataset_name="time_test", rows=10, columns=1, detailed_column_profiles=[col_dt], target_column="label")
    plan = PreprocessingPlan(dataset_name="time_test", target_column="label", steps=[step_future])

    validator = PlanValidator()
    res = validator.validate_plan(plan, dataset_profile=prof)

    assert res.is_valid is False
    assert res.severity == "CRITICAL"
    assert any("Temporal Leakage Error" in err for err in res.errors)


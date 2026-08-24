import pytest
import pandas as pd
import numpy as np

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionSource
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.engine.preprocessing_plan_builder import PreprocessingPlanBuilder
from backend.engine.plan_validator import PlanValidator
from backend.engine.plan_executor import PlanExecutor


def test_imputation_and_scaling_execution():
    """Verify numeric imputation, scaling, and non-mutating safety."""
    df_raw = pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 45.0, 100.0],
        "income": [50000.0, 60000.0, np.nan, 80000.0, 90000.0],
        "target": [0, 1, 0, 1, 0],
    })
    df_copy_orig = df_raw.copy()

    step_impute = PreprocessingStep(
        step_number=1,
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        action="IMPUTE_MEDIAN",
        columns=["age", "income"],
        decision_id="dec_imp",
        decision_source=DecisionSource.RULE,
        confidence=0.95,
    )
    step_scale = PreprocessingStep(
        step_number=2,
        domain=DecisionDomain.SCALING_TRANSFORMATION,
        action="STANDARD_SCALER",
        columns=["age", "income"],
        decision_id="dec_scale",
        decision_source=DecisionSource.RULE,
        confidence=0.90,
    )

    plan = PreprocessingPlan(dataset_name="test_imp", target_column="target", steps=[step_impute, step_scale])

    executor = PlanExecutor()
    df_out, res = executor.execute_plan(plan, df_raw)

    assert res.status == "SUCCESS"
    assert res.executed_steps_count == 2
    assert df_out["age"].isnull().sum() == 0
    assert df_out["income"].isnull().sum() == 0
    # Verify input dataframe was NOT mutated
    pd.testing.assert_frame_equal(df_raw, df_copy_orig)


def test_one_hot_encoding_execution():
    """Verify categorical one-hot encoding execution."""
    df_raw = pd.DataFrame({
        "city": ["NYC", "LA", "NYC", "Chicago", "LA"],
        "target": [100, 200, 150, 300, 250],
    })

    step_ohe = PreprocessingStep(
        step_number=1,
        domain=DecisionDomain.ENCODING_STRATEGY,
        action="ONE_HOT_ENCODING",
        columns=["city"],
        decision_id="dec_ohe",
        decision_source=DecisionSource.RULE,
        confidence=0.95,
    )

    plan = PreprocessingPlan(dataset_name="test_ohe", target_column="target", steps=[step_ohe])

    executor = PlanExecutor()
    df_out, res = executor.execute_plan(plan, df_raw)

    assert res.status == "SUCCESS"
    assert "city" not in df_out.columns
    assert "city_NYC" in df_out.columns or "city_LA" in df_out.columns
    assert "target" in df_out.columns  # Target column preserved!


def test_full_pipeline_builder_validator_executor_integration():
    """Verify complete end-to-end integration: Builder -> Validator -> Executor."""
    df_raw = pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 45.0, 100.0, 30.0, 40.0, 50.0, 22.0, 60.0],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA", "NYC", "Chicago", "LA", "NYC", "LA"],
        "junk_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    col_age = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.10, skewness=0.1)
    col_city = ColumnProfileExtended(name="city", normalized_dtype="categorical", distinct_count=3)
    col_junk = ColumnProfileExtended(name="junk_id", normalized_dtype="numeric", distinct_count=10)

    dataset_prof = DatasetProfile(
        dataset_name="integration_ds",
        rows=10,
        columns=3,
        detailed_column_profiles=[col_age, col_city, col_junk],
        target_column="target",
    )

    # 1. Build Plan
    builder = PreprocessingPlanBuilder()
    plan = builder.build_plan(dataset_prof)

    # 2. Validate Plan
    validator = PlanValidator()
    val_res = validator.validate_plan(plan, dataset_profile=dataset_prof)
    assert val_res.is_valid is True

    # 3. Execute Plan
    executor = PlanExecutor(validator=validator)
    df_out, exec_res = executor.execute_plan(plan, df_raw)

    assert exec_res.status == "SUCCESS"
    assert exec_res.executed_steps_count > 0
    assert df_out["age"].isnull().sum() == 0
    assert "target" in df_out.columns  # Target column preserved
    assert len(df_out) == len(df_raw)

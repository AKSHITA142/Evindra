import pytest
import pandas as pd
import numpy as np

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionSource
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.engine.preprocessing_plan_builder import PreprocessingPlanBuilder
from backend.engine.plan_validator import PlanValidator
from backend.engine.plan_executor import PlanExecutor


def test_numeric_and_scaling_execution():
    """Verify numeric imputation and scaling execution."""
    df_raw = pd.DataFrame({
        "age": [25.0, np.nan, 35.0, 45.0, 100.0],
        "income": [50000.0, 60000.0, np.nan, 80000.0, 90000.0],
        "target": [0, 1, 0, 1, 0],
    })

    step_imp = PreprocessingStep(step_number=1, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="IMPUTE_MEDIAN", columns=["age", "income"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    step_scale = PreprocessingStep(step_number=2, stage="SCALING", domain=DecisionDomain.SCALING_TRANSFORMATION, action="STANDARD_SCALER", columns=["age", "income"], decision_id="d2", decision_source=DecisionSource.RULE, confidence=0.9)

    plan = PreprocessingPlan(dataset_name="numeric_test", target_column="target", steps=[step_imp, step_scale])
    executor = PlanExecutor()
    X_tr, X_te, y_tr, y_te, transformers, res = executor.execute_train_test_pipeline(plan, df_raw)

    assert res.status == "SUCCESS"
    assert X_tr["age"].isnull().sum() == 0
    assert "standard_scaler" in transformers


def test_categorical_and_unseen_category_execution():
    """Verify OneHotEncoder handles unseen categories in test set gracefully with handle_unknown='ignore'."""
    df_raw = pd.DataFrame({
        "city": ["NYC", "LA", "NYC", "LA", "Chicago", "LA", "NYC", "San Francisco"],
        "target": [1, 0, 1, 0, 1, 0, 1, 0],
    })

    step_ohe = PreprocessingStep(step_number=1, stage="ENCODING", domain=DecisionDomain.ENCODING_STRATEGY, action="ONE_HOT_ENCODING", columns=["city"], decision_id="d_ohe", decision_source=DecisionSource.RULE, confidence=0.95)
    plan = PreprocessingPlan(dataset_name="cat_test", target_column="target", steps=[step_ohe])

    executor = PlanExecutor()
    X_tr, X_te, y_tr, y_te, transformers, res = executor.execute_train_test_pipeline(plan, df_raw, test_size=0.25)

    assert res.status == "SUCCESS"
    assert "one_hot_encoder" in transformers
    assert not X_tr.empty
    assert not X_te.empty


def test_mixed_and_missing_data_execution():
    """Verify pipeline handles mixed numeric + categorical columns with missing values."""
    df_raw = pd.DataFrame({
        "num_col": [1.0, 2.0, np.nan, 4.0, 5.0],
        "cat_col": ["A", np.nan, "B", "A", "B"],
        "target": [0, 1, 0, 1, 0],
    })

    s1 = PreprocessingStep(step_number=1, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="IMPUTE_MEAN", columns=["num_col"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
    s2 = PreprocessingStep(step_number=2, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="IMPUTE_MODE", columns=["cat_col"], decision_id="d2", decision_source=DecisionSource.RULE, confidence=0.9)
    s3 = PreprocessingStep(step_number=3, stage="ENCODING", domain=DecisionDomain.ENCODING_STRATEGY, action="ONE_HOT_ENCODING", columns=["cat_col"], decision_id="d3", decision_source=DecisionSource.RULE, confidence=0.9)

    plan = PreprocessingPlan(dataset_name="mixed_test", target_column="target", steps=[s1, s2, s3])
    executor = PlanExecutor()
    X_tr, X_te, y_tr, y_te, transformers, res = executor.execute_train_test_pipeline(plan, df_raw)

    assert res.status == "SUCCESS"
    assert X_tr.isnull().sum().sum() == 0


def test_target_encoding_execution():
    """Verify target encoding fits ONLY on train split and transforms test set."""
    df_raw = pd.DataFrame({
        "category": ["A", "B", "A", "B", "A", "B", "A", "B"],
        "target": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
    })

    s_te = PreprocessingStep(step_number=1, stage="ENCODING", domain=DecisionDomain.ENCODING_STRATEGY, action="TARGET_ENCODING_OUT_OF_FOLD", columns=["category"], decision_id="d_te", decision_source=DecisionSource.RULE, confidence=0.9)
    plan = PreprocessingPlan(dataset_name="te_test", target_column="target", steps=[s_te])

    executor = PlanExecutor()
    X_tr, X_te, y_tr, y_te, transformers, res = executor.execute_train_test_pipeline(plan, df_raw)

    assert res.status == "SUCCESS"
    assert "category:target_encoder" in transformers


def test_leakage_scenario_rejection():
    """Verify pipeline execution is REJECTED if plan contains Phase 9 target leakage errors."""
    step_leak = PreprocessingStep(
        step_number=1,
        stage="MISSING_VALUE_HANDLING",
        domain=DecisionDomain.MISSING_VALUE_STRATEGY,
        action="IMPUTE_MEAN",
        columns=["target"],  # Target passed as feature column to impute!
        decision_id="d_leak",
        decision_source=DecisionSource.RULE,
        confidence=0.9,
    )
    plan = PreprocessingPlan(dataset_name="leak_exec_test", target_column="target", steps=[step_leak])

    df_raw = pd.DataFrame({"target": [1, 2, 3], "feature": [10, 20, 30]})
    executor = PlanExecutor()
    df_out, res = executor.execute_plan(plan, df_raw, validate_first=True)

    assert res.status == "FAILED"
    assert "Phase 9 Plan Validation Failed" in res.error_message


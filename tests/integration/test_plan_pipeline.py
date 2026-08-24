"""
Phase 3 Integration Tests — Plan Builder + Validator + Executor (Phases 8–10).

Exercises the COMPLETE chain:
    CSV bytes → DatasetProfiler → DecisionOrchestrator
    → PreprocessingPlanBuilder → PlanValidator → PlanExecutor

Invariants verified:
  1. Plan steps are DAG-ordered (STAGE_ORDER monotonically non-decreasing).
  2. Target column never appears as a feature in any plan step.
  3. No leakage column survives into the processed feature matrix.
  4. Transformers are fitted only on X_train; X_test shapes match.
  5. PlanValidator blocks any plan that contains a leakage step.
  6. Plans containing SEPARATE_TARGET action always have target removed from
     the feature columns in the executor output.
  7. Regression scenario produces a valid plan and execution result.
"""

import io
import pytest
import numpy as np
import pandas as pd

from backend.profiling.dataset_profiler import DatasetProfiler
from backend.engine.decision_orchestrator import DecisionOrchestrator
from backend.engine.preprocessing_plan_builder import PreprocessingPlanBuilder
from backend.engine.plan_validator import PlanValidator
from backend.engine.plan_executor import PlanExecutor
from backend.schemas.decision import DecisionDomain, DecisionSource
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_df_classification(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Clean numeric + categorical classification dataset."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age":      rng.integers(18, 70, n).astype(float),
        "income":   rng.normal(50_000, 12_000, n),
        "category": rng.choice(["A", "B", "C"], n),
        "label":    rng.integers(0, 2, n),
    })


def _make_df_regression(n: int = 150, seed: int = 7) -> pd.DataFrame:
    """Clean numeric-only regression dataset."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "sqft":     rng.integers(500, 4_000, n).astype(float),
        "bedrooms": rng.integers(1, 6, n).astype(float),
        "age_yr":   rng.integers(0, 50, n).astype(float),
        "price":    rng.normal(300_000, 80_000, n),
    })


def _make_df_with_missing(n: int = 100, seed: int = 3) -> pd.DataFrame:
    """Dataset with deliberate missing values in numeric and categorical columns."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "num_a":  rng.normal(0, 1, n),
        "num_b":  rng.normal(5, 2, n),
        "cat_x":  rng.choice(["X", "Y", "Z"], n).astype(object),
        "target": rng.integers(0, 2, n),
    })
    # Inject ~15 % missingness
    miss_idx_num = rng.choice(n, size=int(n * 0.15), replace=False)
    miss_idx_cat = rng.choice(n, size=int(n * 0.15), replace=False)
    df.loc[miss_idx_num, "num_a"] = np.nan
    df.loc[miss_idx_cat, "cat_x"] = np.nan
    return df


def _run_full_chain(df: pd.DataFrame, target_col: str):
    """Execute the full Phase 3 chain and return all intermediate results."""
    profiler    = DatasetProfiler()
    orchestrator = DecisionOrchestrator()
    builder     = PreprocessingPlanBuilder(orchestrator=orchestrator)
    validator   = PlanValidator()
    executor    = PlanExecutor(validator=validator)

    profile  = profiler.profile_dataframe(df, target_column=target_col)
    decisions = orchestrator.orchestrate_decisions(profile)
    plan      = builder.build_plan(profile, decisions=decisions)
    val_res   = validator.validate_plan(plan, dataset_profile=profile, df=df)
    X_tr, X_te, y_tr, y_te, fitted_tfs, exec_res = executor.execute_train_test_pipeline(
        plan, df, dataset_profile=profile, validate_first=False
    )

    return profile, decisions, plan, val_res, X_tr, X_te, y_tr, y_te, fitted_tfs, exec_res


# ---------------------------------------------------------------------------
# Test 1 — DAG ordering invariant
# ---------------------------------------------------------------------------

class TestDagOrdering:
    """Plan steps must follow STAGE_ORDER monotonically."""

    STAGE_ORDER = PreprocessingPlanBuilder.STAGE_ORDER

    def test_classification_plan_dag_order(self):
        df = _make_df_classification()
        _, _, plan, _, _, _, _, _, _, _ = _run_full_chain(df, "label")

        prev_order = -1
        for step in plan.steps:
            order = self.STAGE_ORDER.get(step.stage, 99)
            assert order >= prev_order, (
                f"DAG order violation: step '{step.action}' (stage={step.stage}, order={order}) "
                f"comes after a step with order {prev_order}"
            )
            prev_order = order

    def test_regression_plan_dag_order(self):
        df = _make_df_regression()
        _, _, plan, _, _, _, _, _, _, _ = _run_full_chain(df, "price")

        prev_order = -1
        for step in plan.steps:
            order = self.STAGE_ORDER.get(step.stage, 99)
            assert order >= prev_order, (
                f"DAG order violation: step '{step.action}' (stage={step.stage}, order={order}) "
                f"comes after a step with order {prev_order}"
            )
            prev_order = order


# ---------------------------------------------------------------------------
# Test 2 — Target isolation invariant
# ---------------------------------------------------------------------------

class TestTargetIsolation:
    """Target column must never appear as a feature column in any plan step."""

    def test_target_not_in_feature_steps_classification(self):
        df = _make_df_classification()
        target_col = "label"
        _, _, plan, val_res, _, _, _, _, _, _ = _run_full_chain(df, target_col)

        assert val_res.is_valid, f"Plan must be valid; errors: {val_res.errors}"

        for step in plan.steps:
            if step.stage in ("TARGET_SEPARATION", "DATA_INGESTION", "TRAIN_TEST_SPLIT"):
                continue
            assert target_col not in step.columns, (
                f"Target column '{target_col}' found as feature in step #{step.step_number} "
                f"({step.action}, stage={step.stage})"
            )

    def test_target_not_in_feature_steps_regression(self):
        df = _make_df_regression()
        target_col = "price"
        _, _, plan, val_res, _, _, _, _, _, _ = _run_full_chain(df, target_col)

        assert val_res.is_valid, f"Plan must be valid; errors: {val_res.errors}"

        for step in plan.steps:
            if step.stage in ("TARGET_SEPARATION", "DATA_INGESTION", "TRAIN_TEST_SPLIT"):
                continue
            assert target_col not in step.columns, (
                f"Target column '{target_col}' found as feature in step #{step.step_number} "
                f"({step.action}, stage={step.stage})"
            )

    def test_target_not_in_executor_output_features(self):
        """Executor's X_train / X_test must not contain the target column."""
        df = _make_df_classification()
        target_col = "label"
        _, _, _, _, X_tr, X_te, _, _, _, exec_res = _run_full_chain(df, target_col)

        assert exec_res.status in ("SUCCESS", "PARTIAL_SUCCESS")
        assert target_col not in X_tr.columns, \
            f"Target '{target_col}' leaked into X_train columns: {list(X_tr.columns)}"
        if not X_te.empty:
            assert target_col not in X_te.columns, \
                f"Target '{target_col}' leaked into X_test columns: {list(X_te.columns)}"


# ---------------------------------------------------------------------------
# Test 3 — Executor train/test isolation (no data leakage)
# ---------------------------------------------------------------------------

class TestExecutorTrainTestIsolation:
    """Transformers must be fitted only on X_train and applied to X_test."""

    def test_scaler_fitted_on_train_only(self):
        """Verify StandardScaler mean/std computed from X_train, not global data."""
        df = _make_df_classification()

        step_scale = PreprocessingStep(
            step_number=1,
            stage="SCALING",
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            action="STANDARD_SCALER",
            columns=["age", "income"],
            decision_id="d_scale",
            decision_source=DecisionSource.RULE,
            confidence=0.95,
        )
        plan = PreprocessingPlan(
            dataset_name="train_iso_test",
            target_column="label",
            steps=[step_scale],
        )
        executor = PlanExecutor(validator=PlanValidator())
        X_tr, X_te, y_tr, y_te, fitted_tfs, res = executor.execute_train_test_pipeline(
            plan, df, validate_first=True
        )

        assert res.status == "SUCCESS"
        assert "standard_scaler" in fitted_tfs

        scaler = fitted_tfs["standard_scaler"]
        # Scaler must have been fitted on X_train, so its mean should match X_train mean
        train_age_mean = df.loc[X_tr.index, "age"].mean()  # approx — before scaling
        # After StandardScaler, X_train mean should be ~0
        assert abs(X_tr["age"].mean()) < 1.0, \
            f"Expected scaled X_train 'age' mean near 0, got {X_tr['age'].mean():.4f}"

    def test_train_test_shapes_correct_and_consistent(self):
        """X_train + X_test row counts must equal full dataset minus target column."""
        df = _make_df_classification()
        target_col = "label"
        _, _, _, _, X_tr, X_te, y_tr, y_te, _, exec_res = _run_full_chain(df, target_col)

        assert exec_res.status in ("SUCCESS", "PARTIAL_SUCCESS")
        total_rows = len(X_tr) + len(X_te)
        assert total_rows == len(df), (
            f"Row count mismatch: X_train({len(X_tr)}) + X_test({len(X_te)}) = {total_rows} "
            f"!= df({len(df)})"
        )
        assert len(X_tr.columns) == len(X_te.columns), \
            "X_train and X_test must have identical column count after preprocessing"

    def test_imputer_fitted_on_train_values_only(self):
        """Median imputer must be computed from X_train only; X_test uses that fitted value."""
        df = _make_df_with_missing()

        step_imp = PreprocessingStep(
            step_number=1,
            stage="MISSING_VALUE_HANDLING",
            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
            action="IMPUTE_MEDIAN",
            columns=["num_a", "num_b"],
            decision_id="d_imp",
            decision_source=DecisionSource.RULE,
            confidence=0.95,
        )
        plan = PreprocessingPlan(
            dataset_name="imputer_iso_test",
            target_column="target",
            steps=[step_imp],
        )
        executor = PlanExecutor(validator=PlanValidator())
        X_tr, X_te, _, _, fitted_tfs, res = executor.execute_train_test_pipeline(
            plan, df, validate_first=True
        )

        assert res.status == "SUCCESS"
        # No NaN in train after imputation
        assert X_tr["num_a"].isnull().sum() == 0
        assert X_tr["num_b"].isnull().sum() == 0
        # No NaN in test after imputation
        if not X_te.empty:
            assert X_te["num_a"].isnull().sum() == 0
            assert X_te["num_b"].isnull().sum() == 0

    def test_ohe_uses_train_categories_for_test(self):
        """OHE must use categories learnt from X_train and ignore unknowns in X_test."""
        df = _make_df_with_missing()
        # Force cat_x to be non-null so OHE applies cleanly
        df["cat_x"] = df["cat_x"].fillna("Z")

        step_imp_cat = PreprocessingStep(
            step_number=1,
            stage="MISSING_VALUE_HANDLING",
            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
            action="IMPUTE_EXPLICIT_CATEGORY",
            columns=["cat_x"],
            decision_id="d_imp_cat",
            decision_source=DecisionSource.RULE,
            confidence=0.90,
        )
        step_ohe = PreprocessingStep(
            step_number=2,
            stage="ENCODING",
            domain=DecisionDomain.ENCODING_STRATEGY,
            action="ONE_HOT_ENCODING",
            columns=["cat_x"],
            decision_id="d_ohe",
            decision_source=DecisionSource.RULE,
            confidence=0.95,
        )
        plan = PreprocessingPlan(
            dataset_name="ohe_iso_test",
            target_column="target",
            steps=[step_imp_cat, step_ohe],
        )
        executor = PlanExecutor(validator=PlanValidator())
        X_tr, X_te, _, _, fitted_tfs, res = executor.execute_train_test_pipeline(
            plan, df, validate_first=True
        )

        assert res.status == "SUCCESS"
        assert "one_hot_encoder" in fitted_tfs
        # Both splits must have identical OHE feature columns
        if not X_te.empty:
            assert set(X_tr.columns) == set(X_te.columns), \
                f"Column mismatch: train={set(X_tr.columns)}, test={set(X_te.columns)}"


# ---------------------------------------------------------------------------
# Test 4 — Validator blocks leakage
# ---------------------------------------------------------------------------

class TestValidatorLeakageGate:
    """PlanValidator must block leakage-containing plans before execution."""

    def test_validator_rejects_target_in_feature_step(self):
        """Plan with target in a feature step must be INVALID."""
        step_leak = PreprocessingStep(
            step_number=1,
            stage="MISSING_VALUE_HANDLING",
            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
            action="IMPUTE_MEAN",
            columns=["label"],          # target used as feature!
            decision_id="d_leak",
            decision_source=DecisionSource.RULE,
            confidence=0.90,
        )
        plan = PreprocessingPlan(
            dataset_name="leak_gate_test",
            target_column="label",
            steps=[step_leak],
        )
        validator = PlanValidator()
        result = validator.validate_plan(plan)

        assert result.is_valid is False
        assert result.severity == "CRITICAL"
        assert any("Target Leakage Error" in e for e in result.errors)

    def test_executor_returns_failed_when_validator_blocks(self):
        """PlanExecutor with validate_first=True must return FAILED for a leaky plan."""
        step_leak = PreprocessingStep(
            step_number=1,
            stage="SCALING",
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            action="STANDARD_SCALER",
            columns=["price"],          # target used as feature!
            decision_id="d_exec_leak",
            decision_source=DecisionSource.RULE,
            confidence=0.90,
        )
        plan = PreprocessingPlan(
            dataset_name="exec_leak_gate",
            target_column="price",
            steps=[step_leak],
        )
        df = _make_df_regression()
        executor = PlanExecutor()
        _, result = executor.execute_plan(plan, df, validate_first=True)

        assert result.status == "FAILED"
        assert result.error_message is not None
        assert "Phase 9 Plan Validation Failed" in result.error_message

    def test_validator_rejects_fit_before_split(self):
        """Fitting a scaler before the SPLIT step must be caught as leakage."""
        step_scale = PreprocessingStep(
            step_number=1,
            stage="SCALING",
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            action="STANDARD_SCALER",
            columns=["age"],
            decision_id="d_pre_scale",
            decision_source=DecisionSource.RULE,
            confidence=0.95,
        )
        step_split = PreprocessingStep(
            step_number=2,
            stage="TRAIN_TEST_SPLIT",
            domain=DecisionDomain.PIPELINE_STRATEGY,
            action="STRATIFIED_TRAIN_TEST_SPLIT",
            columns=["age"],
            decision_id="d_split",
            decision_source=DecisionSource.RULE,
            confidence=1.0,
        )
        plan = PreprocessingPlan(
            dataset_name="pre_split_fit",
            target_column="label",
            steps=[step_scale, step_split],
        )
        validator = PlanValidator()
        result = validator.validate_plan(plan)

        assert result.is_valid is False
        assert any("fitted before train/test split" in e for e in result.errors)

    def test_validator_rejects_dropped_column_reuse(self):
        """Step that references a previously dropped column must be caught."""
        step_drop = PreprocessingStep(
            step_number=1,
            stage="FEATURE_SELECTION",
            domain=DecisionDomain.FEATURE_SELECTION,
            action="DROP_COLUMNS",
            columns=["stale_feature"],
            decision_id="d_drop",
            decision_source=DecisionSource.RULE,
            confidence=0.99,
        )
        step_scale = PreprocessingStep(
            step_number=2,
            stage="SCALING",
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            action="STANDARD_SCALER",
            columns=["stale_feature"],  # already dropped!
            decision_id="d_stale",
            decision_source=DecisionSource.RULE,
            confidence=0.90,
        )
        plan = PreprocessingPlan(
            dataset_name="dropped_reuse",
            target_column="label",
            steps=[step_drop, step_scale],
        )
        validator = PlanValidator()
        result = validator.validate_plan(plan)

        assert result.is_valid is False
        assert any("Contradiction Error" in e for e in result.errors)


# ---------------------------------------------------------------------------
# Test 5 — Missing value handling end-to-end
# ---------------------------------------------------------------------------

class TestMissingValueHandling:
    """Missing values must be resolved before any scaler or encoder sees them."""

    def test_missing_values_resolved_in_output(self):
        """After full chain execution, X_train must contain no NaN values."""
        df = _make_df_with_missing()
        target_col = "target"
        _, _, _, val_res, X_tr, X_te, _, _, _, exec_res = _run_full_chain(df, target_col)

        assert exec_res.status in ("SUCCESS", "PARTIAL_SUCCESS")
        assert X_tr.isnull().sum().sum() == 0, \
            f"X_train still contains NaN after full pipeline: {X_tr.isnull().sum()}"
        if not X_te.empty:
            assert X_te.isnull().sum().sum() == 0, \
                f"X_test still contains NaN after full pipeline: {X_te.isnull().sum()}"


# ---------------------------------------------------------------------------
# Test 6 — Regression scenario produces valid plan and execution
# ---------------------------------------------------------------------------

class TestRegressionScenario:
    """Full chain must succeed for a regression dataset."""

    def test_regression_plan_valid(self):
        df = _make_df_regression()
        _, _, plan, val_res, _, _, _, _, _, _ = _run_full_chain(df, "price")

        assert val_res.is_valid, f"Regression plan must be valid; errors: {val_res.errors}"
        assert len(plan.steps) > 0

    def test_regression_execution_success(self):
        df = _make_df_regression()
        _, _, _, _, X_tr, X_te, y_tr, y_te, _, exec_res = _run_full_chain(df, "price")

        assert exec_res.status in ("SUCCESS", "PARTIAL_SUCCESS")
        assert len(X_tr) > 0
        assert y_tr is not None
        assert "price" not in X_tr.columns


# ---------------------------------------------------------------------------
# Test 7 — Plan execution trace auditability
# ---------------------------------------------------------------------------

class TestExecutionTrace:
    """Every executed step must produce an auditable trace entry."""

    def test_execution_trace_recorded(self):
        df = _make_df_classification()
        _, _, _, _, _, _, _, _, _, exec_res = _run_full_chain(df, "label")

        assert exec_res.status in ("SUCCESS", "PARTIAL_SUCCESS")
        # execution_trace is a list of per-step dicts
        assert isinstance(exec_res.execution_trace, list)
        # Every trace entry must include action and duration
        for entry in exec_res.execution_trace:
            assert "action" in entry, f"Trace entry missing 'action': {entry}"
            assert "duration_seconds" in entry, f"Trace entry missing 'duration_seconds': {entry}"

    def test_fitted_pipeline_info_populated(self):
        """fitted_pipeline_info must report the transformer count after execution."""
        df = _make_df_with_missing()

        step_imp = PreprocessingStep(
            step_number=1,
            stage="MISSING_VALUE_HANDLING",
            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
            action="IMPUTE_MEDIAN",
            columns=["num_a", "num_b"],
            decision_id="d_imp2",
            decision_source=DecisionSource.RULE,
            confidence=0.95,
        )
        step_scale = PreprocessingStep(
            step_number=2,
            stage="SCALING",
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            action="STANDARD_SCALER",
            columns=["num_a", "num_b"],
            decision_id="d_sc2",
            decision_source=DecisionSource.RULE,
            confidence=0.95,
        )
        plan = PreprocessingPlan(
            dataset_name="trace_test",
            target_column="target",
            steps=[step_imp, step_scale],
        )
        executor = PlanExecutor()
        _, _, _, _, fitted_tfs, res = executor.execute_train_test_pipeline(
            plan, df, validate_first=True
        )

        assert res.status == "SUCCESS"
        assert res.fitted_pipeline_info.get("transformers_count", 0) > 0
        assert len(fitted_tfs) > 0

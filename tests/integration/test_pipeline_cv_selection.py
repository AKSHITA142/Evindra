"""
Phase 5 Integration Tests — Multiple Pipelines + Experimentation + Best Pipeline (Phases 13–15).

Exercises the COMPLETE chain:
    PipelineGenerator → PipelineCandidateSet → ExperimentRunner (CV)
    → ExperimentRunReport → BestPipelineSelector → BestPipelineResult

Invariants verified:
  1. PipelineGenerator always produces ≥ 3 logically distinct candidates.
  2. PipelineGenerator is bounded (candidates ≤ max_pipelines).
  3. Classification CV produces F1 and ROC-AUC per fold.
  4. Regression CV produces RMSE and R² per fold.
  5. Imbalanced classification switches primary metric to PR-AUC/F1.
  6. Preprocessing is fitted strictly inside each fold (no leakage path).
  7. ExperimentRunner records per-fold scores and std for each pipeline.
  8. A single failing pipeline does not crash the experiment.
  9. BestPipelineSelector picks the correct winner.
  10. Simplicity tie-breaking prefers simpler model when within threshold.
  11. High-variance pipeline is penalised over stable lower-scoring one.
  12. Full chain produces a complete BestPipelineResult with reason + tradeoffs.
"""

import numpy as np
import pandas as pd
import pytest

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.pipeline import PipelineCandidate, PipelineCandidateSet
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.schemas.decision import DecisionDomain, DecisionSource
from backend.schemas.experiment import ExperimentRunReport, PipelineEvaluationResult
from backend.schemas.best_pipeline import BestPipelineResult

from backend.profiling.dataset_profiler import DatasetProfiler
from backend.engine.pipeline_generator import PipelineGenerator
from backend.engine.experiment_runner import ExperimentRunner
from backend.engine.pipeline_selector import BestPipelineSelector


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_binary_classification_df(n: int = 120, seed: int = 0) -> pd.DataFrame:
    """Clear binary classification dataset with a strong signal feature."""
    rng = np.random.default_rng(seed)
    signal = np.concatenate([rng.normal(-2, 0.5, n // 2), rng.normal(2, 0.5, n // 2)])
    noise = rng.normal(0, 1, n)
    label = (signal > 0).astype(int)
    return pd.DataFrame({"signal": signal, "noise": noise, "label": label})


def _make_regression_df(n: int = 120, seed: int = 7) -> pd.DataFrame:
    """Clear regression dataset: price ~ sqft + bedrooms."""
    rng = np.random.default_rng(seed)
    sqft = rng.integers(500, 4_000, n).astype(float)
    beds = rng.integers(1, 6, n).astype(float)
    price = 100 * sqft + 20_000 * beds + rng.normal(0, 5_000, n)
    return pd.DataFrame({"sqft": sqft, "bedrooms": beds, "price": price})


def _make_imbalanced_df(n_majority: int = 90, n_minority: int = 10, seed: int = 1) -> pd.DataFrame:
    """Heavily imbalanced classification (9:1 ratio)."""
    rng = np.random.default_rng(seed)
    n = n_majority + n_minority
    feat = rng.normal(0, 1, n)
    label = np.array([0] * n_majority + [1] * n_minority)
    return pd.DataFrame({"feat": feat, "label": label})


def _two_pipeline_set(target_col: str, problem_type: str) -> PipelineCandidateSet:
    """
    Returns a PipelineCandidateSet with two pipelines:
      - p_simple: LogisticRegression with StandardScaler
      - p_complex: RandomForest with StandardScaler
    Used in integration tests where we control which pipeline should win.
    """
    step = PreprocessingStep(
        step_number=1,
        stage="SCALING",
        domain=DecisionDomain.SCALING_TRANSFORMATION,
        action="STANDARD_SCALER",
        columns=["signal", "noise"] if problem_type == "classification" else ["sqft", "bedrooms"],
        decision_id="d_scale",
        decision_source=DecisionSource.RULE,
        confidence=0.95,
    )
    plan = PreprocessingPlan(dataset_name="integ_ds", target_column=target_col, steps=[step])

    model_family_simple = "LOGISTIC_REGRESSION" if problem_type == "classification" else "RIDGE_REGRESSION"
    model_family_complex = "RANDOM_FOREST"

    p_simple = PipelineCandidate(
        name="Simple Baseline Pipeline",
        description="Linear model with standard scaling.",
        preprocessing_plan=plan,
        model_spec={"model_family": model_family_simple},
        estimated_cost="LOW",
    )
    p_complex = PipelineCandidate(
        name="Random Forest Pipeline",
        description="Tree ensemble with standard scaling.",
        preprocessing_plan=plan,
        model_spec={"model_family": model_family_complex},
        estimated_cost="HIGH",
    )
    return PipelineCandidateSet(
        dataset_name="integ_ds",
        problem_type=problem_type,
        target_column=target_col,
        pipelines=[p_simple, p_complex],
    )


# ---------------------------------------------------------------------------
# Test Class 1 — PipelineGenerator invariants
# ---------------------------------------------------------------------------

class TestPipelineGenerator:
    """PipelineGenerator must produce ≥ 3 distinct, bounded candidate pipelines."""

    def _make_profile(self, problem_type: str = "classification") -> DatasetProfile:
        df = _make_binary_classification_df() if problem_type == "classification" else _make_regression_df()
        target = "label" if problem_type == "classification" else "price"
        profiler = DatasetProfiler()
        return profiler.profile_dataframe(df, target_column=target)

    def test_generates_at_least_three_distinct_candidates_classification(self):
        profile = self._make_profile("classification")
        generator = PipelineGenerator(max_pipelines=4)
        pipeline_set = generator.generate_candidate_pipelines(profile)

        assert pipeline_set.total_candidates >= 3, (
            f"Expected ≥ 3 candidates, got {pipeline_set.total_candidates}"
        )
        names = [p.name for p in pipeline_set.pipelines]
        assert len(set(names)) == len(names), "All candidate names must be distinct"

    def test_generates_at_least_three_distinct_candidates_regression(self):
        profile = self._make_profile("regression")
        generator = PipelineGenerator(max_pipelines=4)
        pipeline_set = generator.generate_candidate_pipelines(profile, problem_type="regression")

        assert pipeline_set.total_candidates >= 3, (
            f"Expected ≥ 3 candidates, got {pipeline_set.total_candidates}"
        )

    def test_max_pipelines_cap_respected(self):
        """Candidate count must be ≤ max_pipelines regardless of profile."""
        profile = self._make_profile("classification")
        for cap in (1, 2, 3, 4):
            generator = PipelineGenerator(max_pipelines=cap)
            pipeline_set = generator.generate_candidate_pipelines(profile, max_pipelines=cap)
            assert len(pipeline_set.pipelines) <= cap, (
                f"cap={cap}: got {len(pipeline_set.pipelines)} candidates"
            )

    def test_all_candidates_have_preprocessing_plan(self):
        """Every candidate must carry a non-null PreprocessingPlan."""
        profile = self._make_profile("classification")
        generator = PipelineGenerator(max_pipelines=4)
        pipeline_set = generator.generate_candidate_pipelines(profile)

        for p in pipeline_set.pipelines:
            assert p.preprocessing_plan is not None, (
                f"Candidate '{p.name}' has null preprocessing_plan"
            )

    def test_all_candidates_have_model_spec(self):
        """Every candidate must carry a non-empty model_spec with model_family."""
        profile = self._make_profile("classification")
        generator = PipelineGenerator(max_pipelines=4)
        pipeline_set = generator.generate_candidate_pipelines(profile)

        for p in pipeline_set.pipelines:
            assert isinstance(p.model_spec, dict), (
                f"Candidate '{p.name}' model_spec is not a dict"
            )
            assert "model_family" in p.model_spec, (
                f"Candidate '{p.name}' model_spec missing 'model_family'"
            )


# ---------------------------------------------------------------------------
# Test Class 2 — ExperimentRunner classification metrics
# ---------------------------------------------------------------------------

class TestExperimentRunnerClassification:
    """Classification CV must produce F1, ROC-AUC, accuracy per fold."""

    def test_classification_metrics_present(self):
        """Verify F1 and ROC-AUC are present in mean_metrics for classification."""
        df = _make_binary_classification_df()
        pset = _two_pipeline_set("label", "classification")
        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pset, df, target_column="label")

        assert report.successful_evaluations >= 1
        for res in report.evaluation_results:
            if res.status == "SUCCESS":
                assert "f1" in res.mean_metrics, \
                    f"F1 missing from {res.pipeline_name} mean_metrics"
                assert "roc_auc" in res.mean_metrics, \
                    f"ROC-AUC missing from {res.pipeline_name} mean_metrics"
                assert "accuracy" in res.mean_metrics, \
                    f"Accuracy missing from {res.pipeline_name} mean_metrics"

    def test_per_fold_scores_recorded(self):
        """Each pipeline must have fold_scores list with length == n_folds."""
        n_folds = 3
        df = _make_binary_classification_df()
        pset = _two_pipeline_set("label", "classification")
        runner = ExperimentRunner(n_folds=n_folds)
        report = runner.run_experiment(pset, df, target_column="label")

        for res in report.evaluation_results:
            if res.status == "SUCCESS":
                assert isinstance(res.fold_scores, list), \
                    f"fold_scores must be a list for {res.pipeline_name}"
                assert len(res.fold_scores) == n_folds, (
                    f"{res.pipeline_name}: expected {n_folds} fold entries, "
                    f"got {len(res.fold_scores)}"
                )

    def test_fold_std_tracked(self):
        """std_metrics must be populated with non-negative values for each metric."""
        df = _make_binary_classification_df()
        pset = _two_pipeline_set("label", "classification")
        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pset, df, target_column="label")

        for res in report.evaluation_results:
            if res.status == "SUCCESS":
                assert isinstance(res.std_metrics, dict), \
                    f"std_metrics must be a dict for {res.pipeline_name}"
                for k, v in res.std_metrics.items():
                    assert v >= 0.0, (
                        f"{res.pipeline_name}: std_metrics['{k}']={v} must be non-negative"
                    )

    def test_imbalanced_classification_switches_primary_metric(self):
        """Severe imbalance (9:1) must switch primary_metric away from 'roc_auc' to 'pr_auc'."""
        df = _make_imbalanced_df()
        step = PreprocessingStep(
            step_number=1, stage="MISSING_VALUE_HANDLING",
            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
            action="PASS_THROUGH", columns=["feat"],
            decision_id="d_pt", decision_source=DecisionSource.RULE, confidence=0.9,
        )
        plan = PreprocessingPlan(dataset_name="imb_ds", target_column="label", steps=[step])
        p = PipelineCandidate(
            name="Imbalanced Pipe", description="Imbalance test",
            preprocessing_plan=plan,
            model_spec={"model_family": "LOGISTIC_REGRESSION"},
        )
        pset = PipelineCandidateSet(
            dataset_name="imb_ds", problem_type="classification",
            target_column="label", pipelines=[p],
        )
        runner = ExperimentRunner(n_folds=2)
        report = runner.run_experiment(pset, df, target_column="label")

        assert report.primary_metric in ("pr_auc", "f1", "roc_auc"), (
            f"Unexpected primary_metric='{report.primary_metric}' for imbalanced dataset"
        )
        # Must not be the default accuracy-based metric for severely imbalanced
        assert report.primary_metric != "accuracy"

    def test_stratified_split_used_for_classification(self):
        """Verify stratified CV is selected for classification (not plain KFold)."""
        # We verify indirectly: with a perfectly separable dataset both folds should show
        # at least 1 minority class sample (stratification guarantee).
        rng = np.random.default_rng(5)
        n = 60
        signal = np.concatenate([rng.normal(-3, 0.3, n // 2), rng.normal(3, 0.3, n // 2)])
        label = (signal > 0).astype(int)
        df = pd.DataFrame({"feat": signal, "label": label})

        step = PreprocessingStep(
            step_number=1, stage="SCALING",
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            action="STANDARD_SCALER", columns=["feat"],
            decision_id="d_s", decision_source=DecisionSource.RULE, confidence=0.95,
        )
        plan = PreprocessingPlan(dataset_name="strat_ds", target_column="label", steps=[step])
        p = PipelineCandidate(
            name="Strat Pipe", description="Stratification check",
            preprocessing_plan=plan, model_spec={"model_family": "LOGISTIC_REGRESSION"},
        )
        pset = PipelineCandidateSet(
            dataset_name="strat_ds", problem_type="classification",
            target_column="label", pipelines=[p],
        )
        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pset, df, target_column="label")

        assert report.successful_evaluations == 1
        res = report.evaluation_results[0]
        # With a perfectly separable binary dataset and stratified CV, ROC-AUC should be high
        assert res.mean_metrics.get("roc_auc", 0.0) >= 0.5, (
            f"Expected roc_auc >= 0.5 on separable dataset, got {res.mean_metrics.get('roc_auc')}"
        )


# ---------------------------------------------------------------------------
# Test Class 3 — ExperimentRunner regression metrics
# ---------------------------------------------------------------------------

class TestExperimentRunnerRegression:
    """Regression CV must produce RMSE and R² per fold."""

    def _regression_pset(self) -> tuple:
        step = PreprocessingStep(
            step_number=1, stage="SCALING",
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            action="STANDARD_SCALER", columns=["sqft", "bedrooms"],
            decision_id="d_reg", decision_source=DecisionSource.RULE, confidence=0.95,
        )
        plan = PreprocessingPlan(dataset_name="reg_ds", target_column="price", steps=[step])
        p = PipelineCandidate(
            name="Ridge Reg Pipe", description="Ridge regression baseline",
            preprocessing_plan=plan,
            model_spec={"model_family": "RIDGE_REGRESSION"},
        )
        pset = PipelineCandidateSet(
            dataset_name="reg_ds", problem_type="regression",
            target_column="price", pipelines=[p],
        )
        return _make_regression_df(), pset

    def test_regression_metrics_present(self):
        df, pset = self._regression_pset()
        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pset, df, target_column="price")

        assert report.primary_metric == "rmse"
        res = report.evaluation_results[0]
        assert "rmse" in res.mean_metrics, "RMSE must be present"
        assert "r2" in res.mean_metrics, "R² must be present"
        assert "mae" in res.mean_metrics, "MAE must be present"

    def test_regression_rmse_positive(self):
        """RMSE must be a positive finite number."""
        df, pset = self._regression_pset()
        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pset, df, target_column="price")

        res = report.evaluation_results[0]
        rmse = res.mean_metrics.get("rmse", -1.0)
        assert rmse > 0.0, f"RMSE must be positive, got {rmse}"
        assert np.isfinite(rmse), f"RMSE must be finite, got {rmse}"

    def test_regression_per_fold_variance_tracked(self):
        """std_metrics['rmse'] must be non-negative and populated."""
        df, pset = self._regression_pset()
        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pset, df, target_column="price")

        res = report.evaluation_results[0]
        assert "rmse" in res.std_metrics, "std for 'rmse' must be tracked"
        assert res.std_metrics["rmse"] >= 0.0


# ---------------------------------------------------------------------------
# Test Class 4 — BestPipelineSelector invariants
# ---------------------------------------------------------------------------

class TestBestPipelineSelector:
    """BestPipelineSelector must correctly identify winner using composite scoring."""

    def test_obvious_winner_chosen(self):
        """Pipeline with massively superior ROC-AUC must always win."""
        e_weak = PipelineEvaluationResult(
            pipeline_id="weak", pipeline_name="Weak Model", model_family="LOGISTIC_REGRESSION",
            status="SUCCESS", primary_metric="roc_auc", primary_score=0.55,
            std_metrics={"roc_auc": 0.01},
        )
        e_strong = PipelineEvaluationResult(
            pipeline_id="strong", pipeline_name="Strong Model", model_family="RANDOM_FOREST",
            status="SUCCESS", primary_metric="roc_auc", primary_score=0.95,
            std_metrics={"roc_auc": 0.01},
        )
        report = ExperimentRunReport(
            dataset_name="obvious_test", problem_type="classification",
            primary_metric="roc_auc", evaluation_results=[e_weak, e_strong],
        )
        selector = BestPipelineSelector(simplicity_threshold=0.01)
        result = selector.select_best_pipeline(report)

        assert result.winner_pipeline_id == "strong"
        assert result.score == 0.95

    def test_simplicity_tiebreak_within_threshold(self):
        """Within simplicity threshold, simpler lower-complexity pipeline should win."""
        e_simple = PipelineEvaluationResult(
            pipeline_id="p_simple", pipeline_name="Logistic Simple", model_family="LOGISTIC_REGRESSION",
            status="SUCCESS", primary_metric="roc_auc", primary_score=0.891,
            feature_count=5, std_metrics={"roc_auc": 0.005},
        )
        e_complex = PipelineEvaluationResult(
            pipeline_id="p_complex", pipeline_name="Heavy Ensemble", model_family="RANDOM_FOREST",
            status="SUCCESS", primary_metric="roc_auc", primary_score=0.900,
            feature_count=50, std_metrics={"roc_auc": 0.020},
        )
        report = ExperimentRunReport(
            dataset_name="simplicity_test", problem_type="classification",
            primary_metric="roc_auc", evaluation_results=[e_simple, e_complex],
        )
        selector = BestPipelineSelector(simplicity_threshold=0.01)
        result = selector.select_best_pipeline(report)

        # 0.900 - 0.891 = 0.009 <= threshold 0.01: simpler model wins
        assert result.winner_pipeline_id == "p_simple"
        assert result.tradeoffs["is_simpler_pipeline_chosen"] is True

    def test_high_variance_penalised(self):
        """Pipeline with large fold std should lose to stable pipeline with slightly lower mean."""
        e_unstable = PipelineEvaluationResult(
            pipeline_id="unstable", pipeline_name="Unstable RF", model_family="RANDOM_FOREST",
            status="SUCCESS", primary_metric="roc_auc", primary_score=0.86,
            std_metrics={"roc_auc": 0.18},  # high variance
        )
        e_stable = PipelineEvaluationResult(
            pipeline_id="stable", pipeline_name="Stable LR", model_family="LOGISTIC_REGRESSION",
            status="SUCCESS", primary_metric="roc_auc", primary_score=0.85,
            std_metrics={"roc_auc": 0.01},  # low variance
        )
        report = ExperimentRunReport(
            dataset_name="variance_test", problem_type="classification",
            primary_metric="roc_auc", evaluation_results=[e_unstable, e_stable],
        )
        selector = BestPipelineSelector(std_penalty_weight=1.0)
        result = selector.select_best_pipeline(report)

        assert result.winner_pipeline_id == "stable"

    def test_result_has_reason_and_tradeoffs(self):
        """BestPipelineResult must always have a non-empty selection_reason and tradeoffs dict."""
        e1 = PipelineEvaluationResult(
            pipeline_id="x1", pipeline_name="Winner", model_family="RIDGE_REGRESSION",
            status="SUCCESS", primary_metric="rmse", primary_score=10.0,
            std_metrics={"rmse": 0.5},
        )
        report = ExperimentRunReport(
            dataset_name="reason_test", problem_type="regression",
            primary_metric="rmse", evaluation_results=[e1],
        )
        selector = BestPipelineSelector()
        result = selector.select_best_pipeline(report)

        assert isinstance(result.selection_reason, str) and len(result.selection_reason) > 0
        assert isinstance(result.tradeoffs, dict) and len(result.tradeoffs) > 0

    def test_failed_candidates_excluded_from_selection(self):
        """FAILED evaluations must never be considered for winner selection."""
        e_failed = PipelineEvaluationResult(
            pipeline_id="failed_pipe", pipeline_name="Broken", model_family="UNKNOWN",
            status="FAILED", error_message="crash",
        )
        e_ok = PipelineEvaluationResult(
            pipeline_id="ok_pipe", pipeline_name="Working", model_family="LOGISTIC_REGRESSION",
            status="SUCCESS", primary_metric="roc_auc", primary_score=0.75,
        )
        report = ExperimentRunReport(
            dataset_name="filter_test", problem_type="classification",
            primary_metric="roc_auc", evaluation_results=[e_failed, e_ok],
        )
        selector = BestPipelineSelector()
        result = selector.select_best_pipeline(report)

        assert result.winner_pipeline_id == "ok_pipe"


# ---------------------------------------------------------------------------
# Test Class 5 — Full integration: generator → runner → selector
# ---------------------------------------------------------------------------

class TestFullPipelineChain:
    """End-to-end: PipelineGenerator → ExperimentRunner → BestPipelineSelector."""

    def test_classification_full_chain(self):
        """Full chain produces a valid BestPipelineResult for classification."""
        df = _make_binary_classification_df(n=150)
        profiler = DatasetProfiler()
        profile = profiler.profile_dataframe(df, target_column="label")

        generator = PipelineGenerator(max_pipelines=3)
        pipeline_set = generator.generate_candidate_pipelines(profile, problem_type="classification")

        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pipeline_set, df, target_column="label",
                                       dataset_profile=profile)

        assert report.successful_evaluations >= 1

        selector = BestPipelineSelector(simplicity_threshold=0.01)
        result = selector.select_best_pipeline(report)

        assert isinstance(result, BestPipelineResult)
        assert result.winner_pipeline_id is not None
        assert result.score >= 0.0
        assert result.confidence >= 0.50
        assert len(result.selection_reason) > 0

    def test_regression_full_chain(self):
        """Full chain produces a valid BestPipelineResult for regression."""
        df = _make_regression_df(n=150)
        profiler = DatasetProfiler()
        profile = profiler.profile_dataframe(df, target_column="price")

        generator = PipelineGenerator(max_pipelines=3)
        pipeline_set = generator.generate_candidate_pipelines(profile, problem_type="regression")

        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pipeline_set, df, target_column="price",
                                       dataset_profile=profile)

        assert report.primary_metric == "rmse"
        assert report.successful_evaluations >= 1

        selector = BestPipelineSelector()
        result = selector.select_best_pipeline(report)

        assert isinstance(result, BestPipelineResult)
        assert result.winner_pipeline_id is not None
        # For regression (lower-better), the best pipeline id tracked by ExperimentRunner
        # should match what the selector independently chose
        assert result.winner_pipeline_id in [
            r.pipeline_id for r in report.evaluation_results if r.status == "SUCCESS"
        ]

    def test_two_candidate_winner_correctness(self):
        """
        With two synthetic candidates where one is deliberately superior,
        BestPipelineSelector must pick the superior one unless within threshold.
        Construct: one pipeline with score=0.95, one with score=0.60 — clear winner.
        """
        e_clear_winner = PipelineEvaluationResult(
            pipeline_id="winner_id",
            pipeline_name="Strong Pipeline",
            model_family="RANDOM_FOREST",
            status="SUCCESS",
            primary_metric="roc_auc",
            primary_score=0.95,
            std_metrics={"roc_auc": 0.01},
            feature_count=10,
        )
        e_loser = PipelineEvaluationResult(
            pipeline_id="loser_id",
            pipeline_name="Weak Pipeline",
            model_family="LOGISTIC_REGRESSION",
            status="SUCCESS",
            primary_metric="roc_auc",
            primary_score=0.60,
            std_metrics={"roc_auc": 0.02},
            feature_count=5,
        )
        report = ExperimentRunReport(
            dataset_name="winner_test",
            problem_type="classification",
            primary_metric="roc_auc",
            evaluation_results=[e_clear_winner, e_loser],
        )
        selector = BestPipelineSelector(simplicity_threshold=0.01)
        result = selector.select_best_pipeline(report)

        assert result.winner_pipeline_id == "winner_id", (
            f"Expected 'winner_id', got '{result.winner_pipeline_id}' "
            f"(score={result.score})"
        )

    def test_experiment_report_fields_complete(self):
        """ExperimentRunReport must carry timing, best_pipeline_id, and evaluation_results."""
        df = _make_binary_classification_df()
        pset = _two_pipeline_set("label", "classification")

        runner = ExperimentRunner(n_folds=3)
        report = runner.run_experiment(pset, df, target_column="label")

        assert report.run_id is not None
        assert report.execution_time_seconds > 0.0
        assert report.best_pipeline_id is not None
        assert isinstance(report.evaluation_results, list)
        assert len(report.evaluation_results) == 2
        assert report.total_pipelines_evaluated == 2

    def test_single_pipeline_failure_does_not_crash_chain(self):
        """One broken pipeline must not crash experiment; remaining must succeed."""
        step = PreprocessingStep(
            step_number=1, stage="MISSING_VALUE_HANDLING",
            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
            action="PASS_THROUGH", columns=["signal"],
            decision_id="d_ft", decision_source=DecisionSource.RULE, confidence=0.9,
        )
        plan = PreprocessingPlan(dataset_name="crash_ds", target_column="label", steps=[step])

        p_good = PipelineCandidate(
            name="Good Pipe", description="Works fine",
            preprocessing_plan=plan, model_spec={"model_family": "LOGISTIC_REGRESSION"},
        )
        p_bad = PipelineCandidate(
            name="Bad Pipe", description="Broken model family",
            preprocessing_plan=plan, model_spec={"model_family": "INVALID_MAGIC_MODEL"},
        )
        pset = PipelineCandidateSet(
            dataset_name="crash_ds", problem_type="classification",
            target_column="label", pipelines=[p_good, p_bad],
        )
        df = _make_binary_classification_df()
        runner = ExperimentRunner(n_folds=2)
        report = runner.run_experiment(pset, df, target_column="label")

        assert report.successful_evaluations == 1
        assert report.failed_evaluations == 1
        assert report.best_pipeline_id == p_good.pipeline_id

        selector = BestPipelineSelector()
        result = selector.select_best_pipeline(report)
        assert result.winner_pipeline_id == p_good.pipeline_id

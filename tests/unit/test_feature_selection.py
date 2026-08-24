import pytest
import pandas as pd
import numpy as np

from backend.engine.feature_selector import FeatureSelector
from backend.schemas.feature_selection import FeatureSelectionReport, FeatureRemovalDetail


def test_constant_feature_removal():
    """Verify zero variance / constant feature is identified and removed."""
    df_raw = pd.DataFrame({
        "useful": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "constant_col": [42.0] * 10,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    selector = FeatureSelector()
    df_out, report = selector.select_features(df_raw, target_column="target")

    assert "constant_col" not in report.selected_features
    assert any(r.feature_name == "constant_col" and r.method == "ZERO_VARIANCE" for r in report.removed_features)
    assert "useful" in report.selected_features


def test_duplicate_feature_removal():
    """Verify duplicate feature with identical row values is identified and removed."""
    df_raw = pd.DataFrame({
        "feature_a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        "feature_b": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # Exact copy of feature_a
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    selector = FeatureSelector()
    df_out, report = selector.select_features(df_raw, target_column="target")

    assert len(report.selected_features) == 1
    assert any(r.method == "DUPLICATE_FEATURE" for r in report.removed_features)


def test_highly_correlated_feature_removal():
    """Verify feature with >0.95 correlation with another feature is removed while retaining the more important one."""
    np.random.seed(42)
    x1 = np.linspace(1, 10, 20)
    x2 = x1 + np.random.normal(0, 0.01, 20)  # ~0.999 correlation with x1
    y = np.where(x1 > 5, 1, 0)

    df_raw = pd.DataFrame({"x1": x1, "x2": x2, "target": y})

    selector = FeatureSelector(corr_threshold=0.95)
    df_out, report = selector.select_features(df_raw, target_column="target")

    assert any(r.method == "HIGH_CORRELATION" for r in report.removed_features)
    assert len(report.selected_features) == 1


def test_useful_feature_preservation_and_noise_removal():
    """Verify predictive feature is preserved while pure noise feature is removed."""
    np.random.seed(42)
    target = np.array([0] * 25 + [1] * 25)
    signal = target * 3.0 + np.random.normal(0, 0.2, 50)
    noise = np.random.randn(50)

    df_raw = pd.DataFrame({"signal": signal, "noise": noise, "target": target})

    selector = FeatureSelector(importance_threshold=0.25, stability_threshold=0.50)
    df_out, report = selector.select_features(df_raw, target_column="target")

    assert "signal" in report.selected_features
    assert "noise" not in report.selected_features





def test_leakage_feature_removal():
    """Verify feature derived from target name or perfectly correlated with target is removed as target leakage."""
    df_raw = pd.DataFrame({
        "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
        "target_label_ratio": [10.0, 20.0, 30.0, 40.0, 50.0],  # Name leakage
        "target_label": [0, 1, 0, 1, 0],
    })

    selector = FeatureSelector()
    df_out, report = selector.select_features(df_raw, target_column="target_label")

    assert "target_label_ratio" not in report.selected_features
    assert any(r.method == "TARGET_LEAKAGE" for r in report.removed_features)


def test_target_dependent_selection_inside_cv():
    """Verify feature scores and fold stability are computed inside cross-validation."""
    df_raw = pd.DataFrame({
        "feat_a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "feat_b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        "target": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    })

    selector = FeatureSelector(n_folds=3)
    df_out, report = selector.select_features(df_raw, target_column="target")

    assert isinstance(report, FeatureSelectionReport)
    assert len(report.feature_scores) > 0
    assert len(report.fold_stability) > 0
    for feat, stab in report.fold_stability.items():
        assert 0.0 <= stab <= 1.0


# ---------------------------------------------------------------------------
# Phase 4 — Additional invariant tests
# ---------------------------------------------------------------------------

def test_perfectly_target_correlated_column_removed():
    """
    Invariant: a synthetic column that is a perfect linear function of the target
    (corr == 1.0) must be detected as TARGET_LEAKAGE and removed.
    """
    np.random.seed(0)
    n = 40
    target = np.array([0] * 20 + [1] * 20, dtype=float)
    # Perfectly correlated with target (leakage column)
    leaked = target * 100.0

    df = pd.DataFrame({
        "useful_signal": np.linspace(0, 1, n),
        "leaked_col": leaked,
        "target": target,
    })

    selector = FeatureSelector(leakage_corr_threshold=0.999)
    _, report = selector.select_features(df, target_column="target")

    assert "leaked_col" not in report.selected_features, (
        "Perfectly target-correlated column must be removed as TARGET_LEAKAGE"
    )
    assert any(
        r.feature_name == "leaked_col" and r.method == "TARGET_LEAKAGE"
        for r in report.removed_features
    ), "Removal record must have method='TARGET_LEAKAGE' for 'leaked_col'"


def test_zero_variance_column_removed_before_mi_stage():
    """
    Invariant: a zero-variance (constant) column must be removed in Step 1 (ZERO_VARIANCE)
    and must NOT appear in feature_scores, confirming it never reached the MI/RF stage.
    """
    np.random.seed(1)
    n = 30
    df = pd.DataFrame({
        "informative": np.linspace(0, 10, n),
        "constant_col": [99.0] * n,      # zero variance
        "target": np.where(np.linspace(0, 10, n) > 5, 1, 0),
    })

    selector = FeatureSelector()
    _, report = selector.select_features(df, target_column="target")

    assert "constant_col" not in report.selected_features
    # Must be tagged ZERO_VARIANCE (Step 1), not MI/RF stage
    removal = next(
        (r for r in report.removed_features if r.feature_name == "constant_col"),
        None,
    )
    assert removal is not None, "'constant_col' must appear in removed_features"
    assert removal.method == "ZERO_VARIANCE", (
        f"Expected method='ZERO_VARIANCE', got '{removal.method}'"
    )
    # Must NOT have a feature score (was eliminated before CV stage)
    assert "constant_col" not in report.feature_scores, (
        "Zero-variance column must not reach the MI/RF importance scoring stage"
    )


def test_selected_feature_count_never_exceeds_input_count():
    """
    Invariant: selected_feature_count must always be <= number of input feature columns.
    Selector can only remove features, never create new ones.
    """
    np.random.seed(3)
    n = 50
    df = pd.DataFrame({
        f"feat_{i}": np.random.randn(n) for i in range(10)
    })
    df["target"] = np.random.randint(0, 2, n)

    selector = FeatureSelector()
    df_out, report = selector.select_features(df, target_column="target")

    n_input_features = len(df.columns) - 1  # exclude target
    assert report.selected_feature_count <= n_input_features, (
        f"selected_feature_count={report.selected_feature_count} exceeds "
        f"input feature count={n_input_features}"
    )
    assert len(df_out.columns) <= len(df.columns), (
        "Output DataFrame must not have more columns than input"
    )
    assert report.initial_feature_count == n_input_features, (
        f"initial_feature_count={report.initial_feature_count} must equal "
        f"input feature count={n_input_features}"
    )

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

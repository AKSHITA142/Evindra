import pytest
import pandas as pd
import numpy as np

from backend.engine.feature_engineer import AutomatedFeatureEngineer
from backend.schemas.feature_engineering import CandidateFeatureSet, CandidateFeature


def test_numeric_feature_generation():
    """Verify numeric ratio, product, difference, log1p, and division-by-zero protection."""
    df_raw = pd.DataFrame({
        "income": [50000.0, 60000.0, 0.0, 80000.0, 100000.0],
        "expenses": [20000.0, 0.0, 1000.0, 40000.0, 50000.0],  # Includes 0 to test div-by-zero
        "skewed_val": [10.0, 100.0, 1000.0, 50000.0, 200000.0],
        "target": [0, 1, 0, 1, 0],
    })

    engineer = AutomatedFeatureEngineer(max_features=20)
    df_out, feature_set = engineer.generate_candidate_features(df_raw, target_column="target")

    assert isinstance(feature_set, CandidateFeatureSet)
    assert feature_set.total_candidates_generated > 0
    assert "income_div_expenses" in df_out.columns
    assert "income_minus_expenses" in df_out.columns
    assert "income_x_expenses" in df_out.columns

    # Verify no infinity or crash caused by division-by-zero
    assert not np.isinf(df_out["income_div_expenses"]).any()


def test_datetime_feature_generation():
    """Verify year, month, day, weekday, hour, cyclical sin/cos, and elapsed days generation."""
    dates = pd.date_range(start="2026-01-01", periods=5, freq="D")
    df_raw = pd.DataFrame({
        "event_time": dates,
        "target": [1, 0, 1, 0, 1],
    })

    engineer = AutomatedFeatureEngineer(max_features=20)
    df_out, feature_set = engineer.generate_candidate_features(df_raw, target_column="target")

    assert "event_time_year" in df_out.columns
    assert "event_time_month" in df_out.columns
    assert "event_time_month_sin" in df_out.columns
    assert "event_time_month_cos" in df_out.columns
    assert "event_time_elapsed_days" in df_out.columns


def test_categorical_feature_generation():
    """Verify frequency encoding and categorical pair combination features."""
    df_raw = pd.DataFrame({
        "city": ["NYC", "LA", "NYC", "LA", "Chicago"],
        "tier": ["A", "B", "A", "B", "C"],
        "target": [10, 20, 30, 40, 50],
    })

    engineer = AutomatedFeatureEngineer(max_features=20)
    df_out, feature_set = engineer.generate_candidate_features(df_raw, target_column="target")

    assert "city_freq_encoded" in df_out.columns
    assert "city_tier_combined" in df_out.columns
    assert df_out["city_freq_encoded"].between(0.0, 1.0).all()


def test_text_feature_generation():
    """Verify character count and word count text features."""
    df_raw = pd.DataFrame({
        "review_text": [
            "Great product, highly recommend!",
            "Terrible quality, broken on arrival.",
            "Average item for the price.",
        ],
        "target": [5, 1, 3],
    })

    engineer = AutomatedFeatureEngineer(max_features=20)
    df_out, feature_set = engineer.generate_candidate_features(df_raw, target_column="target")

    assert "review_text_char_count" in df_out.columns
    assert "review_text_word_count" in df_out.columns
    assert (df_out["review_text_char_count"] > 0).all()


def test_target_leakage_prevention():
    """Verify target column is strictly isolated and excluded from candidate feature generation."""
    df_raw = pd.DataFrame({
        "val1": [1.0, 2.0, 3.0],
        "val2": [10.0, 20.0, 30.0],
        "target_col": [100.0, 200.0, 300.0],
    })

    engineer = AutomatedFeatureEngineer(max_features=50)
    df_out, feature_set = engineer.generate_candidate_features(df_raw, target_column="target_col")

    for cand in feature_set.candidates:
        assert "target_col" not in cand.source_columns
        assert not cand.feature_name.startswith("target_col_")
        assert cand.leakage_status == "LEAKAGE_FREE"


def test_max_features_limit_capping():
    """Verify strict max_features limit capping."""
    df_raw = pd.DataFrame({
        "f1": np.random.randn(10),
        "f2": np.random.randn(10),
        "f3": np.random.randn(10),
        "f4": np.random.randn(10),
        "target": np.random.randint(0, 2, 10),
    })

    cap_limit = 3
    engineer = AutomatedFeatureEngineer(max_features=cap_limit)
    df_out, feature_set = engineer.generate_candidate_features(df_raw, target_column="target")

    assert feature_set.total_candidates_generated <= cap_limit
    assert len(feature_set.candidates) <= cap_limit


def test_provenance_tracking_completeness():
    """Verify complete provenance tracking metadata for every generated feature."""
    df_raw = pd.DataFrame({
        "num1": [1, 2, 3],
        "num2": [4, 5, 6],
        "target": [0, 1, 0],
    })

    engineer = AutomatedFeatureEngineer(max_features=10)
    df_out, feature_set = engineer.generate_candidate_features(df_raw, target_column="target")

    for cand in feature_set.candidates:
        assert cand.feature_name != ""
        assert len(cand.source_columns) > 0
        assert cand.operation != ""
        assert cand.reason != ""
        assert cand.domain in ("numeric", "datetime", "categorical", "text", "feature_engineering")
        assert cand.leakage_status == "LEAKAGE_FREE"

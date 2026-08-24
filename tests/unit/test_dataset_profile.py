import pytest
import pandas as pd
import numpy as np

from backend.profiling import ProfilingEngine
from backend.profiling.dataset_profiler import DatasetProfiler
from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended


def test_unified_dataset_profile_generation():
    """Verify DatasetProfiler calculates all deterministic column and dataset-level fields."""
    np.random.seed(42)
    rows = 100
    df = pd.DataFrame({
        "id_col": [f"ID_{i:03d}" for i in range(rows)],
        "age": np.random.randint(18, 70, size=rows),
        "income": np.random.exponential(scale=50000, size=rows),
        "education": np.random.choice(["high_school", "bachelor", "master", "phd"], size=rows),
        "constant_col": [10.0] * rows,
        "near_constant_col": ["A"] * 96 + ["B"] * 4,
        "duplicate_col_1": range(rows),
        "duplicate_col_2": range(rows),
        "target": np.random.choice([0, 1], size=rows, p=[0.7, 0.3]),
    })

    # Introduce nulls and outliers
    df.loc[:10, "income"] = np.nan
    df.loc[95, "age"] = 150  # Outlier

    profile, hints = ProfilingEngine.profile_dataframe(df, target_column="target")

    assert isinstance(profile, DatasetProfile)
    assert profile.rows == 100
    assert profile.columns == 9
    assert profile.numeric_count > 0
    assert profile.duplicate_rows == 0
    assert profile.dataset_wide_missingness > 0.0
    assert len(profile.problem_type_candidates) > 0
    assert "binary_classification" in profile.problem_type_candidates

    # Verify column extended metrics
    col_map = {col.name: col for col in profile.detailed_column_profiles}

    # ID column
    id_prof = col_map["id_col"]
    assert id_prof.identifier_likelihood >= 0.8
    assert "identifier" in id_prof.semantic_role_hints

    # Age column (numeric stats & outliers)
    age_prof = col_map["age"]
    assert age_prof.normalized_dtype == "numeric"
    assert age_prof.numeric_statistics is not None
    assert "mean" in age_prof.numeric_statistics
    assert "iqr" in age_prof.numeric_statistics

    # Education (categorical distribution & rare categories)
    edu_prof = col_map["education"]
    assert edu_prof.normalized_dtype == "categorical"
    assert edu_prof.categorical_distribution is not None
    assert "bachelor" in edu_prof.categorical_distribution

    # Constant & Near Constant
    assert col_map["constant_col"].constant_status is True
    assert col_map["near_constant_col"].near_constant_status is True

    # Duplicate column relationship
    dup_1 = col_map["duplicate_col_1"]
    assert "duplicate_col_2" in dup_1.duplicate_column_relationship
    assert "duplicate_feature" in dup_1.semantic_role_hints

    # Verify JSON serializability and new Phase 2 canonical fields
    json_dict = profile.to_dict()
    assert isinstance(json_dict, dict)
    assert json_dict["rows"] == 100
    assert json_dict["columns"] == 9
    assert profile.row_count == 100
    assert profile.column_count == 9
    assert len(profile.numeric_columns) > 0
    assert profile.dataset_fingerprint is not None


def test_numeric_dataset():
    """Verify DatasetProfiler on purely numeric datasets."""
    df = pd.DataFrame({
        "feat_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "feat_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "target": [0.5, 1.5, 2.5, 3.5, 4.5],
    })
    prof = DatasetProfiler.profile_dataframe(df, target_column="target")
    assert prof.numeric_count == 3
    assert "feat_a" in prof.numeric_columns
    assert "regression" in prof.problem_type_candidates
    assert prof.detailed_column_profiles[0].median is not None
    assert prof.detailed_column_profiles[0].iqr is not None


def test_categorical_dataset():
    """Verify DatasetProfiler on purely categorical datasets."""
    df = pd.DataFrame({
        "color": ["red", "blue", "green", "red", "blue"],
        "size": ["S", "M", "L", "XL", "S"],
        "target": ["yes", "no", "yes", "no", "yes"],
    })
    prof = DatasetProfiler.profile_dataframe(df, target_column="target")
    assert prof.categorical_count == 3
    assert "color" in prof.categorical_columns
    assert "binary_classification" in prof.problem_type_candidates or "classification" in prof.problem_type_candidates


def test_mixed_dataset():
    """Verify DatasetProfiler on mixed datasets."""
    df = pd.DataFrame({
        "age": [25, 30, 35, 40],
        "city": ["NYC", "LA", "NYC", "LA"],
        "target": [0, 1, 0, 1],
    })
    prof = DatasetProfiler.profile_dataframe(df, target_column="target")
    assert "age" in prof.numeric_columns
    assert "city" in prof.categorical_columns
    assert prof.class_distribution is not None


def test_missing_values():
    """Verify handling of heavy missing values."""
    df = pd.DataFrame({
        "full_null": [None, None, None, None],
        "partial_null": [1.0, None, 3.0, None],
    })
    prof = DatasetProfiler.profile_dataframe(df)
    assert prof.dataset_wide_missingness == 0.75
    assert prof.global_missingness == 0.75
    col_map = {c.name: c for c in prof.detailed_column_profiles}
    assert col_map["full_null"].missing_ratio == 1.0


def test_duplicate_columns():
    """Verify detection of duplicate columns."""
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4],
        "col_b": [1, 2, 3, 4],
    })
    prof = DatasetProfiler.profile_dataframe(df)
    col_map = {c.name: c for c in prof.detailed_column_profiles}
    assert "col_b" in col_map["col_a"].duplicate_column_relationship
    assert "duplicate_feature" in col_map["col_a"].semantic_role_hints


def test_constant_columns():
    """Verify detection of constant and near-constant columns."""
    df = pd.DataFrame({
        "const": [42] * 20,
        "near_const": [1] * 19 + [2],
    })
    prof = DatasetProfiler.profile_dataframe(df)
    col_map = {c.name: c for c in prof.detailed_column_profiles}
    assert col_map["const"].constant_status is True
    assert col_map["near_const"].near_constant_status is True


def test_datetime_and_text():
    """Verify handling of datetime and text columns."""
    df = pd.DataFrame({
        "date_col": pd.date_range("2025-01-01", periods=5),
        "text_col": ["Sentence one here", "Another sentence text", "Third text entry", "Fourth long string sentence", "Fifth description text"],
    })
    prof = DatasetProfiler.profile_dataframe(df)
    assert prof.datetime_count == 1
    assert "date_col" in prof.datetime_columns
    assert prof.text_count == 1
    assert "text_col" in prof.text_columns


def test_small_dataset():
    """Verify handling of small (1-2 row) datasets."""
    df = pd.DataFrame({
        "x": [10.0],
        "y": ["A"],
    })
    prof = DatasetProfiler.profile_dataframe(df)
    assert prof.rows == 1
    assert prof.columns == 2
    assert prof.dataset_fingerprint is not None


def test_malformed_edge_case_dataset():
    """Verify handling of empty DataFrame or NaN-only edge case dataset."""
    df = pd.DataFrame(columns=["a", "b"])
    prof = DatasetProfiler.profile_dataframe(df)
    assert prof.rows == 0
    assert prof.columns == 2
    assert isinstance(prof.to_dict(), dict)


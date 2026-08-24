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

    # Verify JSON serializability
    json_dict = profile.to_dict()
    assert isinstance(json_dict, dict)
    assert json_dict["rows"] == 100
    assert json_dict["columns"] == 9

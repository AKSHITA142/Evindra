import os
import tempfile
import pytest
import pandas as pd
import numpy as np

from backend.profiling import ProfilingEngine
from backend.schemas.enums import ColumnType, SeverityLevel, TaskType
from backend.schemas.semantic_profile import SemanticProfile


@pytest.fixture
def classification_df():
    """Fixture providing a synthetic classification DataFrame with missing values, skewness, and outliers."""
    np.random.seed(42)
    rows = 200
    df = pd.DataFrame({
        "customer_id": [f"CUST_{i:04d}" for i in range(rows)],
        "age": np.random.randint(18, 70, size=rows),
        "income": np.random.exponential(scale=50000, size=rows),  # Right-skewed
        "city": np.random.choice(["NYC", "LA", "Chicago", "Houston"], size=rows),
        "is_churn": np.random.choice([0, 1], size=rows, p=[0.8, 0.2]),
        "constant_feature": ["fixed_val"] * rows,
    })

    # Add missing values
    df.loc[:30, "age"] = np.nan  # ~15% missing
    return df


@pytest.fixture
def regression_df():
    """Fixture providing a synthetic regression DataFrame."""
    np.random.seed(42)
    rows = 150
    return pd.DataFrame({
        "square_feet": np.random.randint(500, 3500, size=rows),
        "bedrooms": np.random.randint(1, 6, size=rows),
        "price": np.random.uniform(150000, 850000, size=rows),  # Continuous target
    })


def test_profiling_classification_dataset(classification_df):
    """Verify ProfilingEngine on a classification dataset."""
    profile, hints = ProfilingEngine.profile_dataframe(classification_df, target_column="is_churn")

    assert isinstance(profile, SemanticProfile)
    assert profile.dataset_summary["rows"] == 200
    assert profile.dataset_summary["columns"] == 6

    # Verify column types
    col_map = {col.name: col.type for col in profile.column_profiles}
    assert col_map["customer_id"] in [ColumnType.TEXT, ColumnType.CATEGORICAL, ColumnType.CATEGORICAL_HIGH_CARDINALITY]
    assert col_map["age"] == ColumnType.NUMERIC
    assert col_map["city"] in [ColumnType.CATEGORICAL, ColumnType.CATEGORICAL_HIGH_CARDINALITY]

    # Verify target analysis
    target_info = profile.dataset_summary["target"]
    assert target_info["target_column"] == "is_churn"
    assert target_info["task_type"] == "classification"
    assert target_info["is_imbalanced"] is True  # 80/20 split

    # Verify execution hints
    assert hints.execution_mode == "lightweight"
    assert hints.use_lazy_loading is False


def test_profiling_regression_dataset(regression_df):
    """Verify ProfilingEngine on a regression dataset."""
    profile, hints = ProfilingEngine.profile_dataframe(regression_df, target_column="price")

    target_info = profile.dataset_summary["target"]
    assert target_info["target_column"] == "price"
    assert target_info["task_type"] == "regression"


def test_profiling_quality_issues():
    """Verify QualityAnalyzer flags missing values, constant columns, and duplicates."""
    df = pd.DataFrame({
        "normal_col": range(100),
        "constant_col": [5] * 100,  # Constant feature
        "missing_col": [None] * 30 + [1.0] * 70,  # 30% missing
    })
    # Add duplicate rows
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)

    profile, _ = ProfilingEngine.profile_dataframe(df)
    issue_types = [issue.problem for issue in profile.quality_issues]

    assert "high_missing_values" in issue_types
    assert "constant_columns" in issue_types
    assert "duplicate_rows" in issue_types

    # Check confidence validation (0.0 to 1.0)
    for issue in profile.quality_issues:
        assert 0.0 <= issue.confidence <= 1.0


def test_profiling_file_loader():
    """Verify profiling a CSV file directly using profile_file."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [10, 20, 30, 40, 50]})

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_path = f.name

    try:
        profile, hints = ProfilingEngine.profile_file(temp_path, target_column="y")
        assert profile.dataset_summary["rows"] == 5
        assert profile.dataset_summary["columns"] == 2
        assert hints.execution_mode == "lightweight"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

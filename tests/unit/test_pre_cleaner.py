import pandas as pd
import numpy as np
from backend.ml_execution.pre_cleaner import DataPreCleaner, CleaningAudit
from backend.ml_execution.transformers import ImputerTransformer


def test_data_pre_cleaner_deduplication_and_missing_handling():
    # Construct test dataframe with deliberate target nulls, duplicate rows, extreme missing column, and sparse row
    df = pd.DataFrame({
        "id": ["1", "2", "3", "4", "5", "6"],
        "target": [1, 0, 1, np.nan, 0, 1],  # 1 target null at index 3
        "feature_a": [10.0, 20.0, 10.0, 40.0, np.nan, np.nan],  # Index 0 and 2 have identical features (duplicate)
        "feature_b": ["A", "B", "A", "D", "E", np.nan],
        "feature_c": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],  # 100% missing column (>75% drop rule)
    })

    cleaned_df, audit = DataPreCleaner.clean_raw_dataset(
        df=df,
        target_column="target",
        extreme_missing_threshold=75.0,
        sparse_row_threshold=0.50
    )

    # 1. Target null at index 3 dropped
    assert audit.target_nulls_dropped == 1
    assert "target" in cleaned_df.columns
    assert cleaned_df["target"].isnull().sum() == 0

    # 2. Duplicate feature row at index 2 dropped
    assert audit.duplicates_removed >= 1

    # 3. Extreme missing column feature_c dropped
    assert "feature_c" in audit.extreme_cols_dropped
    assert "feature_c" not in cleaned_df.columns

    # 4. Final cleaned DataFrame has non-zero valid rows
    assert len(cleaned_df) > 0


def test_imputer_transformer_with_missing_indicator():
    df_train = pd.DataFrame({
        "age": [25.0, np.nan, 45.0, 50.0, 60.0],
        "income": [50000.0, 60000.0, np.nan, 80000.0, 95000.0],
        "city": ["NY", "LA", np.nan, "SF", "NY"]
    })

    imputer = ImputerTransformer(strategy="median", add_missing_indicator=True)
    imputer.fit(df_train)

    assert "age" in imputer.missing_indicator_cols_
    assert "income" in imputer.missing_indicator_cols_

    df_trans = imputer.transform(df_train)

    # Verify MissingIndicator flags added
    assert "age_isnan" in df_trans.columns
    assert "income_isnan" in df_trans.columns
    assert df_trans["age_isnan"].iloc[1] == 1.0
    assert df_trans["age_isnan"].iloc[0] == 0.0

    # Verify missing values imputed
    assert df_trans["age"].isnull().sum() == 0
    assert df_trans["income"].isnull().sum() == 0
    assert df_trans["city"].isnull().sum() == 0


def test_metadata_column_extraction():
    from backend.ml_execution.executor import MLExecutionEngine

    df = pd.DataFrame({
        "User_ID": [f"U_{i}" for i in range(10)],
        "Full_Name": [f"User Name {i}" for i in range(10)],
        "Customer_Email": [f"user_{i}@example.com" for i in range(10)],
        "age": [20 + i for i in range(10)],
        "income": [30000 + i * 1000 for i in range(10)],
        "target": [0, 1] * 5
    })

    meta_df, features_df = MLExecutionEngine._extract_meta_and_features(df, target_column="target")

    assert "User_ID" in meta_df.columns
    assert "Full_Name" in meta_df.columns
    assert "Customer_Email" in meta_df.columns
    assert "age" in features_df.columns
    assert "income" in features_df.columns
    assert "Full_Name" not in features_df.columns
    assert "Customer_Email" not in features_df.columns

import pandas as pd
import numpy as np
import os
from backend.ml_execution.executor import MLExecutionEngine
from backend.ml_execution.pre_cleaner import DataPreCleaner


def test_subtoken_metadata_identification():
    # Test dataset with compound column names: patient_name, doctor_id, client_email
    df = pd.DataFrame({
        "patient_name": [f"Patient_{i}" for i in range(10)],
        "doctor_id": [f"DOC_{i}" for i in range(10)],
        "client_email": [f"client_{i}@example.com" for i in range(10)],
        "patient_age": [20 + i for i in range(10)],
        "income_amount": [30000 + i * 1000 for i in range(10)],
        "target": [0, 1] * 5
    })

    meta_df, features_df = MLExecutionEngine._extract_meta_and_features(df, target_column="target")

    # Verify compound metadata columns isolated
    assert "patient_name" in meta_df.columns
    assert "doctor_id" in meta_df.columns
    assert "client_email" in meta_df.columns

    # Verify predictive features retained in X
    assert "patient_age" in features_df.columns
    assert "income_amount" in features_df.columns
    assert "patient_name" not in features_df.columns
    assert "doctor_id" not in features_df.columns


def test_dual_export_artifact_generation(tmp_path):
    df = pd.DataFrame({
        "patient_name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Heidi", "Ivan", "Judy"],
        "age": [25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0],
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    })

    meta_df, features_df = MLExecutionEngine._extract_meta_and_features(df, target_column="target")

    os.makedirs("storage/artifacts", exist_ok=True)
    exp_id = "EXP_TEST_DUAL"

    biz_path = f"storage/artifacts/{exp_id}_business_action.csv"
    ml_path = f"storage/artifacts/{exp_id}_ml_ready.csv"

    # Business action CSV
    biz_df = pd.concat([meta_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
    biz_df["target"] = df["target"].values
    biz_df.to_csv(biz_path, index=False)

    # ML-Ready CSV
    ml_df = features_df.copy().reset_index(drop=True)
    ml_df["target"] = df["target"].values
    ml_df.to_csv(ml_path, index=False)

    assert os.path.exists(biz_path)
    assert os.path.exists(ml_path)

    biz_read = pd.read_csv(biz_path)
    ml_read = pd.read_csv(ml_path)

    assert "patient_name" in biz_read.columns
    assert "patient_name" not in ml_read.columns
    assert "age" in ml_read.columns
    assert "target" in ml_read.columns

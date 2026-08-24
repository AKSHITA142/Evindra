import os
import tempfile
import pandas as pd
import numpy as np
import pytest

from backend.profiling.loader import DataLoader
from backend.ml_execution.pre_cleaner import DataPreCleaner
from backend.services.storage.supabase_storage import SupabaseStorageService


def test_optimize_dtypes_memory_savings():
    # Create large dataframe with float64, int64, and repeating string columns
    n_rows = 1000
    df = pd.DataFrame({
        "int_col": np.random.randint(0, 100, size=n_rows, dtype=np.int64),
        "float_col": np.random.randn(n_rows).astype(np.float64),
        "cat_col": np.random.choice(["Finance", "IT", "HR", "Marketing"], size=n_rows),
    })

    initial_memory = df.memory_usage(deep=True).sum()
    df_opt = DataLoader.optimize_dtypes(df)
    optimized_memory = df_opt.memory_usage(deep=True).sum()

    # Optimized dataframe must use significantly less memory
    assert optimized_memory < initial_memory
    assert df_opt["int_col"].dtype.name in ("int8", "int16", "int32")
    assert df_opt["float_col"].dtype.name == "float32"
    assert str(df_opt["cat_col"].dtype) == "category"


def test_count_file_lines_and_lazy_sampling():
    with tempfile.NamedTemporaryFile(suffix=".csv", mode="w+", delete=False) as f:
        f.write("id,name,value\n")
        for i in range(100):
            f.write(f"{i},user_{i},{i * 1.5}\n")
        temp_path = f.name

    try:
        # Fast line count (excluding header)
        line_count = DataLoader.count_file_lines(temp_path)
        assert line_count == 100

        # Load with max_sample_rows = 20
        df_sample, meta, is_sampled = DataLoader.load_lazy_sample(temp_path, max_sample_rows=20)
        assert is_sampled is True
        assert len(df_sample) == 20
        assert meta["total_raw_rows"] == 100

        # Load without capping
        df_full, meta_full, is_sampled_full = DataLoader.load_lazy_sample(temp_path, max_sample_rows=200)
        assert is_sampled_full is False
        assert len(df_full) == 100
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def test_pre_cleaner_smart_sampling():
    n_rows = 500
    df = pd.DataFrame({
        "id": range(n_rows),
        "feature_1": np.random.randn(n_rows),
        "target": np.random.choice([0, 1], size=n_rows),
    })

    # Clean with max_sample_rows = 100
    cleaned_df, audit = DataPreCleaner.clean_raw_dataset(df, target_column="target", max_sample_rows=100)
    assert len(cleaned_df) == 100
    assert audit.final_rows == 100


def test_supabase_storage_service_initialization():
    # Test initialization with explicit empty parameters
    service = SupabaseStorageService(url="", key="")
    assert service.is_configured is False
    assert service.ensure_bucket_exists() is False

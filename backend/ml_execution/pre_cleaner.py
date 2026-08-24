from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import numpy as np
from pydantic import BaseModel, Field


class CleaningAudit(BaseModel):
    """Audit metadata tracking raw dataset pre-cleaning operations."""
    initial_rows: int = 0
    final_rows: int = 0
    initial_cols: int = 0
    final_cols: int = 0
    target_nulls_dropped: int = 0
    duplicates_removed: int = 0
    extreme_cols_dropped: List[str] = Field(default_factory=list)
    sparse_rows_dropped: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump()


class DataPreCleaner:
    """
    Leakage-safe raw data pre-cleaner.
    Executes deduplication, target null handling, extreme column dropping (>75% missing),
    and sparse row dropping (>50% missing features) PRIOR to 80/20 train/test splitting and CV.
    """

    @staticmethod
    def _is_metadata_column(col: str, n_rows: int, series: pd.Series) -> bool:
        import re
        col_str = str(col).lower().strip()
        meta_root_tokens = {
            "id", "name", "email", "ssn", "token", "hash", "uuid", "address",
            "phone", "code", "index", "rowid", "guid", "number"
        }
        tokens = [t for t in re.split(r"[_\-\s\.]+", col_str) if t]
        for token in tokens:
            if token in meta_root_tokens or token.endswith("id"):
                return True
            for root in meta_root_tokens:
                if root in token:
                    return True

        if n_rows > 5 and (series.dtype == object or str(series.dtype) in ("category", "string")):
            if (series.nunique() / float(n_rows)) >= 0.80:
                return True
        return False

    @classmethod
    def clean_raw_dataset(
        cls,
        df: pd.DataFrame,
        target_column: str,
        extreme_missing_threshold: float = 75.0,
        sparse_row_threshold: float = 0.50,
        max_sample_rows: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, CleaningAudit]:
        """
        Executes pre-cleaning on a raw dataset copy.
        Returns cleaned DataFrame and a structured CleaningAudit record.
        """
        audit = CleaningAudit(
            initial_rows=len(df),
            initial_cols=len(df.columns),
        )

        if len(df) == 0 or target_column not in df.columns:
            audit.final_rows = len(df)
            audit.final_cols = len(df.columns)
            return df.copy(), audit

        cleaned_df = df.copy()

        # 1. Drop rows with missing target column values (100% Drop Rule)
        target_series = cleaned_df[target_column]
        valid_target_mask = target_series.notna()
        audit.target_nulls_dropped = int((~valid_target_mask).sum())
        cleaned_df = cleaned_df[valid_target_mask].copy()

        if len(cleaned_df) == 0:
            audit.final_rows = 0
            audit.final_cols = len(cleaned_df.columns)
            return cleaned_df, audit

        # Identify metadata vs feature columns
        n_rows = len(cleaned_df)
        feature_cols = [
            c for c in cleaned_df.columns
            if c != target_column and not cls._is_metadata_column(c, n_rows, cleaned_df[c])
        ]

        # 2. Duplicate Row Removal (Feature-level deduplication to prevent data leakage)
        if feature_cols:
            duplicate_mask = cleaned_df.duplicated(subset=feature_cols, keep="first")
            audit.duplicates_removed = int(duplicate_mask.sum())
            cleaned_df = cleaned_df[~duplicate_mask].copy()
        else:
            duplicate_mask = cleaned_df.duplicated(keep="first")
            audit.duplicates_removed = int(duplicate_mask.sum())
            cleaned_df = cleaned_df[~duplicate_mask].copy()

        if len(cleaned_df) == 0:
            audit.final_rows = 0
            audit.final_cols = len(cleaned_df.columns)
            return cleaned_df, audit

        # 3. Extreme Missing Column Dropping (>75% missing features)
        extreme_cols = []
        for col in feature_cols:
            missing_pct = (cleaned_df[col].isnull().sum() / len(cleaned_df)) * 100.0
            if missing_pct >= extreme_missing_threshold:
                extreme_cols.append(col)

        if extreme_cols:
            cleaned_df = cleaned_df.drop(columns=extreme_cols).copy()
            audit.extreme_cols_dropped = extreme_cols
            feature_cols = [c for c in feature_cols if c not in extreme_cols]

        # 4. Sparse Row Removal (>50% missing feature values in a single row)
        if feature_cols and len(cleaned_df) > 0:
            row_missing_ratio = cleaned_df[feature_cols].isnull().mean(axis=1)
            sparse_mask = row_missing_ratio > sparse_row_threshold
            audit.sparse_rows_dropped = int(sparse_mask.sum())
            cleaned_df = cleaned_df[~sparse_mask].copy()

        # 5. Smart Sampling for Memory Safety during ML Search
        if max_sample_rows and len(cleaned_df) > max_sample_rows:
            try:
                # Try stratified sampling if classification target with reasonable class counts
                if target_column in cleaned_df.columns and cleaned_df[target_column].nunique() < 50:
                    frac = min(1.0, max_sample_rows / len(cleaned_df))
                    cleaned_df = cleaned_df.groupby(target_column, group_keys=False).sample(frac=frac, random_state=42).reset_index(drop=True)
                else:
                    cleaned_df = cleaned_df.sample(n=max_sample_rows, random_state=42).reset_index(drop=True)
            except Exception:
                cleaned_df = cleaned_df.sample(n=min(len(cleaned_df), max_sample_rows), random_state=42).reset_index(drop=True)

        from backend.profiling.loader import DataLoader
        cleaned_df = DataLoader.optimize_dtypes(cleaned_df)

        audit.final_rows = len(cleaned_df)
        audit.final_cols = len(cleaned_df.columns)

        return cleaned_df, audit

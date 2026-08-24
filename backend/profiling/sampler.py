"""Quantile Non-Null Sampler for fair, unbiased dataset sampling across percentiles."""

from typing import List, Dict, Any
import numpy as np
import pandas as pd


class QuantileSampler:
    """Samples non-null data rows at uniform percentiles (10th, 25th, 50th, 75th, 90th)

    to prevent sample bias from raw head(5) rows.
    """

    DEFAULT_QUANTILES = [0.10, 0.25, 0.50, 0.75, 0.90]

    @classmethod
    def sample_representative_rows(
        cls,
        df: pd.DataFrame,
        n_samples: int = 5,
        quantiles: List[float] = None,
    ) -> List[Dict[str, Any]]:
        """Returns n_samples non-null rows sampled at quantile intervals across the dataset.

        Args:
            df: Raw DataFrame.
            n_samples: Number of rows to return (default: 5).
            quantiles: Percentile positions (0.0 to 1.0).

        Returns:
            List of row dictionaries with clean string values.
        """
        if df.empty:
            return []

        quantiles = quantiles or cls.DEFAULT_QUANTILES
        n_rows = len(df)

        if n_rows <= n_samples:
            sampled_df = df.copy()
        else:
            # Calculate row indices corresponding to specified quantiles
            indices = [int(q * (n_rows - 1)) for q in quantiles[:n_samples]]
            # Deduplicate indices while preserving order
            seen = set()
            unique_indices = []
            for idx in indices:
                if idx not in seen:
                    seen.add(idx)
                    unique_indices.append(idx)

            sampled_df = df.iloc[unique_indices].copy()

        # Format rows into clean string representation
        records = []
        for _, row in sampled_df.iterrows():
            clean_row = {}
            for col, val in row.items():
                if pd.isna(val):
                    clean_row[str(col)] = "NaN"
                elif isinstance(val, (float, np.floating)):
                    clean_row[str(col)] = round(float(val), 4)
                else:
                    clean_row[str(col)] = str(val)
            records.append(clean_row)

        return records

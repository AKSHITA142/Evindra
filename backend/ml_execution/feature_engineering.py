from typing import List, Optional, Union
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PolynomialFeatures


class LogTransformTransformer(BaseEstimator, TransformerMixin):
    """Applies np.log1p to numeric features with high skewness."""

    def __init__(self, columns: Optional[List[str]] = None, skew_threshold: float = 1.0):
        self.columns = columns
        self.skew_threshold = skew_threshold
        self.target_cols_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_df = pd.DataFrame(X)
        if self.columns:
            self.target_cols_ = [c for c in self.columns if c in X_df.columns and pd.api.types.is_numeric_dtype(X_df[c])]
        else:
            num_cols = X_df.select_dtypes(include=[np.number]).columns
            self.target_cols_ = [c for c in num_cols if (X_df[c] >= 0).all() and abs(X_df[c].skew()) > self.skew_threshold]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col in self.target_cols_:
            if col in X_df.columns:
                X_df[f"{col}_log"] = np.log1p(np.maximum(0, X_df[col]))
        return X_df


class InteractionFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Generates pairwise multiplication/division interaction features for numeric columns."""

    def __init__(self, max_features: int = 10):
        self.max_features = max_features
        self.feature_pairs_: List[tuple] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_df = pd.DataFrame(X)
        num_cols = list(X_df.select_dtypes(include=[np.number]).columns)
        if len(num_cols) >= 2:
            pairs = [(num_cols[i], num_cols[j]) for i in range(len(num_cols)) for j in range(i + 1, len(num_cols))]
            self.feature_pairs_ = pairs[: self.max_features]
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col1, col2 in self.feature_pairs_:
            if col1 in X_df.columns and col2 in X_df.columns:
                X_df[f"{col1}_x_{col2}"] = X_df[col1] * X_df[col2]
        return X_df


class PolynomialFeaturesTransformer(BaseEstimator, TransformerMixin):
    """Wrapper around scikit-learn PolynomialFeatures."""

    def __init__(self, degree: int = 2, interaction_only: bool = True):
        self.degree = degree
        self.interaction_only = interaction_only
        self.poly_: Optional[PolynomialFeatures] = None
        self.num_cols_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_df = pd.DataFrame(X)
        self.num_cols_ = list(X_df.select_dtypes(include=[np.number]).columns)[:5]  # Cap columns to avoid explosion
        if self.num_cols_:
            self.poly_ = PolynomialFeatures(degree=self.degree, interaction_only=self.interaction_only, include_bias=False)
            self.poly_.fit(X_df[self.num_cols_])
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        if self.poly_ and self.num_cols_:
            poly_arr = self.poly_.transform(X_df[self.num_cols_])
            poly_cols = self.poly_.get_feature_names_out(self.num_cols_)
            poly_df = pd.DataFrame(poly_arr, columns=poly_cols, index=X_df.index)
            # Join non-duplicate new features
            new_cols = [c for c in poly_cols if c not in X_df.columns]
            X_df = X_df.join(poly_df[new_cols])
        return X_df


class DatetimeDecompositionTransformer(BaseEstimator, TransformerMixin):
    """Decomposes datetime features into year, month, day, dayofweek, and hour components."""

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col in X_df.columns:
            # Skip numeric columns completely to prevent float/int timestamps from turning into 1970 constant dates and dropping valid features
            if pd.api.types.is_numeric_dtype(X_df[col]):
                continue

            is_dt_dtype = pd.api.types.is_datetime64_any_dtype(X_df[col])
            col_str = str(col).lower()
            is_dt_named = "date" in col_str or "time" in col_str

            if is_dt_dtype or is_dt_named:
                try:
                    dt_series = pd.to_datetime(X_df[col], errors="coerce")
                    valid_cnt = dt_series.notnull().sum()
                    orig_non_null = X_df[col].notnull().sum()

                    # Require at least 1 valid datetime and >= 80% valid parse ratio for non-null entries
                    if valid_cnt > 0 and (orig_non_null == 0 or (valid_cnt / float(orig_non_null)) >= 0.80):
                        X_df[f"{col}_year"] = dt_series.dt.year
                        X_df[f"{col}_month"] = dt_series.dt.month
                        X_df[f"{col}_day"] = dt_series.dt.day
                        X_df[f"{col}_dayofweek"] = dt_series.dt.dayofweek
                        X_df[f"{col}_hour"] = dt_series.dt.hour
                        X_df = X_df.drop(columns=[col])
                except Exception:
                    pass
        return X_df

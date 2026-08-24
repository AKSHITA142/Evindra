from typing import List, Optional, Dict, Any, Union, Tuple
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler, MinMaxScaler, RobustScaler


class ColumnSelectorTransformer(BaseEstimator, TransformerMixin):
    """Transformer that filters columns by name or data type."""

    def __init__(self, columns: Optional[List[str]] = None, dtype_include: Optional[List[str]] = None):
        self.columns = columns
        self.dtype_include = dtype_include
        self.selected_columns_: List[str] = []

    def fit(self, X: pd.DataFrame, y=None):
        if self.columns is not None:
            self.selected_columns_ = [col for col in self.columns if col in X.columns]
        elif self.dtype_include is not None:
            self.selected_columns_ = list(X.select_dtypes(include=self.dtype_include).columns)
        else:
            self.selected_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_df = pd.DataFrame(X)
        if not self.selected_columns_:
            return X_df
        return X_df[self.selected_columns_]


class ImputerTransformer(BaseEstimator, TransformerMixin):
    """Custom scikit-learn compatible Imputer supporting mean, median, mode, constant with MissingIndicators."""

    def __init__(self, strategy: str = "median", fill_value: Optional[Any] = None, add_missing_indicator: bool = True):
        self.strategy = strategy
        self.fill_value = fill_value
        self.add_missing_indicator = add_missing_indicator
        self.imputers_: Dict[str, Any] = {}
        self.missing_indicator_cols_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_df = pd.DataFrame(X)
        self.imputers_ = {}
        self.missing_indicator_cols_ = []

        for col in X_df.columns:
            has_nulls = X_df[col].isnull().any()
            is_num = pd.api.types.is_numeric_dtype(X_df[col])

            if self.add_missing_indicator and is_num and has_nulls:
                self.missing_indicator_cols_.append(str(col))

            if self.strategy == "mean" and is_num:
                val = X_df[col].mean() if not X_df[col].dropna().empty else 0.0
            elif self.strategy == "median" and is_num:
                val = X_df[col].median() if not X_df[col].dropna().empty else 0.0
            elif self.strategy == "mode":
                mode_res = X_df[col].mode()
                val = mode_res.iloc[0] if not mode_res.empty else (self.fill_value or (0.0 if is_num else "missing"))
            elif self.strategy == "constant":
                val = self.fill_value if self.fill_value is not None else (0.0 if is_num else "missing")
            else:
                val = (X_df[col].median() if not X_df[col].dropna().empty else 0.0) if is_num else "missing"

            self.imputers_[str(col)] = val
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()

        # Add missing indicator binary flags for numeric features that had nulls in X_train
        if self.add_missing_indicator:
            for col in self.missing_indicator_cols_:
                if col in X_df.columns:
                    X_df[f"{col}_isnan"] = X_df[col].isnull().astype(float)

        # Impute fit values
        for col, val in self.imputers_.items():
            if col in X_df.columns:
                if str(X_df[col].dtype) == "category":
                    if val not in X_df[col].cat.categories:
                        X_df[col] = X_df[col].cat.add_categories([val])
                X_df[col] = X_df[col].fillna(val)
        return X_df


class CategoricalEncoderTransformer(BaseEstimator, TransformerMixin):
    """Custom encoder supporting onehot, ordinal, frequency, and target encoding per column."""

    def __init__(self, method: str = "onehot", column_encodings: Optional[Dict[str, str]] = None):
        self.method = method
        self.column_encodings = column_encodings or {}
        self.encoders_: Dict[str, Tuple[str, Any]] = {}
        self.freq_maps_: Dict[str, Dict[Any, float]] = {}

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_df = pd.DataFrame(X)
        obj_cols = set(X_df.select_dtypes(include=["object", "category"]).columns)
        enc_keys = set(self.column_encodings.keys()) if self.column_encodings else set()
        categorical_cols = [c for c in list(X_df.columns) if c in obj_cols or c in enc_keys]

        if not categorical_cols:
            return self

        X_cat = X_df[categorical_cols].astype(str)

        for col in categorical_cols:
            col_method = self.column_encodings.get(col, self.method)
            col_series = X_cat[[col]]
            if col_method == "onehot":
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                enc.fit(col_series)
                self.encoders_[col] = ("onehot", enc)
            elif col_method == "ordinal":
                enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                enc.fit(col_series)
                self.encoders_[col] = ("ordinal", enc)
            elif col_method == "frequency":
                freqs = X_cat[col].value_counts(normalize=True).to_dict()
                self.freq_maps_[col] = freqs
                self.encoders_[col] = ("frequency", None)
            else:
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                enc.fit(col_series)
                self.encoders_[col] = ("onehot", enc)

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        obj_cols = set(X_df.select_dtypes(include=["object", "category"]).columns)
        enc_keys = set(self.encoders_.keys())
        categorical_cols = [c for c in list(X_df.columns) if c in obj_cols or c in enc_keys]

        if not categorical_cols:
            return X_df

        X_cat = X_df[categorical_cols].astype(str)

        cols_to_drop = []
        new_dfs = []

        for col in categorical_cols:
            if col not in self.encoders_:
                continue

            method_type, enc = self.encoders_[col]
            if method_type == "onehot" and enc:
                encoded_arr = enc.transform(X_cat[[col]])
                encoded_cols = enc.get_feature_names_out([col])
                encoded_df = pd.DataFrame(encoded_arr, columns=encoded_cols, index=X_df.index)
                new_dfs.append(encoded_df)
                cols_to_drop.append(col)
            elif method_type == "ordinal" and enc:
                encoded_arr = enc.transform(X_cat[[col]])
                X_df[col] = encoded_arr
            elif method_type == "frequency":
                freq_map = self.freq_maps_.get(col, {})
                X_df[col] = X_cat[col].map(freq_map).fillna(0.0)

        if cols_to_drop:
            numeric_df = X_df.drop(columns=cols_to_drop)
            if new_dfs:
                return pd.concat([numeric_df] + new_dfs, axis=1)
            return numeric_df

        return X_df


class FeatureScalerTransformer(BaseEstimator, TransformerMixin):
    """Custom scaler supporting standard, minmax, and robust scaling per column."""

    def __init__(
        self,
        method: str = "standard",
        column_scalings: Optional[Dict[str, str]] = None,
        column_encodings: Optional[Dict[str, str]] = None,
    ):
        self.method = method
        self.column_scalings = column_scalings or {}
        self.column_encodings = column_encodings or {}
        self.scalers_: Dict[str, Any] = {}
        self.numeric_cols_: List[str] = []

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        X_df = pd.DataFrame(X)
        all_numeric = list(X_df.select_dtypes(include=[np.number]).columns)

        if not all_numeric:
            return self

        encoded_cols = set(self.column_encodings.keys())

        self.numeric_cols_ = []
        for col in all_numeric:
            col_str = str(col)
            # Skip categorical columns and one-hot generated dummy columns
            if col_str in encoded_cols:
                continue
            if any(col_str.startswith(f"{enc_c}_") for enc_c in encoded_cols):
                continue
            # If explicit column_scalings map exists, only scale columns explicitly listed in column_scalings!
            if self.column_scalings and col_str not in self.column_scalings:
                continue

            self.numeric_cols_.append(col)

        if not self.numeric_cols_:
            return self

        for col in self.numeric_cols_:
            col_method = self.column_scalings.get(str(col), self.method)
            if col_method == "robust":
                scaler = RobustScaler()
            elif col_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()

            scaler.fit(X_df[[col]])
            self.scalers_[col] = scaler

        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> pd.DataFrame:
        X_df = pd.DataFrame(X).copy()
        for col in self.numeric_cols_:
            if col in X_df.columns and pd.api.types.is_numeric_dtype(X_df[col]):
                scaler = self.scalers_.get(col)
                if scaler:
                    X_df[[col]] = scaler.transform(X_df[[col]])
        return X_df

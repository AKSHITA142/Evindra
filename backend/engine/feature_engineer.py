import logging
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import pandas as pd

from backend.schemas.dataset_profile import DatasetProfile
from backend.schemas.feature_engineering import CandidateFeature, CandidateFeatureSet

logger = logging.getLogger("datapilot.engine.feature_engineer")


class AutomatedFeatureEngineer:
    """
    Automated Feature Engineer for Evindra Preprocessing Pipeline (Phase 11).
    Deterministically generates candidate features across numeric, datetime, categorical,
    and text columns with strict division-by-zero protection, target leakage prevention,
    deduplication, feature explosion capping, and complete provenance tracking.
    """

    def __init__(self, max_features: int = 50, eps: float = 1e-8):
        self.max_features = max_features
        self.eps = eps

    def generate_candidate_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        dataset_profile: Optional[DatasetProfile] = None,
        max_features: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, CandidateFeatureSet]:
        """
        Generates candidate features for input DataFrame with complete provenance.

        Args:
            df: Input pandas DataFrame.
            target_column: Optional target column name to exclude from candidate generation.
            dataset_profile: Optional DatasetProfile for column type hints.
            max_features: Optional override for max_features limit.

        Returns:
            Tuple of (df_transformed, CandidateFeatureSet)
        """
        limit = max_features if max_features is not None else self.max_features
        df_out = df.copy()

        target_col = target_column
        if not target_col and dataset_profile:
            target_col = getattr(dataset_profile, "target_column", None)

        # Separate feature columns from target
        feature_cols = [c for c in df_out.columns if c != target_col]
        candidates: List[CandidateFeature] = []
        generated_feature_names: List[str] = []

        # Categorize columns based on dtype or profile hints
        numeric_cols: List[str] = []
        datetime_cols: List[str] = []
        categorical_cols: List[str] = []
        text_cols: List[str] = []

        for c in feature_cols:
            col_series = df_out[c]
            if pd.api.types.is_datetime64_any_dtype(col_series):
                datetime_cols.append(c)
            elif pd.api.types.is_numeric_dtype(col_series):
                if col_series.nunique() <= 5 and not pd.api.types.is_float_dtype(col_series):
                    categorical_cols.append(c)
                else:
                    numeric_cols.append(c)
            else:
                # Try parsing datetime strings
                if col_series.dtype == "object":
                    sample_vals = col_series.dropna().head(10).astype(str)
                    if sample_vals.str.contains(r"\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}").any():
                        try:
                            df_out[c] = pd.to_datetime(df_out[c], errors="coerce")
                            datetime_cols.append(c)
                            continue
                        except Exception:
                            pass

                    # Distinguish text vs categorical by average length
                    avg_len = sample_vals.str.len().mean() if not sample_vals.empty else 0
                    if avg_len > 30 or sample_vals.str.contains(" ").any():
                        text_cols.append(c)
                    else:
                        categorical_cols.append(c)
                else:
                    categorical_cols.append(c)

        # -------------------------------------------------------------
        # 1. DATETIME FEATURE GENERATION
        # -------------------------------------------------------------
        for c in datetime_cols:
            if len(candidates) >= limit:
                break
            s_dt = pd.to_datetime(df_out[c], errors="coerce")

            # Year, Month, Day, Weekday, Hour
            dt_parts = [
                ("year", "year", "Extract calendar year component"),
                ("month", "month", "Extract calendar month component"),
                ("day", "day", "Extract day of month component"),
                ("weekday", "dayofweek", "Extract day of week component"),
                ("hour", "hour", "Extract hour of day component"),
            ]
            for suffix, attr, reason in dt_parts:
                if len(candidates) >= limit:
                    break
                feat_name = f"{c}_{suffix}"
                if feat_name not in df_out.columns:
                    val = getattr(s_dt.dt, attr)
                    if val.notna().any():
                        df_out[feat_name] = val
                        candidates.append(
                            CandidateFeature(
                                feature_name=feat_name,
                                source_columns=[c],
                                operation=f"DATETIME_EXTRACT_{suffix.upper()}",
                                reason=reason,
                                domain="datetime",
                                leakage_status="LEAKAGE_FREE",
                            )
                        )
                        generated_feature_names.append(feat_name)

            # Cyclical Sin/Cos Features for Month and Hour
            if len(candidates) < limit and s_dt.dt.month.notna().any():
                month_sin = f"{c}_month_sin"
                month_cos = f"{c}_month_cos"
                if month_sin not in df_out.columns:
                    df_out[month_sin] = np.sin(2 * np.pi * s_dt.dt.month / 12.0)
                    df_out[month_cos] = np.cos(2 * np.pi * s_dt.dt.month / 12.0)
                    candidates.extend([
                        CandidateFeature(
                            feature_name=month_sin,
                            source_columns=[c],
                            operation="CYCLICAL_SIN_MONTH",
                            reason="Cyclical sine encoding for 12-month annual period",
                            domain="datetime",
                            leakage_status="LEAKAGE_FREE",
                        ),
                        CandidateFeature(
                            feature_name=month_cos,
                            source_columns=[c],
                            operation="CYCLICAL_COS_MONTH",
                            reason="Cyclical cosine encoding for 12-month annual period",
                            domain="datetime",
                            leakage_status="LEAKAGE_FREE",
                        ),
                    ])
                    generated_feature_names.extend([month_sin, month_cos])

            # Elapsed Time Feature
            if len(candidates) < limit and s_dt.notna().any():
                min_dt = s_dt.min()
                elapsed_name = f"{c}_elapsed_days"
                if elapsed_name not in df_out.columns:
                    df_out[elapsed_name] = (s_dt - min_dt).dt.total_seconds() / 86400.0
                    candidates.append(
                        CandidateFeature(
                            feature_name=elapsed_name,
                            source_columns=[c],
                            operation="ELAPSED_TIME_DAYS",
                            reason=f"Days elapsed since baseline timestamp {min_dt}",
                            domain="datetime",
                            leakage_status="LEAKAGE_FREE",
                        )
                    )
                    generated_feature_names.append(elapsed_name)

        # -------------------------------------------------------------
        # 2. NUMERIC FEATURE GENERATION (Ratios, Products, Differences, Log)
        # -------------------------------------------------------------
        # Log transforms for skewed non-negative numerics
        for c in numeric_cols:
            if len(candidates) >= limit:
                break
            col_series = df_out[c]
            if (col_series >= 0).all() and col_series.nunique() > 10:
                skew = col_series.skew()
                if skew > 1.5:
                    log_name = f"{c}_log1p"
                    if log_name not in df_out.columns:
                        df_out[log_name] = np.log1p(col_series)
                        candidates.append(
                            CandidateFeature(
                                feature_name=log_name,
                                source_columns=[c],
                                operation="LOG1P_TRANSFORM",
                                reason=f"Log1p variance stabilizing transform for right-skewed feature (skewness: {skew:.2f})",
                                domain="numeric",
                                leakage_status="LEAKAGE_FREE",
                            )
                        )
                        generated_feature_names.append(log_name)

        # Pairwise numeric interactions (Ratios, Products, Differences)
        num_pairs = []
        for i in range(len(numeric_cols)):
            for j in range(i + 1, len(numeric_cols)):
                num_pairs.append((numeric_cols[i], numeric_cols[j]))

        for c1, c2 in num_pairs:
            if len(candidates) >= limit:
                break

            s1 = df_out[c1]
            s2 = df_out[c2]

            # Ratio Feature (Safe division with epsilon protection)
            ratio_name = f"{c1}_div_{c2}"
            if ratio_name not in df_out.columns and len(candidates) < limit:
                safe_denom = np.where(s2 == 0, self.eps, s2)
                df_out[ratio_name] = s1 / safe_denom
                candidates.append(
                    CandidateFeature(
                        feature_name=ratio_name,
                        source_columns=[c1, c2],
                        operation="SAFE_RATIO",
                        reason=f"Safe mathematical ratio between {c1} and {c2} with division-by-zero protection",
                        domain="numeric",
                        leakage_status="LEAKAGE_FREE",
                    )
                )
                generated_feature_names.append(ratio_name)

            # Difference Feature
            diff_name = f"{c1}_minus_{c2}"
            if diff_name not in df_out.columns and len(candidates) < limit:
                df_out[diff_name] = s1 - s2
                candidates.append(
                    CandidateFeature(
                        feature_name=diff_name,
                        source_columns=[c1, c2],
                        operation="DIFFERENCE",
                        reason=f"Numerical difference feature between {c1} and {c2}",
                        domain="numeric",
                        leakage_status="LEAKAGE_FREE",
                    )
                )
                generated_feature_names.append(diff_name)

            # Product Feature
            prod_name = f"{c1}_x_{c2}"
            if prod_name not in df_out.columns and len(candidates) < limit:
                df_out[prod_name] = s1 * s2
                candidates.append(
                    CandidateFeature(
                        feature_name=prod_name,
                        source_columns=[c1, c2],
                        operation="PRODUCT_INTERACTION",
                        reason=f"Multiplicative interaction feature between {c1} and {c2}",
                        domain="numeric",
                        leakage_status="LEAKAGE_FREE",
                    )
                )
                generated_feature_names.append(prod_name)

        # -------------------------------------------------------------
        # 3. CATEGORICAL FEATURE GENERATION (Frequency Encoding, Combination)
        # -------------------------------------------------------------
        for c in categorical_cols:
            if len(candidates) >= limit:
                break

            # Frequency Encoding
            freq_name = f"{c}_freq_encoded"
            if freq_name not in df_out.columns:
                freq_map = (df_out[c].value_counts(normalize=True)).to_dict()
                df_out[freq_name] = df_out[c].map(freq_map).fillna(0.0)
                candidates.append(
                    CandidateFeature(
                        feature_name=freq_name,
                        source_columns=[c],
                        operation="FREQUENCY_ENCODING",
                        reason=f"Unsupervised frequency ratio encoding for categorical column '{c}'",
                        domain="categorical",
                        leakage_status="LEAKAGE_FREE",
                    )
                )
                generated_feature_names.append(freq_name)

        # Categorical Pair Combinations
        for i in range(len(categorical_cols)):
            for j in range(i + 1, len(categorical_cols)):
                if len(candidates) >= limit:
                    break
                c1, c2 = categorical_cols[i], categorical_cols[j]
                comb_name = f"{c1}_{c2}_combined"
                if comb_name not in df_out.columns:
                    df_out[comb_name] = df_out[c1].astype(str) + "_" + df_out[c2].astype(str)
                    candidates.append(
                        CandidateFeature(
                            feature_name=comb_name,
                            source_columns=[c1, c2],
                            operation="CATEGORICAL_COMBINATION",
                            reason=f"Cross-product categorical interaction between '{c1}' and '{c2}'",
                            domain="categorical",
                            leakage_status="LEAKAGE_FREE",
                        )
                    )
                    generated_feature_names.append(comb_name)

        # -------------------------------------------------------------
        # 4. TEXT FEATURE GENERATION (Char Count, Word Count, Text Length)
        # -------------------------------------------------------------
        for c in text_cols:
            if len(candidates) >= limit:
                break
            text_series = df_out[c].fillna("").astype(str)

            # Character Count
            char_name = f"{c}_char_count"
            if char_name not in df_out.columns and len(candidates) < limit:
                df_out[char_name] = text_series.str.len()
                candidates.append(
                    CandidateFeature(
                        feature_name=char_name,
                        source_columns=[c],
                        operation="TEXT_CHAR_COUNT",
                        reason=f"Total character length feature for text column '{c}'",
                        domain="text",
                        leakage_status="LEAKAGE_FREE",
                    )
                )
                generated_feature_names.append(char_name)

            # Word Count
            word_name = f"{c}_word_count"
            if word_name not in df_out.columns and len(candidates) < limit:
                df_out[word_name] = text_series.str.split().str.len().fillna(0)
                candidates.append(
                    CandidateFeature(
                        feature_name=word_name,
                        source_columns=[c],
                        operation="TEXT_WORD_COUNT",
                        reason=f"Total word count feature for text column '{c}'",
                        domain="text",
                        leakage_status="LEAKAGE_FREE",
                    )
                )
                generated_feature_names.append(word_name)

        # Construct CandidateFeatureSet result
        feature_set = CandidateFeatureSet(
            dataset_name=getattr(dataset_profile, "dataset_name", "dataset") if dataset_profile else "dataset",
            target_column=target_col,
            total_candidates_generated=len(candidates),
            candidates=candidates,
            generated_feature_names=generated_feature_names,
            metadata={
                "max_features_limit": limit,
                "input_columns_count": len(df.columns),
                "output_columns_count": len(df_out.columns),
            },
        )

        logger.info(f"Generated {len(candidates)} candidate features (capped at max_features={limit}). Output columns: {len(df_out.columns)}.")
        return df_out, feature_set

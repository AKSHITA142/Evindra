from typing import Optional, Dict, Any, List, Tuple
import pandas as pd
import numpy as np

from backend.schemas.dataset_profile import (
    DatasetProfile,
    ColumnProfileExtended,
    TargetCandidateInfo,
)
from backend.schemas.enums import ColumnType
from backend.profiling.schema_analyzer import SchemaAnalyzer
from backend.profiling.target_detector import SmartTargetDetector
from backend.profiling.quality_analyzer import QualityAnalyzer
from backend.profiling.resource_analyzer import ResourceAnalyzer


class DatasetProfiler:
    """
    Unified deterministic DatasetProfiler for Evindra Phase 2.
    Produces a complete, JSON-serializable DatasetProfile from any pandas DataFrame.
    """

    @classmethod
    def profile_dataframe(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        user_mission: str = "",
        user_task_type: str = "general",
        file_meta: Optional[Dict[str, Any]] = None,
    ) -> DatasetProfile:
        """
        Calculates all deterministic column and dataset-level profile metrics.
        """
        rows, columns = len(df), len(df.columns)
        memory_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))
        dataset_wide_missingness = (
            float(df.isnull().sum().sum() / (rows * columns)) if (rows * columns) > 0 else 0.0
        )
        duplicate_rows = int(df.duplicated().sum())

        # 1. Identify duplicate column relationships
        duplicate_pairs: Dict[str, List[str]] = {}
        cols = list(df.columns)
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                c1, c2 = cols[i], cols[j]
                if df[c1].equals(df[c2]):
                    duplicate_pairs.setdefault(c1, []).append(c2)
                    duplicate_pairs.setdefault(c2, []).append(c1)

        # 2. Target detection & candidate list
        detected_target, target_confidence = SmartTargetDetector.detect_target_with_confidence(
            df, user_mission=user_mission, user_target=target_column, user_task_type=user_task_type
        )
        final_target = target_column or detected_target

        target_candidates: List[Dict[str, Any]] = []
        for col in cols:
            if col == final_target:
                target_candidates.append({
                    "column": col,
                    "score": round(target_confidence, 4),
                    "evidence": ["Explicitly specified or highest scoring target candidate"],
                    "task_type_suitability": user_task_type,
                })

        # 3. Analyze columns
        schema_result = SchemaAnalyzer.analyze_schema(df)
        column_types = schema_result["column_types"]

        detailed_column_profiles: List[ColumnProfileExtended] = []
        numeric_count = 0
        categorical_count = 0
        datetime_count = 0
        text_count = 0

        numeric_columns_list: List[str] = []
        categorical_columns_list: List[str] = []
        datetime_columns_list: List[str] = []
        text_columns_list: List[str] = []

        ORDINAL_KEYWORDS = {
            "low", "medium", "high", "small", "large", "poor", "good",
            "excellent", "grade", "stage", "level", "1st", "2nd", "3rd", "p1", "p2", "p3"
        }

        for col in cols:
            series = df[col]
            missing_count = int(series.isnull().sum())
            missing_ratio = float(missing_count / rows) if rows > 0 else 0.0
            clean_s = series.dropna()
            unique_count = int(clean_s.nunique())
            unique_ratio = float(unique_count / rows) if rows > 0 else 0.0
            cardinality = unique_count
            raw_dtype = str(series.dtype)

            col_enum_type = column_types.get(col, ColumnType.UNKNOWN)

            # Determine normalized dtype
            if col_enum_type == ColumnType.NUMERIC or pd.api.types.is_numeric_dtype(series):
                normalized_dtype = "numeric"
                numeric_count += 1
                numeric_columns_list.append(col)
            elif col_enum_type == ColumnType.DATETIME or pd.api.types.is_datetime64_any_dtype(series):
                normalized_dtype = "datetime"
                datetime_count += 1
                datetime_columns_list.append(col)
            elif col_enum_type == ColumnType.BOOLEAN or unique_count <= 2:
                normalized_dtype = "binary"
                categorical_count += 1
                categorical_columns_list.append(col)
            elif col_enum_type == ColumnType.TEXT:
                normalized_dtype = "text"
                text_count += 1
                text_columns_list.append(col)
            else:
                normalized_dtype = "categorical"
                categorical_count += 1
                categorical_columns_list.append(col)

            # Compute numeric statistics & outliers
            numeric_stats = None
            outlier_ratio = 0.0
            val_median = None
            val_quantiles = None
            val_iqr = None
            val_kurtosis = None

            if normalized_dtype == "numeric" and len(clean_s) > 0:
                try:
                    s_float = clean_s.astype(float)
                    q25 = float(s_float.quantile(0.25))
                    q50 = float(s_float.quantile(0.50))
                    q75 = float(s_float.quantile(0.75))
                    iqr = float(q75 - q25)
                    skewness = float(s_float.skew()) if len(s_float) > 2 else 0.0
                    kurtosis = float(s_float.kurtosis()) if len(s_float) > 3 else 0.0

                    val_median = round(q50, 4)
                    val_quantiles = {"q25": round(q25, 4), "q50": round(q50, 4), "q75": round(q75, 4)}
                    val_iqr = round(iqr, 4)
                    val_kurtosis = round(kurtosis, 4)

                    numeric_stats = {
                        "mean": round(float(s_float.mean()), 4),
                        "median": val_median,
                        "std": round(float(s_float.std()), 4) if len(s_float) > 1 else 0.0,
                        "min": round(float(s_float.min()), 4),
                        "max": round(float(s_float.max()), 4),
                        "quantiles": val_quantiles,
                        "iqr": val_iqr,
                        "skewness": round(skewness, 4),
                        "kurtosis": val_kurtosis,
                    }

                    if iqr > 0:
                        outliers = (s_float < (q25 - 1.5 * iqr)) | (s_float > (q75 + 1.5 * iqr))
                        outlier_ratio = float(outliers.sum() / len(s_float))
                except Exception:
                    pass

            # Compute categorical distribution & rare category ratio
            cat_dist = None
            rare_cat_ratio = 0.0
            if normalized_dtype in ("categorical", "binary", "text") and len(clean_s) > 0:
                try:
                    freqs = clean_s.astype(str).value_counts(normalize=True)
                    cat_dist = {str(k): round(float(v), 4) for k, v in freqs.head(10).items()}
                    rare_cats = freqs[freqs < 0.01]
                    rare_cat_ratio = float(rare_cats.sum())
                except Exception:
                    pass

            # Constant and near-constant status
            constant_status = unique_count <= 1
            most_frequent_ratio = 0.0
            if len(clean_s) > 0:
                top_freq = clean_s.value_counts().max()
                most_frequent_ratio = float(top_freq / rows) if rows > 0 else 0.0
            near_constant_status = most_frequent_ratio >= 0.95

            # Likelihoods
            id_likelihood = 0.0
            col_clean = str(col).lower().strip()
            if any(token in col_clean for token in ["id", "uuid", "guid", "ssn", "hash"]) and unique_ratio > 0.8:
                id_likelihood = 1.0
            elif unique_ratio == 1.0 and normalized_dtype in ("text", "categorical"):
                id_likelihood = 0.8

            datetime_likelihood = 1.0 if normalized_dtype == "datetime" else (
                0.8 if any(token in col_clean for token in ["time", "date", "year", "month", "day"]) else 0.0
            )

            text_likelihood = 1.0 if normalized_dtype == "text" else (
                0.5 if normalized_dtype == "categorical" and unique_ratio > 0.5 else 0.0
            )

            ordinal_likelihood = 0.0
            if normalized_dtype in ("categorical", "binary") and len(clean_s) > 0:
                sample_vals = [str(v).lower() for v in clean_s.unique()[:10]]
                if any(kw in val for kw in ORDINAL_KEYWORDS for val in sample_vals):
                    ordinal_likelihood = 0.9
            elif normalized_dtype == "numeric" and 3 <= unique_count <= 10:
                try:
                    if clean_s.apply(lambda x: float(x).is_integer()).all():
                        ordinal_likelihood = 0.7
                except Exception:
                    pass

            # Semantic role hints
            semantic_role_hints: List[str] = []
            if id_likelihood >= 0.8:
                semantic_role_hints.append("identifier")
            if constant_status:
                semantic_role_hints.append("constant_feature")
            if near_constant_status:
                semantic_role_hints.append("near_constant_feature")
            if duplicate_pairs.get(col):
                semantic_role_hints.append("duplicate_feature")
            if col == final_target:
                semantic_role_hints.append("target_candidate")
            if normalized_dtype == "numeric":
                semantic_role_hints.append("numeric_feature")
            elif normalized_dtype in ("categorical", "binary"):
                semantic_role_hints.append("categorical_feature")
            elif normalized_dtype == "datetime":
                semantic_role_hints.append("datetime_feature")
            elif normalized_dtype == "text":
                semantic_role_hints.append("text_feature")

            if ordinal_likelihood >= 0.7:
                semantic_role_hints.append("ordinal_candidate")

            sample_vals_list = clean_s.iloc[:5].tolist() if len(clean_s) > 0 else []

            col_prof = ColumnProfileExtended(
                name=col,
                type=col_enum_type,
                missing_count=missing_count,
                missing_pct=round(missing_ratio * 100.0, 2),
                distinct_count=unique_count,
                skewness=numeric_stats.get("skewness") if numeric_stats else None,
                mean=numeric_stats.get("mean") if numeric_stats else None,
                std=numeric_stats.get("std") if numeric_stats else None,
                min=numeric_stats.get("min") if numeric_stats else None,
                max=numeric_stats.get("max") if numeric_stats else None,
                sample_values=sample_vals_list,
                dtype=raw_dtype,
                normalized_dtype=normalized_dtype,
                row_count=rows,
                missing_ratio=round(missing_ratio, 4),
                unique_count=unique_count,
                unique_ratio=round(unique_ratio, 4),
                cardinality=cardinality,
                numeric_statistics=numeric_stats,
                median=val_median,
                quantiles=val_quantiles,
                iqr=val_iqr,
                kurtosis=val_kurtosis,
                outlier_ratio=round(outlier_ratio, 4),
                categorical_distribution=cat_dist,
                rare_category_ratio=round(rare_cat_ratio, 4),
                constant_status=constant_status,
                near_constant_status=near_constant_status,
                duplicate_column_relationship=duplicate_pairs.get(col, []),
                identifier_likelihood=round(id_likelihood, 4),
                datetime_likelihood=round(datetime_likelihood, 4),
                text_likelihood=round(text_likelihood, 4),
                ordinal_likelihood=round(ordinal_likelihood, 4),
                semantic_role_hints=semantic_role_hints,
            )

            detailed_column_profiles.append(col_prof)

        # 4. Class distribution & imbalance ratio for target column
        class_dist = None
        imbalance_ratio = None
        feature_target_rel = {}
        problem_candidates: List[str] = []

        if final_target and final_target in df.columns:
            target_series = df[final_target].dropna()
            t_unique = int(target_series.nunique())
            if t_unique > 0:
                if t_unique == 2:
                    problem_candidates = ["binary_classification", "classification"]
                    vc = target_series.value_counts(normalize=True).to_dict()
                    class_dist = {str(k): round(float(v), 4) for k, v in vc.items()}
                    counts = target_series.value_counts()
                    imbalance_ratio = round(float(counts.max() / max(1, counts.min())), 4)
                elif pd.api.types.is_float_dtype(target_series) and t_unique == len(target_series):
                    problem_candidates = ["regression"]
                elif t_unique <= 20 or not pd.api.types.is_numeric_dtype(target_series):
                    problem_candidates = ["multiclass_classification", "classification"]
                    vc = target_series.value_counts(normalize=True).head(10).to_dict()
                    class_dist = {str(k): round(float(v), 4) for k, v in vc.items()}
                    counts = target_series.value_counts()
                    imbalance_ratio = round(float(counts.max() / max(1, counts.min())), 4)
                else:
                    problem_candidates = ["regression"]

                # Numeric correlations with target (safely check std > 0 to avoid zero-variance division warning)
                if pd.api.types.is_numeric_dtype(target_series) and target_series.std() > 0:
                    for c in cols:
                        if c != final_target and pd.api.types.is_numeric_dtype(df[c]) and df[c].std() > 0:
                            try:
                                corr = float(df[c].corr(df[final_target]))
                                if not np.isnan(corr):
                                    feature_target_rel[c] = round(corr, 4)
                            except Exception:
                                pass

        # 5. Quality issues & Resource profile
        quality_issues = QualityAnalyzer.analyze_quality(df, detailed_column_profiles)
        resource_prof, exec_hints = ResourceAnalyzer.analyze_resources(df, int(memory_mb * 1024 * 1024))

        # Map task type for summary compatibility
        primary_task = "general"
        if problem_candidates:
            if "classification" in problem_candidates[0] or "binary" in problem_candidates[0] or "multiclass" in problem_candidates[0]:
                primary_task = "classification"
            else:
                primary_task = "regression"

        dataset_summary = {
            "rows": rows,
            "columns": columns,
            "memory_mb": round(memory_mb, 4),
            "filename": (file_meta or {}).get("filename", "dataset.csv"),
            "file_size_bytes": (file_meta or {}).get("file_size_bytes", int(memory_mb * 1024 * 1024)),
            "target": {
                "target_column": final_target,
                "confidence": target_confidence,
                "task_type": primary_task,
                "is_imbalanced": (imbalance_ratio > 4.0) if imbalance_ratio is not None else False,
            },
            "id_candidates": schema_result["id_candidates"],
            "timestamp_candidates": schema_result["timestamp_candidates"],
        }

        import hashlib
        fingerprint_raw = f"{rows}_{columns}_{cols}_{[str(df[c].dtype) for c in cols]}"
        dataset_fingerprint = hashlib.sha256(fingerprint_raw.encode("utf-8")).hexdigest()[:16]

        # Construct unified DatasetProfile
        return DatasetProfile(
            dataset_name=(file_meta or {}).get("filename", "dataset"),
            rows=rows,
            columns=columns,
            row_count=rows,
            column_count=columns,
            memory_estimate_mb=round(memory_mb, 4),
            memory_usage=round(memory_mb, 4),
            dataset_wide_missingness=round(dataset_wide_missingness, 4),
            global_missingness=round(dataset_wide_missingness, 4),
            duplicate_rows=duplicate_rows,
            duplicate_row_count=duplicate_rows,
            numeric_count=numeric_count,
            categorical_count=categorical_count,
            datetime_count=datetime_count,
            text_count=text_count,
            numeric_columns=numeric_columns_list,
            categorical_columns=categorical_columns_list,
            datetime_columns=datetime_columns_list,
            text_columns=text_columns_list,
            target_column=final_target,
            target_candidate_list=target_candidates,
            target_candidates=target_candidates,
            class_distribution=class_dist,
            imbalance_ratio=imbalance_ratio,
            feature_target_relationships=feature_target_rel,
            problem_type_candidates=problem_candidates,
            dataset_fingerprint=dataset_fingerprint,
            detailed_column_profiles=detailed_column_profiles,
            column_profiles=detailed_column_profiles,
            quality_issues=quality_issues,
            resource_profile=resource_prof,
            dataset_summary=dataset_summary,
            execution_hints=exec_hints.to_dict() if hasattr(exec_hints, "to_dict") else {},
        )

from typing import Dict, Any, Optional
import pandas as pd
import numpy as np

from backend.schemas.enums import TaskType, ColumnType
from backend.profiling.target_detector import SmartTargetDetector


class TargetAnalyzer:
    """Analyzes target column to infer task type and class balance."""

    @classmethod
    def analyze_target(
        cls,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        column_types: Optional[Dict[str, ColumnType]] = None,
        user_mission: str = "",
        user_task_type: str = "general",
    ) -> Dict[str, Any]:
        """Infers target column, task type, and class distribution.

        Args:
            user_task_type: One of "classification", "regression", or "general" (auto-detect).
                            If not "general", the user's explicit choice overrides auto-detection.
        """
        # --- Smart target column detection ---
        target_column = SmartTargetDetector.detect_target(
            df, user_mission=user_mission, user_target=target_column, user_task_type=user_task_type
        )

        if not target_column or target_column not in df.columns:
            return {"target_column": None, "task_type": TaskType.CLASSIFICATION.value}

        series = df[target_column].dropna()
        unique_count = series.nunique()
        total = len(series)

        # --- Task type inference ---
        # If user explicitly chose classification or regression, honour it
        if user_task_type in ("classification", "regression"):
            task_type = TaskType.CLASSIFICATION if user_task_type == "classification" else TaskType.REGRESSION
        else:
            # Auto-detect using multi-signal heuristic
            task_type = cls._infer_task_type(series, unique_count, total, column_types, target_column)

        # Build class distribution (only for classification)
        if task_type == TaskType.REGRESSION:
            is_imbalanced = False
            class_distribution = {}
        else:
            counts = series.value_counts(normalize=True).round(4).to_dict()
            class_distribution = {str(k): float(v) for k, v in counts.items()}
            min_prop = min(class_distribution.values()) if class_distribution else 0.5
            is_imbalanced = min_prop < 0.20

        return {
            "target_column": target_column,
            "task_type": task_type.value if hasattr(task_type, "value") else str(task_type),
            "is_imbalanced": is_imbalanced,
            "class_distribution": class_distribution,
            "distinct_targets": unique_count,
        }

    @classmethod
    def _infer_task_type(
        cls,
        series: pd.Series,
        unique_count: int,
        total: int,
        column_types: Optional[Dict[str, ColumnType]],
        target_column: str,
    ) -> TaskType:
        """Multi-signal heuristic that works on both small and large datasets.

        Scoring signals (each adds to a regression_score):
        +2  unique_ratio > 0.5   (high cardinality relative to rows)
        +1  unique_count > 10    (absolute cardinality)
        +1  float dtype          (continuous values)
        +1  large values > 100   (dollar amounts, counts, etc.)
        Threshold: regression_score >= 3 → REGRESSION
        """
        col_type = (
            column_types.get(target_column, ColumnType.UNKNOWN) if column_types else ColumnType.UNKNOWN
        )

        # Non-numeric columns are always classification
        if col_type not in (ColumnType.NUMERIC, ColumnType.UNKNOWN):
            return TaskType.CLASSIFICATION

        # If column_type is UNKNOWN, check pandas dtype directly
        if col_type == ColumnType.UNKNOWN and not pd.api.types.is_numeric_dtype(series):
            return TaskType.CLASSIFICATION

        unique_ratio = unique_count / max(total, 1)
        is_float = series.dtype in (np.float64, np.float32, float)
        has_large_values = bool((series.abs() > 100).any()) if len(series) > 0 else False

        regression_score = 0
        if unique_ratio > 0.5:
            regression_score += 2
        if unique_count > 10:
            regression_score += 1
        if is_float:
            regression_score += 1
        if has_large_values:
            regression_score += 1

        return TaskType.REGRESSION if regression_score >= 3 else TaskType.CLASSIFICATION

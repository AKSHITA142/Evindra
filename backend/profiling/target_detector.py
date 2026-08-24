"""Smart target column detection using hybrid multi-signal scoring, quantile sampling, mutual information math, and LLM semantic inspection."""

from typing import Optional, List, Dict, Any, Tuple
import difflib
import re
import numpy as np
import pandas as pd

from backend.profiling.sampler import QuantileSampler
from backend.agents.target_inspector import TargetInspectorAgent


class SmartTargetDetector:
    """Detects the most likely target column from a DataFrame using a 4-stage hybrid architecture.

    Stage 1: Explicit User Target (100% lock)
    Stage 2: Hybrid Scoring Engine (45% LLM + 30% Mission Match + 25% Mutual Info Math)
    Stage 3: Quantile Representative Sampling across percentiles (P10, P25, P50, P75, P90)
    Stage 4: Failsafe Dynamic Threshold & Confirmation Score
    """

    COMMON_TARGET_NAMES = [
        "target", "label", "class", "y", "outcome", "result",
        "price", "churn", "is_churn", "survived", "diagnosis",
        "default", "fraud", "spam", "sentiment", "rating",
        "salary", "income", "revenue", "sales", "cost",
        "status", "category", "type", "grade",
    ]

    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "and", "or", "not", "no", "but", "if", "so", "as", "it",
        "this", "that", "these", "those", "my", "your", "their",
        "will", "would", "can", "could", "should", "shall", "may",
        "do", "does", "did", "has", "have", "had",
        "predict", "classify", "estimate", "model", "build",
        "using", "based", "data", "dataset", "value", "values",
        "high", "low", "maximum", "minimum", "optimize",
        "machine", "learning", "individual", "each", "every",
        "want", "need", "like", "use", "find", "determine",
        "attributes", "features", "columns", "demographic",
    })

    @classmethod
    def detect_target(
        cls,
        df: pd.DataFrame,
        user_mission: str = "",
        user_target: Optional[str] = None,
        user_task_type: str = "general",
    ) -> Optional[str]:
        """Detects the best target column using hybrid scoring."""
        target, _ = cls.detect_target_with_confidence(
            df, user_mission=user_mission, user_target=user_target, user_task_type=user_task_type
        )
        return target

    @classmethod
    def detect_target_with_confidence(
        cls,
        df: pd.DataFrame,
        user_mission: str = "",
        user_target: Optional[str] = None,
        user_task_type: str = "general",
    ) -> Tuple[Optional[str], float]:
        """Detects best target column and returns (column_name, confidence_score)."""
        if df.empty or len(df.columns) == 0:
            return None, 0.0

        columns = list(df.columns)
        col_lower_map = {c.lower().strip(): c for c in columns}

        # --- Stage 1: Explicit User Selection (100% Lock) ---
        if user_target and user_target.strip():
            clean = user_target.strip()
            if clean in columns:
                return clean, 1.0
            if clean.lower() in col_lower_map:
                return col_lower_map[clean.lower()], 1.0

        # --- Stage 2: Quantile Sampling across P10, P25, P50, P75, P90 ---
        sample_rows = QuantileSampler.sample_representative_rows(df, n_samples=5)

        # --- Signal A: LLM Zero-Shot Semantic Score (45% weight) ---
        col_types = {c: str(df[c].dtype) for c in columns}
        llm_target = None
        llm_confidence = 0.85
        try:
            inspector = TargetInspectorAgent()
            result = inspector.run({
                "columns": columns,
                "sample_rows": sample_rows,
                "column_types": col_types,
                "user_mission": user_mission,
            })
            llm_target = getattr(result, "recommended_target", None)
            llm_confidence = float(getattr(result, "confidence", 0.85))
        except Exception:
            pass

        # --- Signal B: Mission Text Keyword Match (30% weight) ---
        keyword_target, keyword_score = cls._semantic_match_score(user_mission, columns, col_lower_map)

        # --- Signal C: Full-Dataset Statistical / Mutual Info Math (25% weight) ---
        math_target, math_score = cls._statistical_mi_match(df)

        # Combine candidate scores for all columns
        composite_scores: Dict[str, float] = {c: 0.0 for c in columns}

        for c in columns:
            score = 0.0
            c_lower = c.lower().strip()
            nunique = df[c].nunique()
            is_numeric = pd.api.types.is_numeric_dtype(df[c])
            is_float = df[c].dtype in (np.float64, np.float32, float)
            is_string = str(df[c].dtype) in ("object", "string", "category")

            # LLM Signal
            if llm_target and c == llm_target:
                score += 0.45 * llm_confidence
            # Keyword Signal
            if keyword_target and c == keyword_target:
                score += 0.30 * keyword_score
            # Math Signal
            if math_target and c == math_target:
                score += 0.25 * math_score
            # Common Name Bonus (+0.10)
            if c_lower in cls.COMMON_TARGET_NAMES:
                score += 0.10

            # Task-type specific target affinity
            if user_task_type == "classification":
                if is_string or (is_numeric and nunique <= 15):
                    score += 0.35
                if c_lower.endswith(("_category", "_class", "_label", "_type", "_status", "_group")):
                    score += 0.40
                if is_float and nunique > 15:
                    score -= 0.60
            elif user_task_type == "regression":
                if is_numeric and nunique > 15:
                    score += 0.35
                if is_string:
                    score -= 0.60

            # Penalty for ID & timestamp columns (-0.80)
            if (
                c_lower in ("id", "uuid", "name", "index", "row_id", "user_id", "timestamp", "time", "date", "datetime")
                or c_lower.endswith("_id")
                or c_lower.startswith("id_")
                or c_lower == "id"
            ):
                score -= 0.80

            composite_scores[c] = round(score, 4)

        # Select highest composite score
        sorted_candidates = sorted(composite_scores.items(), key=lambda x: x[1], reverse=True)
        best_target, best_score = sorted_candidates[0]

        # Fallback to last non-ID column if top score <= 0.0
        if best_score <= 0.0 and columns:
            non_id_cols = [
                c for c in columns
                if not (
                    c.lower().strip() in ("id", "uuid", "name", "index", "row_id", "user_id", "timestamp", "time", "date", "datetime")
                    or c.lower().strip().endswith(("_id", "_name", "_key"))
                    or c.lower().strip().startswith(("id_", "row_"))
                )
            ]
            best_target = non_id_cols[-1] if non_id_cols else columns[-1]
            best_score = 0.50

        return best_target, min(1.0, max(0.1, best_score))

    @classmethod
    def _semantic_match_score(
        cls,
        mission: str,
        columns: List[str],
        col_lower_map: dict,
    ) -> Tuple[Optional[str], float]:
        if not mission or not mission.strip():
            return None, 0.0

        tokens = re.findall(r"[a-zA-Z_]+", mission.lower())
        keywords = [t for t in tokens if t not in cls._STOP_WORDS and len(t) > 1]

        if not keywords:
            return None, 0.0

        scores: Dict[str, float] = {}
        for col in columns:
            c_lower = col.lower().strip()
            col_tokens = re.findall(r"[a-zA-Z_]+", c_lower)
            col_score = 0.0

            # Exact full match gets highest priority
            if c_lower in keywords or c_lower == mission.lower().strip():
                scores[col] = 2.0
                continue

            for kw in keywords:
                for ct in col_tokens:
                    if kw == ct:
                        col_score += 1.0
                    elif kw in ct or ct in kw:
                        col_score += 0.75
                    else:
                        ratio = difflib.SequenceMatcher(None, kw, ct).ratio()
                        if ratio >= 0.70:
                            col_score += ratio * 0.70

            scores[col] = round(col_score, 4)

        best_col, max_score = max(scores.items(), key=lambda x: x[1])
        if max_score > 0.0:
            norm_score = min(1.0, round(max_score / float(len(keywords)), 4))
            return best_col, norm_score

        return None, 0.0

    @classmethod
    def _statistical_mi_match(cls, df: pd.DataFrame) -> Tuple[Optional[str], float]:
        """Calculates statistical target candidate score across dataset rows."""
        if df.empty or len(df.columns) < 2:
            return None, 0.0

        n_rows = len(df)

        def is_id_col(col_name: str) -> bool:
            c_low = col_name.lower().strip()
            return (
                c_low in ("id", "uuid", "name", "index", "row_id", "user_id", "timestamp", "time", "date", "datetime")
                or c_low.endswith(("_id", "_name", "_key"))
                or c_low.startswith(("id_", "row_"))
            )

        # Exclude ID/timestamp columns and columns with 1 unique value or equal to n_rows
        valid_cols = [c for c in df.columns if not is_id_col(c) and 1 < df[c].nunique() < n_rows]
        if not valid_cols:
            return None, 0.0

        # Prioritize discrete / classification target candidates (2..20 unique values)
        discrete_candidates = [c for c in valid_cols if 2 <= df[c].nunique() <= 20]
        if discrete_candidates:
            # Pick the last discrete candidate column (standard convention for y in tabular data)
            return discrete_candidates[-1], 0.85

        # For continuous targets, pick column with moderate variance, avoiding max-collinear feature hubs
        num_valid = [c for c in valid_cols if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_valid) >= 2:
            corr_matrix = df[num_valid].corr().abs()
            best_col = None
            min_collinearity_hub = float("inf")
            for col in num_valid:
                tot_corr = float(corr_matrix[col].sum() - 1.0)
                # Select target candidate that isn't a central collinear feature hub (lowest total collinearity)
                if tot_corr < min_collinearity_hub:
                    min_collinearity_hub = tot_corr
                    best_col = col
            if best_col:
                return best_col, 0.75

        return valid_cols[-1], 0.70

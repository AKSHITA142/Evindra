from typing import Dict, Any, List, Optional
from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import (
    DecisionDomain,
    DecisionSource,
    ValidationStatus,
    DecisionRequest,
    DecisionResult,
)
from backend.core.confidence_policy import ConfidencePolicy, DEFAULT_CONFIDENCE_POLICY


class RuleEngine:
    """
    Deterministic Rule-Based Decision Engine for Evindra Phase 4.
    Evaluates structured DatasetProfile metrics to produce standardized DecisionResult objects.
    """

    def __init__(self, confidence_policy: Optional[ConfidencePolicy] = None):
        self.policy = confidence_policy or DEFAULT_CONFIDENCE_POLICY

    # --- 1. Missing Value Strategy ---
    def evaluate_missing_value_strategy(self, col_profile: ColumnProfileExtended) -> DecisionResult:
        """Determines missing value imputation strategy for a single column."""
        col_name = col_profile.name
        missing_ratio = col_profile.missing_ratio
        normalized_dtype = col_profile.normalized_dtype

        if missing_ratio == 0.0:
            return DecisionResult(
                domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                decision="NO_OP",
                confidence=1.0,
                reasoning=f"Column '{col_name}' has zero missing values.",
                evidence=["missing_ratio=0.0"],
                source=DecisionSource.RULE,
                requires_validation=False,
                validation_status=ValidationStatus.PASSED,
            )

        if missing_ratio >= 0.75:
            return DecisionResult(
                domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                decision="DROP_FEATURE",
                confidence=0.95,
                reasoning=f"Column '{col_name}' has extreme missingness ({missing_ratio:.1%}). Dropping feature to eliminate noise.",
                evidence=[f"missing_ratio={missing_ratio:.4f}"],
                alternatives=[{"decision": "IMPUTE_MISSING_INDICATOR", "confidence": 0.50}],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        if normalized_dtype == "numeric":
            skewness = col_profile.skewness or 0.0
            if abs(skewness) > 1.0:
                return DecisionResult(
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    decision="IMPUTE_MEDIAN",
                    confidence=0.95,
                    reasoning=f"Numeric column '{col_name}' is skewed (skewness={skewness:.2f}). Median imputation is robust to skewness.",
                    evidence=[f"missing_ratio={missing_ratio:.4f}", f"skewness={skewness:.2f}"],
                    alternatives=[{"decision": "IMPUTE_MEAN", "confidence": 0.60}],
                    source=DecisionSource.RULE,
                    requires_validation=True,
                )
            else:
                return DecisionResult(
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    decision="IMPUTE_MEAN",
                    confidence=0.95,
                    reasoning=f"Numeric column '{col_name}' is approximately symmetric (skewness={skewness:.2f}). Mean imputation is optimal.",
                    evidence=[f"missing_ratio={missing_ratio:.4f}", f"skewness={skewness:.2f}"],
                    alternatives=[{"decision": "IMPUTE_MEDIAN", "confidence": 0.85}],
                    source=DecisionSource.RULE,
                    requires_validation=True,
                )

        if normalized_dtype in ("categorical", "binary"):
            if missing_ratio < 0.10:
                return DecisionResult(
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    decision="IMPUTE_MODE",
                    confidence=0.90,
                    reasoning=f"Categorical column '{col_name}' has low missingness ({missing_ratio:.1%}). Mode imputation preserves dominant category.",
                    evidence=[f"missing_ratio={missing_ratio:.4f}"],
                    alternatives=[{"decision": "IMPUTE_EXPLICIT_CATEGORY", "confidence": 0.80}],
                    source=DecisionSource.RULE,
                    requires_validation=True,
                )
            else:
                return DecisionResult(
                    domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                    decision="IMPUTE_EXPLICIT_CATEGORY",
                    confidence=0.92,
                    reasoning=f"Categorical column '{col_name}' has moderate missingness ({missing_ratio:.1%}). Explicit 'missing' category prevents frequency distortion.",
                    evidence=[f"missing_ratio={missing_ratio:.4f}"],
                    alternatives=[{"decision": "IMPUTE_MODE", "confidence": 0.60}],
                    source=DecisionSource.RULE,
                    requires_validation=True,
                )

        return DecisionResult(
            domain=DecisionDomain.MISSING_VALUE_STRATEGY,
            decision="IMPUTE_EXPLICIT_CATEGORY",
            confidence=0.75,
            reasoning=f"Column '{col_name}' fallback imputation.",
            evidence=[f"missing_ratio={missing_ratio:.4f}"],
            source=DecisionSource.RULE,
            requires_validation=True,
        )

    # --- 2. Encoding Strategy ---
    def evaluate_encoding_strategy(self, col_profile: ColumnProfileExtended) -> DecisionResult:
        """Determines encoding strategy for a single column."""
        col_name = col_profile.name
        normalized_dtype = col_profile.normalized_dtype
        cardinality = col_profile.cardinality

        if normalized_dtype in ("numeric", "datetime"):
            return DecisionResult(
                domain=DecisionDomain.ENCODING_STRATEGY,
                decision="PASS_THROUGH",
                confidence=1.0,
                reasoning=f"Column '{col_name}' is {normalized_dtype} and does not require categorical encoding.",
                evidence=[f"normalized_dtype={normalized_dtype}"],
                source=DecisionSource.RULE,
                requires_validation=False,
                validation_status=ValidationStatus.PASSED,
            )

        if col_profile.constant_status:
            return DecisionResult(
                domain=DecisionDomain.ENCODING_STRATEGY,
                decision="DROP_FEATURE",
                confidence=1.0,
                reasoning=f"Column '{col_name}' is constant (1 distinct value).",
                evidence=["constant_status=True"],
                source=DecisionSource.RULE,
                requires_validation=False,
            )

        if col_profile.identifier_likelihood >= 0.8:
            return DecisionResult(
                domain=DecisionDomain.ENCODING_STRATEGY,
                decision="CLASSIFY_IDENTIFIER_AND_DROP",
                confidence=0.95,
                reasoning=f"Column '{col_name}' is an identifier (likelihood={col_profile.identifier_likelihood:.2f}). Dropping from model features.",
                evidence=[f"identifier_likelihood={col_profile.identifier_likelihood:.2f}"],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        if col_profile.ordinal_likelihood >= 0.7:
            return DecisionResult(
                domain=DecisionDomain.ENCODING_STRATEGY,
                decision="ORDINAL_ENCODING",
                confidence=0.88,
                reasoning=f"Column '{col_name}' has ordinal characteristics (ordinal_likelihood={col_profile.ordinal_likelihood:.2f}).",
                evidence=[f"ordinal_likelihood={col_profile.ordinal_likelihood:.2f}"],
                alternatives=[{"decision": "ONE_HOT_ENCODING", "confidence": 0.70}],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        if cardinality <= 10:
            return DecisionResult(
                domain=DecisionDomain.ENCODING_STRATEGY,
                decision="ONE_HOT_ENCODING",
                confidence=0.95,
                reasoning=f"Low-cardinality categorical column '{col_name}' (cardinality={cardinality}). One-hot encoding creates orthogonal binary features.",
                evidence=[f"cardinality={cardinality}"],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        if 10 < cardinality <= 50:
            return DecisionResult(
                domain=DecisionDomain.ENCODING_STRATEGY,
                decision="FREQUENCY_ENCODING",
                confidence=0.85,
                reasoning=f"Moderate-cardinality column '{col_name}' (cardinality={cardinality}). Frequency encoding captures category prevalence.",
                evidence=[f"cardinality={cardinality}"],
                alternatives=[{"decision": "RARE_CATEGORY_GROUPING_AND_ONE_HOT", "confidence": 0.80}],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        return DecisionResult(
            domain=DecisionDomain.ENCODING_STRATEGY,
            decision="TARGET_ENCODING_OUT_OF_FOLD",
            confidence=0.82,
            reasoning=f"High-cardinality column '{col_name}' (cardinality={cardinality}). Out-of-fold target encoding avoids high feature space explosion.",
            evidence=[f"cardinality={cardinality}"],
            alternatives=[{"decision": "FREQUENCY_ENCODING", "confidence": 0.75}],
            source=DecisionSource.RULE,
            requires_validation=True,
        )

    # --- 3. Scaling Transformation ---
    def evaluate_scaling_transformation(
        self, col_profile: ColumnProfileExtended, model_family: str = "general"
    ) -> DecisionResult:
        """Determines scaling transformation for a single column."""
        col_name = col_profile.name
        normalized_dtype = col_profile.normalized_dtype

        if model_family.lower() in ("tree", "random_forest", "xgboost", "lightgbm", "catboost"):
            return DecisionResult(
                domain=DecisionDomain.SCALING_TRANSFORMATION,
                decision="NO_SCALING",
                confidence=0.98,
                reasoning=f"Tree-based model family '{model_family}' is scale invariant.",
                evidence=[f"model_family={model_family}"],
                source=DecisionSource.RULE,
                requires_validation=False,
            )

        if normalized_dtype != "numeric" or col_profile.constant_status:
            return DecisionResult(
                domain=DecisionDomain.SCALING_TRANSFORMATION,
                decision="NO_SCALING",
                confidence=1.0,
                reasoning=f"Column '{col_name}' is not a variable numeric feature.",
                evidence=[f"normalized_dtype={normalized_dtype}"],
                source=DecisionSource.RULE,
                requires_validation=False,
            )

        outlier_ratio = col_profile.outlier_ratio
        skewness = abs(col_profile.skewness or 0.0)

        if outlier_ratio > 0.05:
            return DecisionResult(
                domain=DecisionDomain.SCALING_TRANSFORMATION,
                decision="ROBUST_SCALER",
                confidence=0.92,
                reasoning=f"Numeric column '{col_name}' has significant outliers ({outlier_ratio:.1%}). RobustScaler uses median and IQR.",
                evidence=[f"outlier_ratio={outlier_ratio:.4f}"],
                alternatives=[{"decision": "STANDARD_SCALER", "confidence": 0.60}],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        if skewness > 1.5:
            return DecisionResult(
                domain=DecisionDomain.SCALING_TRANSFORMATION,
                decision="POWER_TRANSFORM_ROBUST",
                confidence=0.85,
                reasoning=f"Numeric column '{col_name}' is strongly skewed (skewness={skewness:.2f}). Power/log transform stabilizes variance.",
                evidence=[f"skewness={skewness:.2f}"],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        return DecisionResult(
            domain=DecisionDomain.SCALING_TRANSFORMATION,
            decision="STANDARD_SCALER",
            confidence=0.95,
            reasoning=f"Numeric column '{col_name}' is symmetric with low outliers. StandardScaler is optimal.",
            evidence=[f"skewness={skewness:.2f}", f"outlier_ratio={outlier_ratio:.4f}"],
            source=DecisionSource.RULE,
            requires_validation=True,
        )

    # --- 4. Outlier Handling ---
    def evaluate_outlier_handling(
        self, col_profile: ColumnProfileExtended, model_family: str = "general"
    ) -> DecisionResult:
        """Determines outlier handling strategy for a single column."""
        col_name = col_profile.name
        outlier_ratio = col_profile.outlier_ratio

        if col_profile.normalized_dtype != "numeric" or outlier_ratio == 0.0:
            return DecisionResult(
                domain=DecisionDomain.OUTLIER_HANDLING,
                decision="KEEP_OUTLIERS",
                confidence=1.0,
                reasoning=f"Column '{col_name}' has zero detected outliers.",
                evidence=["outlier_ratio=0.0"],
                source=DecisionSource.RULE,
                requires_validation=False,
            )

        if model_family.lower() in ("tree", "random_forest", "xgboost", "lightgbm"):
            return DecisionResult(
                domain=DecisionDomain.OUTLIER_HANDLING,
                decision="KEEP_OUTLIERS",
                confidence=0.95,
                reasoning=f"Tree model family '{model_family}' handles extreme values naturally via split thresholds.",
                evidence=[f"model_family={model_family}"],
                source=DecisionSource.RULE,
                requires_validation=False,
            )

        if outlier_ratio > 0.10:
            return DecisionResult(
                domain=DecisionDomain.OUTLIER_HANDLING,
                decision="WINSORIZE_CLIPPING",
                confidence=0.85,
                reasoning=f"Column '{col_name}' has high outlier ratio ({outlier_ratio:.1%}). Clipping extreme values to 1st/99th percentiles bounds leverage.",
                evidence=[f"outlier_ratio={outlier_ratio:.4f}"],
                source=DecisionSource.RULE,
                requires_validation=True,
            )

        return DecisionResult(
            domain=DecisionDomain.OUTLIER_HANDLING,
            decision="ROBUST_SCALING_ONLY",
            confidence=0.90,
            reasoning=f"Column '{col_name}' has moderate outlier ratio ({outlier_ratio:.1%}). Using robust scaling without row deletion.",
            evidence=[f"outlier_ratio={outlier_ratio:.4f}"],
            source=DecisionSource.RULE,
            requires_validation=True,
        )

    # --- 5. Feature Selection ---
    def evaluate_feature_selection(self, dataset_profile: DatasetProfile) -> DecisionResult:
        """Evaluates dataset-wide feature selection requirements."""
        constant_cols = [c.name for c in dataset_profile.detailed_column_profiles if c.constant_status]
        near_constant_cols = [c.name for c in dataset_profile.detailed_column_profiles if c.near_constant_status]

        actions = []
        if constant_cols:
            actions.append(f"drop_constant:{','.join(constant_cols)}")
        if near_constant_cols:
            actions.append(f"drop_near_constant:{','.join(near_constant_cols)}")

        decision_str = ";".join(actions) if actions else "VARIANCE_AND_CORRELATION_FILTER"

        return DecisionResult(
            domain=DecisionDomain.FEATURE_SELECTION,
            decision=decision_str,
            confidence=0.92 if actions else 0.85,
            reasoning=f"Identified {len(constant_cols)} constant and {len(near_constant_cols)} near-constant features.",
            evidence=[f"constant_count={len(constant_cols)}", f"near_constant_count={len(near_constant_cols)}"],
            metadata={"constant_columns": constant_cols, "near_constant_columns": near_constant_cols},
            source=DecisionSource.RULE,
            requires_validation=True,
        )

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
            decision="ROBUST_SCALER",
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

    # --- 6. Column Intelligence ---
    def evaluate_column_intelligence(self, col_profile: ColumnProfileExtended) -> DecisionResult:
        """Evaluates semantic classification for a single column."""
        col_name = col_profile.name
        if col_profile.identifier_likelihood >= 0.8:
            return DecisionResult(
                domain=DecisionDomain.COLUMN_INTELLIGENCE,
                decision="CLASSIFY_IDENTIFIER",
                confidence=0.95,
                reasoning=f"Column '{col_name}' has high identifier likelihood ({col_profile.identifier_likelihood:.2f}).",
                evidence=[f"identifier_likelihood={col_profile.identifier_likelihood:.2f}"],
                source=DecisionSource.RULE,
            )
        if col_profile.datetime_likelihood >= 0.8:
            return DecisionResult(
                domain=DecisionDomain.COLUMN_INTELLIGENCE,
                decision="CLASSIFY_DATETIME",
                confidence=0.95,
                reasoning=f"Column '{col_name}' has high datetime likelihood.",
                evidence=[f"datetime_likelihood={col_profile.datetime_likelihood:.2f}"],
                source=DecisionSource.RULE,
            )
        if col_profile.text_likelihood >= 0.8:
            return DecisionResult(
                domain=DecisionDomain.COLUMN_INTELLIGENCE,
                decision="CLASSIFY_TEXT",
                confidence=0.90,
                reasoning=f"Column '{col_name}' is free-form text.",
                evidence=[f"text_likelihood={col_profile.text_likelihood:.2f}"],
                source=DecisionSource.RULE,
            )
        if col_profile.ordinal_likelihood >= 0.7:
            return DecisionResult(
                domain=DecisionDomain.COLUMN_INTELLIGENCE,
                decision="CLASSIFY_ORDINAL",
                confidence=0.88,
                reasoning=f"Column '{col_name}' exhibits ordinal categorical properties.",
                evidence=[f"ordinal_likelihood={col_profile.ordinal_likelihood:.2f}"],
                source=DecisionSource.RULE,
            )

        # Ambiguous case: return low confidence to trigger escalation
        return DecisionResult(
            domain=DecisionDomain.COLUMN_INTELLIGENCE,
            decision="AMBIGUOUS_TYPE",
            confidence=0.50,
            reasoning=f"Column '{col_name}' semantic classification is ambiguous from rules alone.",
            evidence=[f"unique_ratio={col_profile.unique_ratio:.2f}", f"dtype={col_profile.dtype}"],
            source=DecisionSource.RULE,
        )

    # --- 7. Target Detection ---
    def evaluate_target_detection(self, dataset_profile: DatasetProfile) -> DecisionResult:
        """Determines target column and problem type."""
        if dataset_profile.target_column:
            prob_type = dataset_profile.problem_type_candidates[0] if dataset_profile.problem_type_candidates else "general"
            return DecisionResult(
                domain=DecisionDomain.TARGET_DETECTION,
                decision=f"TARGET:{dataset_profile.target_column}:{prob_type}",
                confidence=1.0,
                reasoning=f"Target column '{dataset_profile.target_column}' is explicitly specified.",
                evidence=[f"target_column={dataset_profile.target_column}"],
                source=DecisionSource.RULE,
            )

        if dataset_profile.target_candidate_list or dataset_profile.target_candidates:
            cands = dataset_profile.target_candidates or dataset_profile.target_candidate_list
            best = cands[0]
            if best.get("score", 0.0) >= 0.8:
                return DecisionResult(
                    domain=DecisionDomain.TARGET_DETECTION,
                    decision=f"TARGET:{best['column']}:general",
                    confidence=best["score"],
                    reasoning=f"Target candidate '{best['column']}' detected with high confidence.",
                    evidence=[f"candidate_score={best['score']}"],
                    source=DecisionSource.RULE,
                )

        return DecisionResult(
            domain=DecisionDomain.TARGET_DETECTION,
            decision="NO_TARGET_DETECTED",
            confidence=0.40,
            reasoning="No explicit or high-confidence target column detected in dataset profile.",
            evidence=["target_column=None"],
            source=DecisionSource.RULE,
        )

    # --- 8. Leakage Detection ---
    def evaluate_leakage_detection(self, dataset_profile: DatasetProfile) -> DecisionResult:
        """Evaluates potential data leakage features."""
        target_col = dataset_profile.target_column
        rel_map = dataset_profile.feature_target_relationships or {}
        leak_cols = [col for col, corr in rel_map.items() if abs(corr) >= 0.99]

        if leak_cols:
            return DecisionResult(
                domain=DecisionDomain.LEAKAGE_DETECTION,
                decision=f"FLAG_LEAKAGE:{','.join(leak_cols)}",
                confidence=0.98,
                reasoning=f"Features {leak_cols} show near-perfect correlation (>=0.99) with target '{target_col}'. Flagged as leakage risk.",
                evidence=[f"high_corr_features={leak_cols}"],
                source=DecisionSource.RULE,
            )

        return DecisionResult(
            domain=DecisionDomain.LEAKAGE_DETECTION,
            decision="NO_LEAKAGE_DETECTED",
            confidence=0.90,
            reasoning="No features show near-perfect correlation with target column.",
            evidence=["max_correlation<0.99"],
            source=DecisionSource.RULE,
        )

    # --- 9. Feature Engineering ---
    def evaluate_feature_engineering(self, col_profile: ColumnProfileExtended) -> DecisionResult:
        """Evaluates feature engineering rules for a column."""
        if col_profile.normalized_dtype == "datetime" or col_profile.datetime_likelihood >= 0.8:
            return DecisionResult(
                domain=DecisionDomain.FEATURE_ENGINEERING,
                decision="EXTRACT_DATETIME_COMPONENTS",
                confidence=0.95,
                reasoning=f"Datetime column '{col_profile.name}' is suitable for component extraction (year, month, day, dayofweek, hour).",
                evidence=[f"datetime_likelihood={col_profile.datetime_likelihood:.2f}"],
                source=DecisionSource.RULE,
            )

        return DecisionResult(
            domain=DecisionDomain.FEATURE_ENGINEERING,
            decision="PASS_THROUGH",
            confidence=0.60,
            reasoning=f"No high-confidence deterministic feature engineering rule applies for column '{col_profile.name}'.",
            evidence=[f"normalized_dtype={col_profile.normalized_dtype}"],
            source=DecisionSource.RULE,
        )

    # --- 10. Pipeline Strategy ---
    def evaluate_pipeline_strategy(self, dataset_profile: DatasetProfile) -> DecisionResult:
        """Determines overarching preprocessing pipeline topology."""
        rows = dataset_profile.rows
        cols = dataset_profile.columns
        missing_ratio = dataset_profile.dataset_wide_missingness

        if rows < 100 or cols < 3:
            strategy = "LIGHTWEIGHT_PREPROCESSING"
        elif missing_ratio > 0.30:
            strategy = "IMPUTATION_HEAVY_PREPROCESSING"
        else:
            strategy = "STANDARD_PREPROCESSING"

        return DecisionResult(
            domain=DecisionDomain.PIPELINE_STRATEGY,
            decision=strategy,
            confidence=0.90,
            reasoning=f"Dataset topology ({rows} rows, {cols} cols, {missing_ratio:.1%} missingness) mapped to pipeline strategy '{strategy}'.",
            evidence=[f"rows={rows}", f"columns={cols}", f"missing_ratio={missing_ratio:.4f}"],
            source=DecisionSource.RULE,
        )

    # --- Unified Request Evaluator ---
    def evaluate_request(self, request: DecisionRequest, dataset_profile: DatasetProfile) -> DecisionResult:
        """
        Unified entrypoint: Evaluates any DecisionRequest using deterministic rules.
        """
        domain = request.domain
        col_profile = None
        if request.column_name:
            for col in dataset_profile.detailed_column_profiles:
                if col.name == request.column_name:
                    col_profile = col
                    break

        if domain == DecisionDomain.MISSING_VALUE_STRATEGY and col_profile:
            return self.evaluate_missing_value_strategy(col_profile)
        elif domain == DecisionDomain.ENCODING_STRATEGY and col_profile:
            return self.evaluate_encoding_strategy(col_profile)
        elif domain == DecisionDomain.SCALING_TRANSFORMATION and col_profile:
            model_family = request.context.get("model_family", "general")
            return self.evaluate_scaling_transformation(col_profile, model_family=model_family)
        elif domain == DecisionDomain.OUTLIER_HANDLING and col_profile:
            model_family = request.context.get("model_family", "general")
            return self.evaluate_outlier_handling(col_profile, model_family=model_family)
        elif domain == DecisionDomain.FEATURE_SELECTION:
            return self.evaluate_feature_selection(dataset_profile)
        elif domain == DecisionDomain.COLUMN_INTELLIGENCE and col_profile:
            return self.evaluate_column_intelligence(col_profile)
        elif domain == DecisionDomain.TARGET_DETECTION:
            return self.evaluate_target_detection(dataset_profile)
        elif domain == DecisionDomain.LEAKAGE_DETECTION:
            return self.evaluate_leakage_detection(dataset_profile)
        elif domain == DecisionDomain.FEATURE_ENGINEERING and col_profile:
            return self.evaluate_feature_engineering(col_profile)
        elif domain == DecisionDomain.PIPELINE_STRATEGY:
            return self.evaluate_pipeline_strategy(dataset_profile)

        # Default fallback for unhandled or profile-less queries
        return DecisionResult(
            domain=domain,
            decision="NO_ACTION",
            confidence=0.50,
            reasoning=f"No explicit deterministic rule matched request for domain '{domain}'.",
            evidence=[],
            source=DecisionSource.RULE,
        )


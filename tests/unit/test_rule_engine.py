import pytest
import pandas as pd

from backend.core.confidence_policy import ConfidencePolicy, DomainConfidenceThresholds
from backend.engine.rule_engine import RuleEngine
from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionDomain, DecisionSource, ValidationStatus, DecisionResult


def test_confidence_policy_custom_thresholds():
    """Verify ConfidencePolicy allows configurable per-domain thresholds and escalation checks."""
    custom_thresholds = {
        DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(
            rule_strong=0.92,
            rule_acceptable=0.80,
        )
    }
    policy = ConfidencePolicy(domain_overrides=custom_thresholds)

    # Missing value strategy domain
    assert policy.is_rule_high_confidence(DecisionDomain.MISSING_VALUE_STRATEGY, 0.95) is True
    assert policy.is_rule_high_confidence(DecisionDomain.MISSING_VALUE_STRATEGY, 0.90) is False  # Below 0.92
    assert policy.should_escalate_rule(DecisionDomain.MISSING_VALUE_STRATEGY, 0.78) is True  # Below 0.80

    # Default encoding domain
    assert policy.is_rule_high_confidence(DecisionDomain.ENCODING_STRATEGY, 0.90) is True


def test_rule_engine_missing_value_decisions():
    """Verify RuleEngine evaluates deterministic missing value strategies."""
    engine = RuleEngine()

    # 1. Zero missing
    col_zero = ColumnProfileExtended(name="age", normalized_dtype="numeric", missing_ratio=0.0)
    res_zero = engine.evaluate_missing_value_strategy(col_zero)
    assert res_zero.decision == "NO_OP"
    assert res_zero.confidence == 1.0

    # 2. Extreme missing (>= 75%)
    col_extreme = ColumnProfileExtended(name="sparse_col", normalized_dtype="numeric", missing_ratio=0.80)
    res_extreme = engine.evaluate_missing_value_strategy(col_extreme)
    assert res_extreme.decision == "DROP_FEATURE"
    assert res_extreme.confidence >= 0.90

    # 3. Numeric skewed missing
    col_skewed = ColumnProfileExtended(name="income", normalized_dtype="numeric", missing_ratio=0.05, skewness=2.5)
    res_skewed = engine.evaluate_missing_value_strategy(col_skewed)
    assert res_skewed.decision == "IMPUTE_MEDIAN"
    assert res_skewed.confidence == 0.95

    # 4. Numeric symmetric missing
    col_sym = ColumnProfileExtended(name="height", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1)
    res_sym = engine.evaluate_missing_value_strategy(col_sym)
    assert res_sym.decision == "IMPUTE_MEAN"
    assert res_sym.confidence == 0.95

    # 5. Categorical low missing (< 10%)
    col_cat_low = ColumnProfileExtended(name="city", normalized_dtype="categorical", missing_ratio=0.03)
    res_cat_low = engine.evaluate_missing_value_strategy(col_cat_low)
    assert res_cat_low.decision == "IMPUTE_MODE"

    # 6. Categorical moderate missing (>= 10%)
    col_cat_mod = ColumnProfileExtended(name="city", normalized_dtype="categorical", missing_ratio=0.15)
    res_cat_mod = engine.evaluate_missing_value_strategy(col_cat_mod)
    assert res_cat_mod.decision == "IMPUTE_EXPLICIT_CATEGORY"


def test_rule_engine_encoding_decisions():
    """Verify RuleEngine evaluates categorical encoding strategies based on cardinality and likelihoods."""
    engine = RuleEngine()

    # Low cardinality (<= 10) -> One-hot
    col_low = ColumnProfileExtended(name="gender", normalized_dtype="categorical", cardinality=2)
    res_low = engine.evaluate_encoding_strategy(col_low)
    assert res_low.decision == "ONE_HOT_ENCODING"

    # Moderate cardinality (10-50) -> Frequency
    col_mod = ColumnProfileExtended(name="zipcode_prefix", normalized_dtype="categorical", cardinality=35)
    res_mod = engine.evaluate_encoding_strategy(col_mod)
    assert res_mod.decision == "FREQUENCY_ENCODING"

    # High cardinality (> 50) -> Target Encoding Out-of-fold
    col_high = ColumnProfileExtended(name="user_id_hash", normalized_dtype="categorical", cardinality=150)
    res_high = engine.evaluate_encoding_strategy(col_high)
    assert res_high.decision == "TARGET_ENCODING_OUT_OF_FOLD"

    # Identifier column -> Classify ID and drop
    col_id = ColumnProfileExtended(name="transaction_id", normalized_dtype="text", identifier_likelihood=0.95)
    res_id = engine.evaluate_encoding_strategy(col_id)
    assert res_id.decision == "CLASSIFY_IDENTIFIER_AND_DROP"


def test_rule_engine_scaling_and_outliers():
    """Verify scaling and outlier decisions adhere to model family invariants."""
    engine = RuleEngine()

    col_num = ColumnProfileExtended(name="price", normalized_dtype="numeric", outlier_ratio=0.08)

    # Tree model -> No scaling
    res_tree_scale = engine.evaluate_scaling_transformation(col_num, model_family="xgboost")
    assert res_tree_scale.decision == "NO_SCALING"
    assert res_tree_scale.confidence >= 0.95

    # Linear model with outliers -> RobustScaler
    res_lin_scale = engine.evaluate_scaling_transformation(col_num, model_family="linear")
    assert res_lin_scale.decision == "ROBUST_SCALER"

    # Outliers with tree model -> Keep outliers
    res_out_tree = engine.evaluate_outlier_handling(col_num, model_family="random_forest")
    assert res_out_tree.decision == "KEEP_OUTLIERS"

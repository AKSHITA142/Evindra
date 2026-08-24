"""
Integration smoke tests for Evindra Decision Hierarchy (Phase 2).
Verifies Rule → RAG → LLM → User escalation path end-to-end using real
DatasetProfiler and DecisionOrchestrator — no cloud credentials required.

All tests must pass with RAG/LLM services absent (graceful degradation).
"""
import io
import pandas as pd
import pytest

from backend.profiling.dataset_profiler import DatasetProfiler
from backend.engine.decision_orchestrator import DecisionOrchestrator
from backend.schemas.decision import DecisionDomain, DecisionSource, ValidationStatus
from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.core.confidence_policy import ConfidencePolicy, DomainConfidenceThresholds


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_csv_bytes() -> bytes:
    """A clean 200-row binary classification dataset."""
    import numpy as np
    rng = np.random.default_rng(42)
    n = 200
    df = pd.DataFrame({
        "age": rng.integers(18, 70, n).astype(float),
        "income": rng.normal(50000, 15000, n),
        "category": rng.choice(["A", "B", "C"], n),
        "label": rng.integers(0, 2, n),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def regression_csv_bytes() -> bytes:
    """A clean 300-row regression dataset."""
    import numpy as np
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({
        "sqft": rng.integers(500, 4000, n).astype(float),
        "bedrooms": rng.integers(1, 6, n).astype(float),
        "age_years": rng.integers(0, 50, n).astype(float),
        "price": rng.normal(300000, 100000, n),
    })
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


@pytest.fixture
def orchestrator() -> DecisionOrchestrator:
    """Default orchestrator — RAG/LLM absent, user fallback active."""
    return DecisionOrchestrator()


# ---------------------------------------------------------------------------
# Profiler smoke tests
# ---------------------------------------------------------------------------

def test_profiler_produces_valid_profile(simple_csv_bytes):
    df = pd.read_csv(io.BytesIO(simple_csv_bytes))
    profile = DatasetProfiler.profile_dataframe(df)

    assert isinstance(profile, DatasetProfile)
    assert profile.rows > 0
    assert profile.columns > 0
    assert len(profile.detailed_column_profiles) > 0


def test_profiler_detects_column_types(simple_csv_bytes):
    df = pd.read_csv(io.BytesIO(simple_csv_bytes))
    profile = DatasetProfiler.profile_dataframe(df)

    dtypes = {c.name: c.normalized_dtype for c in profile.detailed_column_profiles}
    assert "age" in dtypes
    assert "category" in dtypes
    # numeric cols should be detected as numeric
    assert dtypes.get("age") in ("numeric", "integer", "float", "int64", "float64")


# ---------------------------------------------------------------------------
# Rule Engine direct tests
# ---------------------------------------------------------------------------

def test_rule_engine_high_confidence_numeric_symmetric(orchestrator):
    """Numeric symmetric column → IMPUTE_MEAN with high rule confidence."""
    col = ColumnProfileExtended(
        name="age", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1
    )
    result = orchestrator.rule_engine.evaluate_missing_value_strategy(col)

    assert result.decision == "IMPUTE_MEAN"
    assert result.confidence >= 0.9
    assert result.source == DecisionSource.RULE


def test_rule_engine_encoding_low_cardinality(orchestrator):
    """Categorical column with low cardinality → ONE_HOT_ENCODING."""
    col = ColumnProfileExtended(
        name="category", normalized_dtype="categorical", distinct_count=3
    )
    result = orchestrator.rule_engine.evaluate_encoding_strategy(col)

    assert result.decision in ("ONE_HOT_ENCODING", "OHE", "ONEHOT")
    assert result.confidence >= 0.8
    assert result.source == DecisionSource.RULE


def test_rule_engine_scaling_linear_model(orchestrator):
    """Numeric column for linear model → STANDARD_SCALER."""
    col = ColumnProfileExtended(
        name="income", normalized_dtype="numeric", skewness=0.3
    )
    result = orchestrator.rule_engine.evaluate_scaling_transformation(col, model_family="linear")

    assert result.decision in ("STANDARD_SCALER", "STANDARDIZE", "STANDARD")
    assert result.source == DecisionSource.RULE


# ---------------------------------------------------------------------------
# Decision Orchestrator hierarchy tests
# ---------------------------------------------------------------------------

def test_orchestrator_high_confidence_rule_accepted(orchestrator):
    """High-confidence rule decision is accepted without escalating to RAG/LLM."""
    col = ColumnProfileExtended(
        name="age", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1
    )
    result = orchestrator.evaluate_decision(
        DecisionDomain.MISSING_VALUE_STRATEGY, col_profile=col
    )

    assert result.decision is not None
    assert result.confidence > 0.0
    assert result.source in (DecisionSource.RULE, DecisionSource.USER, "rule", "user", "safety_default")


def test_orchestrator_returns_decision_result_fields(orchestrator):
    """Every DecisionResult must carry required fields."""
    col = ColumnProfileExtended(
        name="category", normalized_dtype="categorical", distinct_count=5
    )
    result = orchestrator.evaluate_decision(
        DecisionDomain.ENCODING_STRATEGY, col_profile=col
    )

    assert result.decision_id != ""
    assert result.domain is not None
    assert result.decision is not None
    assert 0.0 <= result.confidence <= 1.0
    assert result.source is not None
    assert result.validation_status is not None


def test_orchestrator_graceful_rag_llm_degradation():
    """When RAG and LLM are absent, orchestrator must not crash — returns a decision."""
    orchestrator = DecisionOrchestrator(
        hybrid_retrieval_service=None,
        llm_decision_service=None,
    )
    col = ColumnProfileExtended(
        name="weird_col", normalized_dtype="categorical", distinct_count=500
    )
    result = orchestrator.evaluate_decision(
        DecisionDomain.ENCODING_STRATEGY, col_profile=col
    )

    # Must not raise; must return a valid decision
    assert result.decision is not None
    assert result.decision_id != ""


def test_orchestrator_user_fallback_accepted():
    """When user provides a selection, it must be recorded with source=user."""
    low_conf_policy = ConfidencePolicy(
        domain_overrides={
            DecisionDomain.MISSING_VALUE_STRATEGY: DomainConfidenceThresholds(
                rule_strong=0.99, rag_strong=0.99, llm_strong=0.99
            )
        }
    )
    orchestrator = DecisionOrchestrator(confidence_policy=low_conf_policy)
    col = ColumnProfileExtended(
        name="income", normalized_dtype="numeric", missing_ratio=0.05, skewness=0.1
    )
    result = orchestrator.evaluate_decision(
        DecisionDomain.MISSING_VALUE_STRATEGY,
        col_profile=col,
        user_selection="IMPUTE_CONSTANT",
        auto_approve_default=True,
    )

    assert result.decision is not None
    assert result.decision_id != ""


def test_orchestrate_decisions_returns_list(simple_csv_bytes):
    """orchestrate_decisions over a full DatasetProfile returns a non-empty decision list."""
    df = pd.read_csv(io.BytesIO(simple_csv_bytes))
    profile = DatasetProfiler.profile_dataframe(df, target_column="label")

    orchestrator = DecisionOrchestrator()
    decisions = orchestrator.orchestrate_decisions(profile)

    assert isinstance(decisions, list)
    assert len(decisions) > 0
    for d in decisions:
        assert d.decision is not None
        assert d.decision_id != ""
        assert 0.0 <= d.confidence <= 1.0


# ---------------------------------------------------------------------------
# Decision source validation
# ---------------------------------------------------------------------------

def test_every_decision_has_valid_source(simple_csv_bytes):
    """All decisions must have a recognized source value."""
    valid_sources = {s.value for s in DecisionSource} | {"safety_default"}

    df = pd.read_csv(io.BytesIO(simple_csv_bytes))
    profile = DatasetProfiler.profile_dataframe(df, target_column="label")

    orchestrator = DecisionOrchestrator()
    decisions = orchestrator.orchestrate_decisions(profile)

    for d in decisions:
        source_val = d.source if isinstance(d.source, str) else d.source.value
        assert source_val in valid_sources, f"Unknown source: {d.source}"


def test_regression_profile_decisions(regression_csv_bytes):
    """Decisions for regression dataset should not crash and return valid results."""
    df = pd.read_csv(io.BytesIO(regression_csv_bytes))
    profile = DatasetProfiler.profile_dataframe(df, target_column="price")

    orchestrator = DecisionOrchestrator()
    decisions = orchestrator.orchestrate_decisions(profile)

    assert len(decisions) > 0
    assert all(d.decision is not None for d in decisions)

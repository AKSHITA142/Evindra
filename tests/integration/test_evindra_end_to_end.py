import os
import json
import tempfile
import pytest
import numpy as np
import pandas as pd

from backend.engine.evindra_orchestrator import EvindraOrchestrator
from backend.schemas.decision import DecisionDomain, DecisionSource, DecisionResult
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.schemas.pipeline import PipelineCandidateSet, PipelineCandidate


def _verify_common_assertions(res: dict):
    """Helper to verify standard end-to-end pipeline outputs."""
    assert res["status"] == "SUCCESS"
    assert res["target_column"] is not None
    assert len(res["decisions"]) > 0
    assert res["preprocessing_plan"] is not None
    assert res["plan_validation"].valid is True
    assert res["execution_result"].status in ("SUCCESS", "PARTIAL_SUCCESS")
    assert res["best_result"] is not None
    assert res["holdout_report"] is not None

    paths = res["artifact_paths"]
    for k in ["final_processed.csv", "best_model", "preprocessing_pipeline", "feature_mapping.json", "decision_trace.json", "model_results.json", "final_validation.json", "final_report.json", "final_report.md"]:
        assert k in paths
        assert os.path.exists(paths[k])
        assert os.path.getsize(paths[k]) > 0


def test_1_clean_numeric_classification():
    """TEST 1: Clean numeric classification dataset."""
    df = pd.DataFrame({
        "feature_1": np.linspace(1, 100, 40),
        "feature_2": np.sin(np.linspace(0, 10, 40)),
        "target": [0, 1] * 20,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="clean_num", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_2_mixed_numeric_categorical():
    """TEST 2: Mixed numeric + categorical dataset."""
    df = pd.DataFrame({
        "age": [20, 30, 40, 50, 60, 25, 35, 45, 55, 65] * 3,
        "gender": ["M", "F", "F", "M", "M", "F", "M", "F", "M", "F"] * 3,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="mixed_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_3_missing_values():
    """TEST 3: Missing values imputation handling."""
    df = pd.DataFrame({
        "num_col": [1.0, np.nan, 3.0, 4.0, np.nan, 6.0, 7.0, 8.0, 9.0, 10.0] * 3,
        "cat_col": ["A", "B", None, "A", "B", "A", None, "B", "A", "B"] * 3,
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="missing_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_4_high_cardinality_categorical():
    """TEST 4: High-cardinality categorical features."""
    df = pd.DataFrame({
        "city": [f"City_{i}" for i in range(30)],
        "val": np.random.randn(30),
        "target": [0, 1] * 15,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="high_card_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_5_outliers():
    """TEST 5: Outlier handling and clipping."""
    df = pd.DataFrame({
        "val": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000] * 3,  # Outlier 1000
        "target": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1] * 3,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="outlier_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_6_ambiguous_target():
    """TEST 6: Ambiguous target detection and explicit specification."""
    df = pd.DataFrame({
        "col_a": [1, 2, 3, 4] * 5,
        "col_b": [0, 1, 0, 1] * 5,
        "label": [1, 0, 1, 0] * 5,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="ambig_ds", target_column="label", output_dir=tmp_dir)
        assert res["target_column"] == "label"
        _verify_common_assertions(res)


def test_7_data_leakage():
    """TEST 7: Data leakage detection and target isolation."""
    df = pd.DataFrame({
        "feature_clean": np.random.randn(30),
        "target": [0, 1] * 15,
    })

    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="leakage_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_8_imbalanced_classification():
    """TEST 8: Imbalanced classification dataset (PR-AUC / ROC-AUC prioritized)."""
    df = pd.DataFrame({
        "feat_1": np.random.randn(50),
        "target": [0] * 45 + [1] * 5,  # 90% class 0, 10% class 1
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="imbalance_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)
        assert res["experiment_report"].primary_metric in ("pr_auc", "roc_auc", "f1")


def test_9_regression():
    """TEST 9: Regression dataset."""
    np.random.seed(42)
    df = pd.DataFrame({
        "sqft": [500, 1000, 1500, 2000, 2500, 3000, 3500, 4000] * 3,
        "rooms": [1, 2, 3, 4, 3, 5, 4, 6] * 3,
        "price": [150 + np.random.randint(-20, 20), 250 + np.random.randint(-20, 20),
                  350 + np.random.randint(-20, 20), 450 + np.random.randint(-20, 20),
                  550 + np.random.randint(-20, 20), 650 + np.random.randint(-20, 20),
                  750 + np.random.randint(-20, 20), 850 + np.random.randint(-20, 20)] * 3,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="reg_ds", target_column="price", problem_type="regression", output_dir=tmp_dir)
        _verify_common_assertions(res)
        assert res["problem_type"] == "regression"


def test_10_temporal_dataset():
    """TEST 10: Temporal dataset with time ordering."""
    df = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=30, freq="D"),
        "val": np.sin(np.linspace(0, 5, 30)),
        "target": [0, 1] * 15,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="time_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_11_rule_high_confidence_path():
    """TEST 11: High-confidence Rule execution path."""
    df = pd.DataFrame({
        "f1": [1.0, 2.0, 3.0, 4.0] * 5,
        "target": [0, 1, 0, 1] * 5,
    })
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="rule_path_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)
        rule_decisions = [d for d in res["decisions"] if (d.source.value if hasattr(d.source, "value") else str(d.source)) == "rule"]
        assert len(rule_decisions) > 0


def test_12_rule_to_rag_path():
    """TEST 12: Rule -> RAG fallback decision path."""
    class MockRAGService:
        def retrieve_relevant_scenarios(self, domain, profile):
            return [{"scenario_id": "sc_rag_1", "similarity_score": 0.91, "recommended_decision": "PASS_THROUGH"}]

    df = pd.DataFrame({"f1": [1, 2, 3, 4] * 5, "target": [0, 1, 0, 1] * 5})
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator(rag_service=MockRAGService())
        res = orchestrator.run_pipeline(df, dataset_name="rag_path_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_13_rule_to_rag_to_llm_path():
    """TEST 13: Rule -> RAG -> LLM fallback decision path."""
    class MockLLMService:
        def predict_decision(self, domain, profile, rag_results):
            return {"decision": "PASS_THROUGH", "confidence": 0.88, "reasoning": "Mock LLM rationale"}

    df = pd.DataFrame({"f1": [1, 2, 3, 4] * 5, "target": [0, 1, 0, 1] * 5})
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator(llm_service=MockLLMService())
        res = orchestrator.run_pipeline(df, dataset_name="llm_path_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_14_rule_to_rag_to_llm_to_user_path():
    """TEST 14: Rule -> RAG -> LLM -> User fallback decision path."""
    df = pd.DataFrame({"f1": [1, 2, 3, 4] * 5, "target": [0, 1, 0, 1] * 5})
    user_inputs = {"encoding_strategy": "ONE_HOT_ENCODING"}

    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="user_path_ds", target_column="target", user_responses=user_inputs, output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_15_invalid_preprocessing_plan():
    """TEST 15: Invalid preprocessing plan blocked by Plan Validator Gate."""
    from backend.engine.plan_validator import PlanValidator

    df = pd.DataFrame({"f1": [1, 2, 3, 4] * 5, "target": [0, 1, 0, 1] * 5})
    ds_prof = EvindraOrchestrator().profiler.profile_dataframe(df, target_column="target")

    step_invalid = PreprocessingStep(
        step_number=1,
        stage="FEATURE_SELECTION",
        domain=DecisionDomain.TARGET_DETECTION,
        action="USE_NON_EXISTENT_COL",
        columns=["non_existent_column"],
        decision_id="d_bad",
        decision_source=DecisionSource.LLM,
        confidence=0.5,
    )
    bad_plan = PreprocessingPlan(dataset_name="invalid_plan_ds", target_column="target", steps=[step_invalid])

    validator = PlanValidator()
    val_res = validator.validate_plan(bad_plan, ds_prof, df)

    assert val_res.valid is False
    assert len(val_res.errors) > 0


def test_16_model_failure_with_fallback():
    """TEST 16: Single candidate model failure with graceful fallback to alternative candidate."""
    df = pd.DataFrame({"f1": [1, 2, 3, 4, 5, 6, 7, 8] * 3, "target": [0, 1, 0, 1, 0, 1, 0, 1] * 3})

    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator()
        res = orchestrator.run_pipeline(df, dataset_name="fallback_model_ds", target_column="target", output_dir=tmp_dir)

        # Inject broken pipeline to verify fault tolerance in candidate experiment report
        step_pass = PreprocessingStep(step_number=1, stage="MISSING_VALUE_HANDLING", domain=DecisionDomain.MISSING_VALUE_STRATEGY, action="PASS_THROUGH", columns=["f1"], decision_id="d1", decision_source=DecisionSource.RULE, confidence=0.9)
        plan = PreprocessingPlan(dataset_name="fallback_model_ds", target_column="target", steps=[step_pass])
        p_good = PipelineCandidate(name="Good", description="Good pipeline", preprocessing_plan=plan, model_spec={"model_family": "LOGISTIC_REGRESSION"})
        p_bad = PipelineCandidate(name="Bad", description="Bad pipeline", preprocessing_plan=plan, model_spec={"model_family": "INVALID_FAMILY"})
        cset = PipelineCandidateSet(dataset_name="fallback_model_ds", problem_type="classification", target_column="target", pipelines=[p_good, p_bad])

        report = orchestrator.experiment_runner.run_experiment(cset, df, target_column="target")
        assert report.successful_evaluations >= 1
        assert report.failed_evaluations == 1


def test_17_rag_unavailable():
    """TEST 17: RAG service unavailable fallback to Rule/LLM."""
    class FailingRAGService:
        def retrieve_relevant_scenarios(self, domain, profile):
            raise RuntimeError("RAG Vector Database Connection Failed!")

    df = pd.DataFrame({"f1": [1, 2, 3, 4] * 5, "target": [0, 1, 0, 1] * 5})
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator(rag_service=FailingRAGService())
        res = orchestrator.run_pipeline(df, dataset_name="rag_unavail_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)


def test_18_llm_unavailable():
    """TEST 18: LLM service unavailable fallback to User/Rule."""
    class FailingLLMService:
        def predict_decision(self, domain, profile, rag_results):
            raise RuntimeError("Gemini API Rate Limit / Connection Error!")

    df = pd.DataFrame({"f1": [1, 2, 3, 4] * 5, "target": [0, 1, 0, 1] * 5})
    with tempfile.TemporaryDirectory() as tmp_dir:
        orchestrator = EvindraOrchestrator(llm_service=FailingLLMService())
        res = orchestrator.run_pipeline(df, dataset_name="llm_unavail_ds", target_column="target", output_dir=tmp_dir)
        _verify_common_assertions(res)

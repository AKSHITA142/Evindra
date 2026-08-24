import os
import json
import tempfile
import pytest
import joblib
import pandas as pd

from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.schemas.decision import DecisionResult, DecisionDomain, DecisionSource
from backend.schemas.preprocessing_plan import PreprocessingStep, PreprocessingPlan
from backend.schemas.pipeline import PipelineCandidateSet, PipelineCandidate
from backend.schemas.experiment import ExperimentRunReport, PipelineEvaluationResult
from backend.schemas.best_pipeline import BestPipelineResult
from backend.schemas.holdout_validation import FinalValidationReport
from backend.engine.artifact_generator import ArtifactGenerator


def test_all_9_artifacts_generation_and_readability():
    """Verify all 9 output files exist, are non-empty, and readable."""
    df_proc = pd.DataFrame({"age": [25.0, 30.0], "income_scaled": [0.1, 0.2], "target": [0, 1]})

    col_age = ColumnProfileExtended(name="age", normalized_dtype="numeric")
    col_inc = ColumnProfileExtended(name="income", normalized_dtype="numeric")
    ds_prof = DatasetProfile(dataset_name="art_ds", rows=2, columns=2, detailed_column_profiles=[col_age, col_inc], target_column="target", problem_type="classification")

    d1 = DecisionResult(decision_id="dec_1", domain=DecisionDomain.MISSING_VALUE_STRATEGY, decision="IMPUTE_MEDIAN", confidence=0.95, reasoning="Numeric median imputation", evidence=["missing_ratio: 0.0"], source=DecisionSource.RULE)

    step1 = PreprocessingStep(step_number=1, stage="SCALING", domain=DecisionDomain.SCALING_TRANSFORMATION, action="STANDARD_SCALER", columns=["income"], decision_id="dec_1", decision_source=DecisionSource.RULE, confidence=0.95)
    plan1 = PreprocessingPlan(dataset_name="art_ds", target_column="target", steps=[step1])

    pipe1 = PipelineCandidate(name="Winner Pipe", description="Best candidate", preprocessing_plan=plan1, model_spec={"model_family": "LOGISTIC_REGRESSION"})
    cset = PipelineCandidateSet(dataset_name="art_ds", problem_type="classification", target_column="target", pipelines=[pipe1])

    eval1 = PipelineEvaluationResult(pipeline_id=pipe1.pipeline_id, pipeline_name="Winner Pipe", model_family="LOGISTIC_REGRESSION", status="SUCCESS", primary_metric="roc_auc", primary_score=0.92)
    exp_report = ExperimentRunReport(dataset_name="art_ds", problem_type="classification", primary_metric="roc_auc", best_pipeline_id=pipe1.pipeline_id, best_primary_score=0.92, evaluation_results=[eval1])

    best_res = BestPipelineResult(winner_pipeline_id=pipe1.pipeline_id, winner_pipeline_name="Winner Pipe", winner_model_family="LOGISTIC_REGRESSION", metric="roc_auc", score=0.92, winner_evaluation=eval1)

    hval_report = FinalValidationReport(pipeline_id=pipe1.pipeline_id, pipeline_name="Winner Pipe", model_family="LOGISTIC_REGRESSION", primary_metric="roc_auc", cv_score=0.92, holdout_score=0.91, difference=-0.01, generalization_assessment="GOOD")

    with tempfile.TemporaryDirectory() as tmp_dir:
        generator = ArtifactGenerator()
        artifact_paths = generator.generate_all_artifacts(
            output_dir=tmp_dir,
            df_processed=df_proc,
            dataset_profile=ds_prof,
            decisions=[d1],
            preprocessing_plan=plan1,
            candidate_set=cset,
            experiment_report=exp_report,
            best_result=best_res,
            holdout_report=hval_report,
        )

        expected_keys = [
            "final_processed.csv", "best_model", "preprocessing_pipeline",
            "feature_mapping.json", "decision_trace.json", "model_results.json",
            "final_validation.json", "final_report.json", "final_report.md"
        ]
        for key in expected_keys:
            assert key in artifact_paths
            file_path = artifact_paths[key]
            assert os.path.exists(file_path)
            assert os.path.getsize(file_path) > 0

        # Readability check for CSV
        df_read = pd.read_csv(artifact_paths["final_processed.csv"])
        assert len(df_read) == 2

        # Readability check for Joblib model
        model_obj = joblib.load(artifact_paths["best_model"])
        assert model_obj is not None

        # Readability check for JSON files
        with open(artifact_paths["feature_mapping.json"], "r") as f:
            mapping = json.load(f)
            assert isinstance(mapping, dict)

        with open(artifact_paths["decision_trace.json"], "r") as f:
            trace = json.load(f)
            assert len(trace) == 1

        with open(artifact_paths["final_report.md"], "r") as f:
            md_text = f.read()
            assert "# DataPilot AI" in md_text


def test_markdown_report_20_sections():
    """Verify final_report.md contains all 20 required sections."""
    df_proc = pd.DataFrame({"f1": [1.0], "target": [1]})
    ds_prof = DatasetProfile(dataset_name="md_test", rows=1, columns=1, detailed_column_profiles=[ColumnProfileExtended(name="f1", normalized_dtype="numeric")], target_column="target", problem_type="classification")
    d1 = DecisionResult(decision_id="d1", domain=DecisionDomain.MISSING_VALUE_STRATEGY, decision="IMPUTE_MEDIAN", confidence=0.90, reasoning="Rule median", evidence=["evidence_1"], source=DecisionSource.RULE)
    plan1 = PreprocessingPlan(dataset_name="md_test", target_column="target", steps=[])
    pipe1 = PipelineCandidate(name="Pipe", description="Test", preprocessing_plan=plan1)
    cset = PipelineCandidateSet(dataset_name="md_test", pipelines=[pipe1])
    eval1 = PipelineEvaluationResult(pipeline_id=pipe1.pipeline_id, status="SUCCESS")
    exp_report = ExperimentRunReport(dataset_name="md_test", evaluation_results=[eval1])
    best_res = BestPipelineResult(winner_pipeline_id=pipe1.pipeline_id, score=0.90)
    hval_report = FinalValidationReport(pipeline_id=pipe1.pipeline_id, cv_score=0.90, holdout_score=0.89)

    with tempfile.TemporaryDirectory() as tmp_dir:
        generator = ArtifactGenerator()
        artifact_paths = generator.generate_all_artifacts(
            output_dir=tmp_dir,
            df_processed=df_proc,
            dataset_profile=ds_prof,
            decisions=[d1],
            preprocessing_plan=plan1,
            candidate_set=cset,
            experiment_report=exp_report,
            best_result=best_res,
            holdout_report=hval_report,
        )

        with open(artifact_paths["final_report.md"], "r") as f:
            md_text = f.read()

        for sec in range(1, 21):
            assert f"## {sec}." in md_text

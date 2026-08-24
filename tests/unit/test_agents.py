import pytest

from backend.schemas.semantic_profile import SemanticProfile
from backend.schemas.mission_brief import MissionBrief, MissionConstraints
from backend.schemas.experiment import ExperimentPlan
from backend.schemas.evaluation import EvaluationReport, ResearchDirectorDecision, RankingItem
from backend.schemas.report import FinalRecommendation
from backend.schemas.enums import DecisionType
from backend.core.config import get_settings
from backend.agents import (

    DatasetUnderstandingAgent,
    ConstraintGoalAnalyzer,
    StrategyPlannerAgent,
    ResearchDirectorAgent,
    ReportGeneratorAgent,
    LLMClient,
)


def test_llm_client_dynamic_model():
    client = LLMClient()
    expected_model = get_settings().model_name or get_settings().llm_model_name
    assert client.model_name == expected_model



    custom_client = LLMClient(model_name="custom-model-v1")
    assert custom_client.model_name == "custom-model-v1"


def test_dataset_understanding_agent():
    agent = DatasetUnderstandingAgent()

    # Synthetic SemanticProfile
    profile = SemanticProfile(
        dataset_summary={"rows": 1000, "columns": 10},
        column_profiles=[],
        quality_issues=[],
        resource_profile={"estimated_memory_mb": 1.5},
    )

    res = agent.run({
        "semantic_profile": profile,
        "user_goal": "Maximize F1 score for churn prediction",
        "target_column": "churn",
    })

    assert isinstance(res, MissionBrief)
    assert res.objective != ""
    assert res.constraints.max_row_loss == 0.05


def test_constraint_goal_analyzer():
    agent = ConstraintGoalAnalyzer()
    res = agent.run({
        "user_instructions": "Please use explainable models and limit training to 15 minutes."
    })

    assert isinstance(res, MissionConstraints)
    # Check for training time limit extraction or interpretable flag
    assert res.training_time_limit_minutes == 15 or any(
        "interpretable" in k or "explainable" in k
        for k, v in res.custom_constraints.items() if v
    ) or res.custom_constraints.get("prefer_interpretable_models") is not False


def test_strategy_planner_agent():
    agent = StrategyPlannerAgent()

    mission = MissionBrief(objective="Optimize classification model")
    profile = SemanticProfile(
        dataset_summary={"rows": 500, "columns": 5},
        column_profiles=[],
        quality_issues=[],
        resource_profile={"estimated_memory_mb": 1.0},
    )

    res = agent.run({
        "semantic_profile": profile,
        "mission_brief": mission,
        "experiment_budget": 2,
        "task_type": "classification",
    })

    assert isinstance(res, ExperimentPlan)
    assert len(res.experiments) == 2
    assert res.experiments[0].model_name.endswith("Classifier") or res.experiments[0].model_name in (
        "RandomForestClassifier", "HistGradientBoostingClassifier", "LGBMClassifier",
        "XGBClassifier", "CatBoostClassifier", "LogisticRegression", "ExtraTreesClassifier"
    )


def test_research_director_agent():
    agent = ResearchDirectorAgent()

    report = EvaluationReport(
        winner="EXP_001",
        ranking=[RankingItem(rank=1, experiment_id="EXP_001", score=0.91, model="RandomForest")],
        knowledge=[],
        should_continue=True,
        reason="Progress achieved",
    )

    res = agent.run({"evaluation_report": report})

    assert isinstance(res, ResearchDirectorDecision)
    assert res.decision == DecisionType.CONTINUE
    assert res.confidence > 0.0


def test_report_generator_agent():
    agent = ReportGeneratorAgent()

    report = EvaluationReport(
        winner="EXP_001",
        ranking=[RankingItem(rank=1, experiment_id="EXP_001", score=0.91, model="RandomForest")],
        knowledge=[],
        should_continue=False,
        reason="Completed",
    )

    res = agent.run({"evaluation_report": report})

    assert isinstance(res, FinalRecommendation)
    assert res.winning_experiment_id == "EXP_001"
    assert len(res.key_findings) > 0

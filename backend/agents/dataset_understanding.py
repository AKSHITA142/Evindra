from typing import Any, Dict, Optional, Type
from pydantic import BaseModel

from backend.schemas.semantic_profile import SemanticProfile
from backend.schemas.mission_brief import MissionBrief, DatasetCharacteristics, MissionConstraints
from backend.agents.base import BaseAgent


class DatasetUnderstandingAgent(BaseAgent):
    """Reasoning agent that converts dataset profile and user goal into a MissionBrief."""

    @property
    def name(self) -> str:
        return "Dataset Understanding Agent"

    @property
    def response_model(self) -> Type[BaseModel]:
        return MissionBrief

    def format_prompt(self, inputs: Dict[str, Any]) -> str:
        profile: Optional[SemanticProfile] = inputs.get("semantic_profile")
        user_goal: str = inputs.get("user_goal", "Optimize machine learning model performance.")
        target_col: str = inputs.get("target_column", "target")
        task_type: str = inputs.get("task_type", "classification")

        rows = profile.dataset_summary.get("rows", 0) if profile else 0
        cols = profile.dataset_summary.get("columns", 0) if profile else 0

        if task_type == "regression":
            metric_guidance = "Select regression success metrics ONLY (r2, rmse, mae, explained_variance)."
        else:
            metric_guidance = "Select classification success metrics ONLY (f1, accuracy, precision, recall, roc_auc)."

        return (
            f"Analyze the dataset profile and user goal to form a MissionBrief:\n"
            f"User Goal: {user_goal}\n"
            f"Target Column: {target_col}\n"
            f"Strict Task Type: {task_type.upper()}\n"
            f"Dataset Size: {rows} rows, {cols} columns.\n"
            f"{metric_guidance}\n"
            f"Formulate domain, constraints, and appropriate success metrics for {task_type}."
        )

    def get_fallback_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        user_goal = inputs.get("user_goal", "Maximize model predictive accuracy")
        task_type = inputs.get("task_type", "classification")

        metrics = ["r2", "rmse", "mae"] if task_type == "regression" else ["f1", "accuracy", "roc_auc"]

        return {
            "objective": user_goal,
            "constraints": {
                "max_row_loss": 0.05,
                "use_only_open_source_models": True,
                "training_time_limit_minutes": 30,
                "forbidden_operations": [],
                "custom_constraints": {},
            },
            "dataset_characteristics": {
                "domain": "General Tabular Data",
                "risk_level": "Low",
                "complexity": "Medium",
            },
            "success_metrics": metrics,
            "avoid": ["data_leakage", "severe_overfitting"],
        }

    def run(self, inputs: Dict[str, Any]) -> MissionBrief:
        brief: MissionBrief = super().run(inputs)
        task_type = inputs.get("task_type", "classification")

        # Post-process validation: enforce metrics strictly match task_type
        if task_type == "regression":
            reg_allowed = {"r2", "rmse", "mae", "explained_variance", "mean_absolute_error", "root_mean_squared_error"}
            filtered = [m for m in brief.success_metrics if m.lower() in reg_allowed]
            brief.success_metrics = filtered if filtered else ["r2", "rmse", "mae"]
        else:
            clf_allowed = {"f1", "accuracy", "precision", "recall", "roc_auc", "balanced_accuracy"}
            filtered = [m for m in brief.success_metrics if m.lower() in clf_allowed]
            brief.success_metrics = filtered if filtered else ["f1", "accuracy", "roc_auc"]

        return brief

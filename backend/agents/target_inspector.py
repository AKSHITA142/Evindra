"""Reasoning agent for zero-shot LLM semantic target column inspection."""

from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, Field
from backend.agents.base import BaseAgent


class ColumnTargetScore(BaseModel):
    """Semantic confidence score for a candidate target column."""
    column_name: str
    confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""


class TargetInspectionResult(BaseModel):
    """Result of zero-shot target column semantic inspection."""
    recommended_target: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    column_scores: List[ColumnTargetScore] = Field(default_factory=list)


class TargetInspectorAgent(BaseAgent):
    """Reasoning agent that inspects quantile-sampled rows and column signatures

    to identify the dependent target variable using domain semantics.
    """

    @property
    def name(self) -> str:
        return "Target Inspector Agent"

    @property
    def response_model(self) -> Type[BaseModel]:
        return TargetInspectionResult

    def format_prompt(self, inputs: Dict[str, Any]) -> str:
        columns: List[str] = inputs.get("columns", [])
        sample_rows: List[Dict[str, Any]] = inputs.get("sample_rows", [])
        col_types: Dict[str, str] = inputs.get("column_types", {})
        user_mission: str = inputs.get("user_mission", "")

        return (
            f"Analyze the dataset columns and representative sample rows to identify the primary target variable to predict.\n\n"
            f"User Research Mission: '{user_mission or 'Identify predictive target'}'\n"
            f"Columns and Types: {col_types}\n"
            f"Quantile Sample Rows: {sample_rows[:3]}\n\n"
            f"Evaluate domain semantics and select the single column that represents the target/dependent variable. "
            f"Do NOT select metadata or ID columns (id, uuid, name, index)."
        )

    def get_fallback_data(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        columns: List[str] = inputs.get("columns", [])
        user_mission: str = inputs.get("user_mission", "").lower()

        target = columns[-1] if columns else "target"
        # Check if last column or common targets match
        common = ["income", "price", "churn", "target", "label", "outcome", "status", "salary"]
        for c in columns:
            c_clean = c.lower().strip()
            if c_clean in common or c_clean in user_mission:
                target = c
                break

        return {
            "recommended_target": target,
            "confidence": 0.85,
            "column_scores": [
                {"column_name": target, "confidence_score": 0.85, "reasoning": "Highest domain relevance."}
            ],
        }

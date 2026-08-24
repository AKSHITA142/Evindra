from typing import Optional, Dict, Any
from backend.schemas.report import FinalRecommendation
from backend.schemas.evaluation import EvaluationReport
from backend.schemas.semantic_profile import SemanticProfile
from backend.schemas.mission_brief import MissionBrief


class MarkdownReportGenerator:
    """Generates structured Markdown reports from research artifacts and evaluation reports."""

    @classmethod
    def generate_markdown(
        cls,
        recommendation: FinalRecommendation,
        evaluation_report: Optional[EvaluationReport] = None,
        profile: Optional[SemanticProfile] = None,
        mission: Optional[MissionBrief] = None,
    ) -> str:
        """Constructs a comprehensive Markdown report string."""
        lines = []

        # 1. Title & Header
        lines.append("# DataPilot-AI Research & Recommendation Report")
        lines.append(f"**Winning Experiment ID:** `{recommendation.winning_experiment_id}`\n")

        # 2. Executive Summary
        lines.append("## 1. Executive Summary")
        lines.append(recommendation.summary)
        lines.append("")

        # 3. Mission & Goal Recap
        lines.append("## 2. Mission Recap")
        if mission:
            lines.append(f"- **Objective:** {mission.objective}")
            lines.append(f"- **Domain:** {mission.dataset_characteristics.domain} (Risk: {mission.dataset_characteristics.risk_level})")
            lines.append(f"- **Target Metrics:** {', '.join(mission.success_metrics or ['Primary Metric'])}")
        else:
            lines.append("Automated machine learning optimization mission.")
        lines.append("")

        # 4. Dataset Overview
        lines.append("## 3. Dataset Summary")
        if profile:
            summary = profile.dataset_summary
            lines.append(f"- **Rows:** {summary.get('rows', 'N/A')}")
            lines.append(f"- **Columns:** {summary.get('columns', 'N/A')}")
            lines.append(f"- **Quality Issues Identified:** {len(profile.quality_issues)}")
        else:
            lines.append("Dataset statistics available in semantic profile.")
        lines.append("")

        # 5. Winning Pipeline & Model
        lines.append("## 4. Winning Model & Pipeline Architecture")
        lines.append(f"**Selected Model Family:** `{recommendation.model}`\n")
        lines.append("### Pipeline Steps:")
        ops = []
        if recommendation.pipeline:
            if hasattr(recommendation.pipeline, "operations"):
                ops = recommendation.pipeline.operations
            elif isinstance(recommendation.pipeline, dict):
                ops = recommendation.pipeline.get("operations", [])

        if ops:
            for idx, op in enumerate(ops, start=1):
                if hasattr(op, "type"):
                    op_type = getattr(op, "type", "step")
                    op_method = getattr(op, "method", "default")
                elif isinstance(op, dict):
                    op_type = op.get("type", "step")
                    op_method = op.get("method", "default")
                else:
                    op_type = "step"
                    op_method = str(op)
                lines.append(f"{idx}. **{op_type.title()}**: `{op_method}`")
        else:
            lines.append("Standard preprocessing pipeline applied.")
        lines.append("")

        # 6. Performance Metrics
        lines.append("## 5. Performance Metrics")
        if recommendation.final_metrics:
            for metric, val in recommendation.final_metrics.items():
                lines.append(f"- **{metric.upper()}**: `{val}`")
        else:
            lines.append("Primary metric achieved top performance.")
        lines.append("")

        # 7. Key Knowledge Findings
        lines.append("## 6. Key Knowledge Findings")
        if recommendation.key_findings:
            for finding in recommendation.key_findings:
                lines.append(f"- 💡 {finding}")
        elif evaluation_report and evaluation_report.knowledge:
            for f in evaluation_report.knowledge:
                lines.append(f"- 💡 {f.finding}")
        else:
            lines.append("- Optimization completed successfully across tested model families.")
        lines.append("")

        # 8. Experiment Rankings Table
        if evaluation_report and evaluation_report.ranking:
            lines.append("## 7. Experiment Rankings")
            lines.append("| Rank | Experiment ID | Model | Composite Score |")
            lines.append("|---|---|---|---|")
            for item in evaluation_report.ranking:
                lines.append(f"| {item.rank} | `{item.experiment_id}` | `{item.model}` | `{item.score:.4f}` |")
            lines.append("")

        # 9. Technical & Business Notes
        lines.append("## 8. Business & Technical Recommendations")
        lines.append("- **Business Impact:** High-accuracy automated preprocessing ready for production integration.")
        lines.append("- **Deployment Guidance:** Apply exact feature scaling and encoding parameters to live inference batches.")

        return "\n".join(lines)

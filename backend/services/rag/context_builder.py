import logging
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional

logger = logging.getLogger("datapilot.rag.context_builder")


@dataclass
class RAGEvidencePackage:
    """
    Structured evidence package containing formatted dataset facts, historical scenario evidence,
    and a clean prompt context string for LLM consumption.
    """
    dataset_summary: Dict[str, Any]
    evidence_items: List[Dict[str, Any]]
    total_evidence_count: int
    prompt_context_str: str

    def to_dict(self) -> Dict[str, Any]:
        """Converts evidence package to dictionary format."""
        return asdict(self)


class RAGContextBuilder:
    """
    RAG Context Builder Service for Evindra RAG System (Phase C).
    Transforms current dataset profile information and reranked historical scenarios into
    a compact, structured evidence package and formatted prompt context for LLM decision-making.

    Maintains explicit scenario ID traceability and separates dataset facts from historical evidence.
    """

    def build_evidence_package(
        self,
        dataset_profile: Dict[str, Any],
        retrieved_scenarios: List[Dict[str, Any]],
        max_evidence_count: Optional[int] = None,
    ) -> RAGEvidencePackage:
        """
        Builds a structured RAGEvidencePackage from a dataset profile and retrieved scenarios.

        Args:
            dataset_profile: Dictionary containing current dataset properties:
                - dataset_name: str
                - target_column: Optional[str]
                - problem_type: str (e.g. "regression", "binary_classification")
                - target_feature: Optional[str] (column under investigation)
                - feature_dtype: Optional[str]
                - issue_description: Optional[str]
            retrieved_scenarios: List of reranked candidate scenario dicts from ScenarioRerankerService.
            max_evidence_count: Optional limit on the number of evidence scenarios to include.

        Returns:
            RAGEvidencePackage object containing structured evidence and prompt string.
        """
        scenarios = retrieved_scenarios or []
        if max_evidence_count and max_evidence_count > 0:
            scenarios = scenarios[:max_evidence_count]

        # Step 1: Clean dataset summary
        dataset_summary = {
            "dataset_name": dataset_profile.get("dataset_name", "uploaded_dataset.csv"),
            "target_column": dataset_profile.get("target_column"),
            "problem_type": dataset_profile.get("problem_type", "general_tabular"),
            "target_feature": dataset_profile.get("target_feature") or dataset_profile.get("column"),
            "feature_dtype": dataset_profile.get("feature_dtype") or dataset_profile.get("column_type"),
            "issue_description": dataset_profile.get("issue_description", "Data preprocessing analysis."),
        }

        # Step 2: Extract structured evidence items with traceability
        evidence_items: List[Dict[str, Any]] = []
        for i, sc in enumerate(scenarios, 1):
            metadata = sc.get("metadata", {}) or {}
            ans_key = metadata.get("answer_key", {}) or {}
            val_info = metadata.get("validation", {}) or {}
            final_dec = metadata.get("final_decision", {}) or {}

            # Determine historical decision & action
            decision = (
                final_dec.get("recommended_role")
                or final_dec.get("imbalance_severity")
                or ans_key.get("selected_option")
                or ans_key.get("classification")
                or "VALIDATED_STRATEGY"
            )
            action = (
                ans_key.get("recommended_action")
                or metadata.get("action")
                or "APPLY_RECOMMENDED_TRANSFORMATION"
            )
            rationale = (
                ans_key.get("rationale")
                or metadata.get("rationale")
                or "Validated historical benchmark scenario."
            )
            val_status = val_info.get("status", "RULE_VALIDATED")
            final_score = float(sc.get("final_score", sc.get("semantic_score", 0.0)))
            explanation = sc.get("rank_explanation", f"Relevant scenario with similarity {final_score:.4f}.")

            evidence_items.append({
                "rank": i,
                "scenario_id": sc.get("scenario_id", f"SCENARIO_{i:03d}"),
                "domain": sc.get("domain", "general_tabular"),
                "scenario_type": sc.get("scenario_type", "preprocessing"),
                "relevance_score": round(final_score, 4),
                "semantic_score": round(float(sc.get("semantic_score", 0.0)), 4),
                "structured_score": round(float(sc.get("structured_score", 0.5)), 4),
                "rank_explanation": explanation,
                "historical_decision": str(decision),
                "recommended_action": str(action),
                "historical_rationale": str(rationale),
                "validation_status": str(val_status),
                "retrieval_text": sc.get("retrieval_text", ""),
            })

        # Step 3: Format clean prompt context string for LLM
        prompt_str = self._format_prompt_context(dataset_summary, evidence_items)

        logger.info(
            f"Built RAGEvidencePackage for '{dataset_summary['dataset_name']}' with {len(evidence_items)} historical evidence scenarios."
        )

        return RAGEvidencePackage(
            dataset_summary=dataset_summary,
            evidence_items=evidence_items,
            total_evidence_count=len(evidence_items),
            prompt_context_str=prompt_str,
        )

    def _format_prompt_context(
        self, summary: Dict[str, Any], evidence_items: List[Dict[str, Any]]
    ) -> str:
        """Formats dataset summary and historical evidence items into a clean markdown string."""
        lines = [
            "==================================================",
            "  SECTION 1: CURRENT DATASET FACTS & CONTEXT",
            "==================================================",
            f"- Dataset Name: {summary['dataset_name']}",
            f"- Problem Type: {summary['problem_type']}",
        ]
        if summary.get("target_column"):
            lines.append(f"- Target Column: {summary['target_column']}")
        if summary.get("target_feature"):
            lines.append(f"- Column Under Investigation: {summary['target_feature']} (dtype: {summary.get('feature_dtype', 'unknown')})")
        lines.append(f"- Issue Description: {summary['issue_description']}")
        lines.append("")

        lines.extend([
            "==================================================",
            "  SECTION 2: HISTORICAL EVIDENTIAL SCENARIOS (RANKED)",
            "==================================================",
            "The following historically validated scenarios from the Evindra knowledge base represent relevant empirical evidence for this situation:",
            ""
        ])

        if not evidence_items:
            lines.append("No historical evidence scenarios found for this query.")
            return "\n".join(lines)

        for item in evidence_items:
            lines.append(f"### [Evidence #{item['rank']}] Scenario ID: {item['scenario_id']}")
            lines.append(f"- Domain: {item['domain']} | Scenario Type: {item['scenario_type']}")
            lines.append(f"- Relevance Score: {item['relevance_score']:.4f} (Semantic: {item['semantic_score']:.4f}, Structured: {item['structured_score']:.4f})")
            lines.append(f"- Selection Rationale: {item['rank_explanation']}")
            lines.append(f"- Historical Validated Decision: {item['historical_decision']}")
            lines.append(f"- Recommended Action: {item['recommended_action']}")
            lines.append(f"- Evidence Rationale: {item['historical_rationale']}")
            lines.append(f"- Validation Status: {item['validation_status']}")
            lines.append("")

        return "\n".join(lines)


# Convenience function for direct module usage
def build_rag_evidence_package(
    dataset_profile: Dict[str, Any],
    retrieved_scenarios: List[Dict[str, Any]],
    max_evidence_count: Optional[int] = None,
) -> RAGEvidencePackage:
    """Convenience wrapper around RAGContextBuilder.build_evidence_package."""
    builder = RAGContextBuilder()
    return builder.build_evidence_package(dataset_profile, retrieved_scenarios, max_evidence_count=max_evidence_count)

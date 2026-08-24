import logging
from typing import Optional, Dict, Any, List, Tuple

from backend.schemas.decision import (
    DecisionDomain,
    DecisionSource,
    ValidationStatus,
    DecisionRequest,
    DecisionResult,
    UserFallbackRequest,
    UserFallbackResponse,
)
from backend.schemas.dataset_profile import DatasetProfile, ColumnProfileExtended
from backend.core.confidence_policy import ConfidencePolicy, DEFAULT_CONFIDENCE_POLICY
from backend.engine.rule_engine import RuleEngine
from backend.engine.user_fallback import UserFallbackHandler
from backend.services.rag.hybrid_retrieval_service import HybridRetrievalService, retrieve_relevant_scenarios
from backend.services.rag.reranker_service import rerank_scenarios
from backend.services.rag.decision_service import LLMDecisionService
from backend.services.rag.context_builder import build_rag_evidence_package, RAGContextBuilder

logger = logging.getLogger("datapilot.engine.orchestrator")


class DecisionOrchestrator:
    """
    Core Decision Orchestrator enforcing the Evindra Decision Hierarchy:
    RULE -> RAG -> LLM -> USER.

    Evaluates RuleEngine first. If rule confidence is high, returns immediately.
    If rule confidence is low/uncertain, escalates to domain-aware RAG retrieval.
    If RAG confidence is low/uncertain, escalates to LLM Decision Service.
    If LLM confidence is low/uncertain or unavailable, escalates to User Fallback.
    """

    def __init__(
        self,
        rule_engine: Optional[RuleEngine] = None,
        confidence_policy: Optional[ConfidencePolicy] = None,
        hybrid_retrieval_service: Optional[HybridRetrievalService] = None,
        llm_decision_service: Optional[LLMDecisionService] = None,
        user_fallback_handler: Optional[UserFallbackHandler] = None,
    ):
        self.policy = confidence_policy or DEFAULT_CONFIDENCE_POLICY
        self.rule_engine = rule_engine or RuleEngine(confidence_policy=self.policy)
        self.retrieval_service = hybrid_retrieval_service
        self.llm_service = llm_decision_service
        self.user_fallback_handler = user_fallback_handler or UserFallbackHandler()

    def get_retrieval_service(self) -> Optional[HybridRetrievalService]:
        """Lazy loads HybridRetrievalService gracefully."""
        if self.retrieval_service is None:
            try:
                self.retrieval_service = HybridRetrievalService()
            except Exception as e:
                logger.warning(f"RAG HybridRetrievalService unavailable: {e}")
                self.retrieval_service = None
        return self.retrieval_service

    def get_llm_service(self) -> Optional[LLMDecisionService]:
        """Lazy loads LLMDecisionService gracefully."""
        if self.llm_service is None:
            try:
                self.llm_service = LLMDecisionService()
            except Exception as e:
                logger.warning(f"LLMDecisionService unavailable: {e}")
                self.llm_service = None
        return self.llm_service

    def evaluate_decision(
        self,
        domain: DecisionDomain,
        col_profile: Optional[ColumnProfileExtended] = None,
        dataset_profile: Optional[DatasetProfile] = None,
        model_family: str = "general",
        user_mission: str = "",
        user_response: Optional[UserFallbackResponse] = None,
        user_selection: Optional[str] = None,
        auto_approve_default: bool = True,
    ) -> DecisionResult:
        """
        Executes Rule -> RAG -> LLM -> USER fallback hierarchy sequence for a decision request.
        """
        # Step 1: Rule Engine Evaluation
        rule_result = self._evaluate_rule_engine(
            domain, col_profile=col_profile, dataset_profile=dataset_profile, model_family=model_family
        )

        # Step 2: High Confidence Rule Check
        if self.policy.is_rule_high_confidence(domain, rule_result.confidence):
            logger.info(f"Rule Engine high confidence ({rule_result.confidence:.2f}) for domain '{domain.value}'. Accepting rule decision.")
            return rule_result

        logger.info(
            f"Rule Engine low confidence ({rule_result.confidence:.2f}) for domain '{domain.value}'. Escalating to RAG retrieval..."
        )

        # Step 3: Domain-aware RAG Escalation
        rag_result, retrieved_scenarios = self.query_rag_fallback_with_scenarios(
            domain=domain,
            rule_result=rule_result,
            col_profile=col_profile,
            dataset_profile=dataset_profile,
            user_mission=user_mission,
        )

        if rag_result is not None and self.policy.is_rag_high_confidence(domain, rag_result.confidence):
            logger.info(f"RAG high confidence ({rag_result.confidence:.2f}) for domain '{domain.value}'. Accepting RAG decision.")
            return rag_result

        logger.info(
            f"RAG low confidence or unavailable for domain '{domain.value}'. Escalating to LLM Decision Service..."
        )

        # Step 4: RAG -> LLM Escalation
        llm_result = self.query_llm_fallback(
            domain=domain,
            rule_result=rule_result,
            retrieved_scenarios=retrieved_scenarios,
            col_profile=col_profile,
            dataset_profile=dataset_profile,
            user_mission=user_mission,
        )

        if llm_result is not None and self.policy.is_llm_high_confidence(domain, llm_result.confidence):
            logger.info(f"LLM high confidence ({llm_result.confidence:.2f}) for domain '{domain.value}'. Accepting LLM decision.")
            return llm_result

        # Step 5: Escalation to User Fallback Handler (Rule, RAG, and LLM confidence below thresholds)
        logger.info(f"Rule, RAG, and LLM confidence below thresholds for domain '{domain.value}'. Escalating to User Fallback...")
        preceding_result = llm_result if llm_result is not None else rule_result

        return self.user_fallback_handler.resolve_user_fallback(
            decision_result=preceding_result,
            user_response=user_response,
            user_selection=user_selection,
            auto_approve_default=auto_approve_default,
            column_name=col_profile.name if col_profile else None,
        )

    def _evaluate_rule_engine(
        self,
        domain: DecisionDomain,
        col_profile: Optional[ColumnProfileExtended] = None,
        dataset_profile: Optional[DatasetProfile] = None,
        model_family: str = "general",
    ) -> DecisionResult:
        """Invokes appropriate RuleEngine method based on domain."""
        if domain == DecisionDomain.MISSING_VALUE_STRATEGY and col_profile:
            return self.rule_engine.evaluate_missing_value_strategy(col_profile)
        elif domain == DecisionDomain.ENCODING_STRATEGY and col_profile:
            return self.rule_engine.evaluate_encoding_strategy(col_profile)
        elif domain == DecisionDomain.SCALING_TRANSFORMATION and col_profile:
            return self.rule_engine.evaluate_scaling_transformation(col_profile, model_family=model_family)
        elif domain == DecisionDomain.OUTLIER_HANDLING and col_profile:
            return self.rule_engine.evaluate_outlier_handling(col_profile, model_family=model_family)
        elif domain == DecisionDomain.FEATURE_SELECTION and dataset_profile:
            return self.rule_engine.evaluate_feature_selection(dataset_profile)

        # Default fallback rule result
        return DecisionResult(
            domain=domain,
            decision="RULE_DEFER",
            confidence=0.50,
            reasoning=f"Rule engine deferred decision for domain '{domain.value}'.",
            source=DecisionSource.RULE,
            requires_validation=True,
        )

    def query_rag_fallback(
        self,
        domain: DecisionDomain,
        rule_result: DecisionResult,
        col_profile: Optional[ColumnProfileExtended] = None,
        dataset_profile: Optional[DatasetProfile] = None,
        user_mission: str = "",
    ) -> Optional[DecisionResult]:
        """Wrapper method for RAG fallback returning decision result."""
        res, _ = self.query_rag_fallback_with_scenarios(
            domain=domain,
            rule_result=rule_result,
            col_profile=col_profile,
            dataset_profile=dataset_profile,
            user_mission=user_mission,
        )
        return res

    def query_rag_fallback_with_scenarios(
        self,
        domain: DecisionDomain,
        rule_result: DecisionResult,
        col_profile: Optional[ColumnProfileExtended] = None,
        dataset_profile: Optional[DatasetProfile] = None,
        user_mission: str = "",
    ) -> Tuple[Optional[DecisionResult], List[Dict[str, Any]]]:
        """
        Executes domain-aware RAG scenario retrieval to resolve rule uncertainty.
        Returns (DecisionResult or None, List of retrieved scenarios).
        """
        retrieval_svc = self.get_retrieval_service()
        if not retrieval_svc:
            logger.warning("RAG retrieval service not active. Continuing to LLM.")
            return None, []

        col_name = col_profile.name if col_profile else "dataset"
        col_type = col_profile.normalized_dtype if col_profile else "unknown"

        query_text = (
            f"Preprocessing decision for domain '{domain.value}' on column '{col_name}' "
            f"with type '{col_type}'. Rule result was '{rule_result.decision}' with confidence {rule_result.confidence:.2f}. {user_mission}"
        )

        context_filter = {
            "domain": domain.value,
            "column_type": col_type,
        }

        try:
            scenarios = retrieval_svc.retrieve_relevant_scenarios(
                query_text=query_text,
                context=context_filter,
                top_k=3,
            )

            if not scenarios:
                logger.info(f"No relevant RAG scenarios found for domain '{domain.value}'.")
                return None, []

            reranked = rerank_scenarios(scenarios)
            top_scenario = reranked[0] if reranked else scenarios[0]

            rag_score = float(top_scenario.get("final_score") or top_scenario.get("similarity_score") or 0.0)

            best_rec = (
                top_scenario.get("metadata", {}).get("recommendation")
                or top_scenario.get("scenario_type")
                or rule_result.decision
            )
            scen_id = top_scenario.get("scenario_id", "unknown")

            rag_res = DecisionResult(
                domain=domain,
                decision=best_rec,
                confidence=round(rag_score, 4),
                reasoning=f"RAG retrieval found aligned scenario ({scen_id}) with similarity {rag_score:.2f}.",
                evidence=[top_scenario],
                source=DecisionSource.RAG,
                requires_validation=True,
                metadata={"rule_result": rule_result.to_dict(), "scenario_id": scen_id},
            )

            return rag_res, reranked

        except Exception as e:
            logger.warning(f"RAG retrieval query encountered error: {e}. Gracefully falling back.")
            return None, []

    def query_llm_fallback(
        self,
        domain: DecisionDomain,
        rule_result: DecisionResult,
        retrieved_scenarios: List[Dict[str, Any]],
        col_profile: Optional[ColumnProfileExtended] = None,
        dataset_profile: Optional[DatasetProfile] = None,
        user_mission: str = "",
    ) -> Optional[DecisionResult]:
        """
        Executes LLM fallback when Rule and RAG confidence are low/uncertain.
        Returns a structured DecisionResult with source=DecisionSource.LLM.
        """
        llm_svc = self.get_llm_service()
        if not llm_svc:
            logger.warning("LLM service unavailable for fallback.")
            return None

        col_name = col_profile.name if col_profile else "dataset"

        profile_dict = dataset_profile.to_dict() if dataset_profile else {
            "column": col_name,
            "missing_ratio": col_profile.missing_ratio if col_profile else 0.0,
            "normalized_dtype": col_profile.normalized_dtype if col_profile else "unknown",
            "user_mission": user_mission,
        }

        # Build evidence package for LLM
        evidence_package = build_rag_evidence_package(
            dataset_profile=profile_dict,
            retrieved_scenarios=retrieved_scenarios,
        )

        try:
            llm_rec = llm_svc.generate_preprocessing_recommendation(
                dataset_profile=profile_dict,
                evidence_package=evidence_package,
            )

            is_high_conf = self.policy.is_llm_high_confidence(domain, llm_rec.confidence_score)
            should_user = self.policy.should_escalate_llm(domain, llm_rec.confidence_score)

            warnings_list = list(llm_rec.risk_analysis or [])
            if should_user:
                warnings_list.append(f"LLM confidence ({llm_rec.confidence_score:.2f}) below threshold. Escalating to User Fallback.")

            return DecisionResult(
                domain=domain,
                decision=llm_rec.primary_recommendation,
                confidence=round(llm_rec.confidence_score, 4),
                reasoning=llm_rec.reasoning,
                evidence=llm_rec.evidence_scenarios,
                alternatives=llm_rec.alternative_strategies,
                source=DecisionSource.LLM,
                requires_validation=True,
                warnings=warnings_list,
                metadata={
                    "rule_result": rule_result.to_dict(),
                    "escalate_to_user": should_user,
                },
            )

        except Exception as e:
            logger.warning(f"LLM decision service encountered error: {e}. Gracefully falling back.")
            return None

    def orchestrate_decisions(
        self,
        dataset_profile: DatasetProfile,
        user_responses: Optional[Dict[str, Any]] = None,
    ) -> List[DecisionResult]:
        """
        Runs complete decision hierarchy across all decision domains for a DatasetProfile.
        """
        user_resp_map = user_responses or {}
        decisions: List[DecisionResult] = []

        all_cols = getattr(dataset_profile, "detailed_column_profiles", []) or getattr(dataset_profile, "column_profiles", [])
        cat_cols = [c for c in all_cols if getattr(c, "normalized_dtype", None) in ("categorical", "string", "category")]
        num_cols = [c for c in all_cols if getattr(c, "normalized_dtype", None) in ("numeric", "integer", "float")]

        # 1. Missing Value Strategy per column
        for col_prof in all_cols:
            if col_prof.name == dataset_profile.target_column:
                continue
            user_sel = user_resp_map.get(col_prof.name) or user_resp_map.get("missing_value_strategy")
            res = self.evaluate_decision(
                domain=DecisionDomain.MISSING_VALUE_STRATEGY,
                col_profile=col_prof,
                dataset_profile=dataset_profile,
                user_selection=user_sel,
            )
            decisions.append(res)

        # 2. Categorical Encoding Strategy
        for col_prof in cat_cols:
            if col_prof.name == dataset_profile.target_column:
                continue
            user_sel = user_resp_map.get(col_prof.name) or user_resp_map.get("encoding_strategy")
            res = self.evaluate_decision(
                domain=DecisionDomain.ENCODING_STRATEGY,
                col_profile=col_prof,
                dataset_profile=dataset_profile,
                user_selection=user_sel,
            )
            decisions.append(res)

        # 3. Scaling Transformation
        for col_prof in num_cols:
            if col_prof.name == dataset_profile.target_column:
                continue
            user_sel = user_resp_map.get(col_prof.name) or user_resp_map.get("scaling_transformation")
            res = self.evaluate_decision(
                domain=DecisionDomain.SCALING_TRANSFORMATION,
                col_profile=col_prof,
                dataset_profile=dataset_profile,
                user_selection=user_sel,
            )
            decisions.append(res)

        # 4. Outlier Strategy
        for col_prof in num_cols:
            if col_prof.name == dataset_profile.target_column:
                continue
            user_sel = user_resp_map.get(col_prof.name) or user_resp_map.get("outlier_handling")
            res = self.evaluate_decision(
                domain=DecisionDomain.OUTLIER_HANDLING,
                col_profile=col_prof,
                dataset_profile=dataset_profile,
                user_selection=user_sel,
            )
            decisions.append(res)

        return decisions


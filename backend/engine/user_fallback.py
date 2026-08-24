import logging
from typing import Optional, Dict, Any, List

from backend.schemas.decision import (
    DecisionDomain,
    DecisionSource,
    DecisionResult,
    UserFallbackRequest,
    UserFallbackResponse,
)

logger = logging.getLogger("datapilot.engine.user_fallback")


class UserFallbackHandler:
    """
    User Fallback Handler for Evindra Decision Engine (Phase 7).
    Handles final escalation when Rule, RAG, and LLM confidence scores are below thresholds.

    - Constructs structured UserFallbackRequest.
    - Processes human choices / overrides or auto-approves safe defaults in non-interactive execution mode.
    - Wraps final choice in DecisionResult with source=DecisionSource.USER.
    """

    def create_fallback_request(
        self,
        decision_result: DecisionResult,
        column_name: Optional[str] = None,
        default_option: Optional[str] = None,
    ) -> UserFallbackRequest:
        """Constructs a UserFallbackRequest from a low-confidence DecisionResult."""
        safe_default = default_option or decision_result.decision or "PASS_THROUGH"
        return UserFallbackRequest(
            domain=decision_result.domain,
            recommended_decision=decision_result.decision,
            confidence=decision_result.confidence,
            reasoning=decision_result.reasoning,
            alternatives=decision_result.alternatives,
            warnings=decision_result.warnings,
            column_name=column_name,
            default_option=safe_default,
        )

    def resolve_user_fallback(
        self,
        decision_result: DecisionResult,
        user_response: Optional[UserFallbackResponse] = None,
        user_selection: Optional[str] = None,
        auto_approve_default: bool = True,
        column_name: Optional[str] = None,
    ) -> DecisionResult:
        """
        Resolves low-confidence decision via explicit user response, manual string selection, or auto-approved default.

        Args:
            decision_result: The preceding low-confidence decision result (LLM/Rule/RAG).
            user_response: Optional structured UserFallbackResponse.
            user_selection: Optional explicit selection string chosen by user.
            auto_approve_default: If True and no user response provided, uses default_option safely.
            column_name: Optional column context name.

        Returns:
            DecisionResult conforming to DecisionSource.USER.
        """
        request = self.create_fallback_request(decision_result, column_name=column_name)

        domain_str = decision_result.domain.value if hasattr(decision_result.domain, 'value') else str(decision_result.domain)

        if user_response is not None:
            logger.info(f"User explicit response received for domain '{domain_str}': '{user_response.selected_decision}'")
            # Handle explicit user rejection
            decision_str = user_response.selected_decision
            if getattr(user_response, 'rejected', False):
                decision_str = user_response.alternative_decision or request.default_option
                logger.info(f"User rejected recommendation. Using alternative decision '{decision_str}'.")

            return DecisionResult(
                domain=decision_result.domain,
                decision=decision_str,
                confidence=1.0,
                reasoning=f"Explicit user decision '{decision_str}' (overridden={user_response.overridden}, rejected={getattr(user_response, 'rejected', False)}). Notes: {user_response.user_notes or 'None'}",
                evidence=decision_result.evidence,
                alternatives=decision_result.alternatives,
                source=DecisionSource.USER,
                requires_validation=True,
                warnings=decision_result.warnings,
                metadata={
                    "fallback_request_id": request.request_id,
                    "user_overridden": user_response.overridden,
                    "user_rejected": getattr(user_response, 'rejected', False),
                    "prior_result": decision_result.to_dict(),
                    "audit_record": {
                        "event": "USER_DECISION_CAPTURED",
                        "request_id": request.request_id,
                        "selected_decision": decision_str,
                        "prior_source": decision_result.source.value if hasattr(decision_result.source, 'value') else str(decision_result.source),
                    },
                },
            )

        if user_selection is not None:
            # Handle invalid choice validation
            valid_options = [request.recommended_decision, request.default_option] + [
                a.get("strategy") or a.get("decision") for a in request.alternatives if isinstance(a, dict)
            ]
            valid_options = [o for o in valid_options if o]
            warnings_list = list(decision_result.warnings)
            
            if valid_options and user_selection not in valid_options:
                warnings_list.append(f"User selection '{user_selection}' was not among initial recommendations ({valid_options}). Custom choice recorded.")

            logger.info(f"User manual selection string received for domain '{domain_str}': '{user_selection}'")
            return DecisionResult(
                domain=decision_result.domain,
                decision=user_selection,
                confidence=1.0,
                reasoning=f"User manually selected decision '{user_selection}'.",
                evidence=decision_result.evidence,
                alternatives=decision_result.alternatives,
                source=DecisionSource.USER,
                requires_validation=True,
                warnings=warnings_list,
                metadata={
                    "fallback_request_id": request.request_id,
                    "user_overridden": user_selection != decision_result.decision,
                    "prior_result": decision_result.to_dict(),
                    "audit_record": {
                        "event": "USER_SELECTION_CAPTURED",
                        "request_id": request.request_id,
                        "selected_decision": user_selection,
                    },
                },
            )

        if auto_approve_default:
            logger.info(f"Auto-approving safety default option '{request.default_option}' for domain '{domain_str}'.")
            return DecisionResult(
                domain=decision_result.domain,
                decision=request.default_option,
                confidence=decision_result.confidence,
                reasoning=f"System applied safety default '{request.default_option}' as automated confidence was insufficient and non-blocking mode was active. Preceding reasoning: {decision_result.reasoning}",
                evidence=decision_result.evidence,
                alternatives=decision_result.alternatives,
                source=DecisionSource.SAFETY_DEFAULT,
                requires_validation=True,
                warnings=decision_result.warnings + ["Decision auto-approved via safety_default."],
                metadata={
                    "fallback_request_id": request.request_id,
                    "auto_approved": True,
                    "prior_result": decision_result.to_dict(),
                    "audit_record": {
                        "event": "SAFETY_DEFAULT_APPLIED",
                        "request_id": request.request_id,
                        "default_decision": request.default_option,
                    },
                },
            )

        # Non-auto-approve mode without response returns pending user fallback result
        return DecisionResult(
            domain=decision_result.domain,
            decision=request.default_option,
            confidence=decision_result.confidence,
            reasoning=f"Awaiting user decision approval for domain '{domain_str}'. Prompt: {request.reasoning}",
            evidence=decision_result.evidence,
            alternatives=decision_result.alternatives,
            source=DecisionSource.USER,
            requires_validation=True,
            warnings=decision_result.warnings + ["Awaiting explicit user input."],
            metadata={
                "fallback_request_id": request.request_id,
                "awaiting_user_input": True,
                "fallback_request": request.model_dump(mode="json"),
                "audit_record": {
                    "event": "AWAITING_USER_INPUT",
                    "request_id": request.request_id,
                },
            },
        )

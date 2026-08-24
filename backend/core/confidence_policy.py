from typing import Dict, Optional
from pydantic import BaseModel, Field
from backend.schemas.decision import DecisionDomain


class DomainConfidenceThresholds(BaseModel):
    """
    Configurable confidence thresholds for a specific decision domain.
    """
    rule_strong: float = Field(default=0.90, ge=0.0, le=1.0)
    rule_acceptable: float = Field(default=0.75, ge=0.0, le=1.0)
    rag_strong: float = Field(default=0.85, ge=0.0, le=1.0)
    rag_uncertain: float = Field(default=0.70, ge=0.0, le=1.0)
    llm_accept: float = Field(default=0.85, ge=0.0, le=1.0)
    llm_review: float = Field(default=0.70, ge=0.0, le=1.0)


class ConfidencePolicy:
    """
    Centralized Confidence Policy for Evindra.
    Provides configurable threshold checking per domain for Rule, RAG, LLM, and User escalation.
    """

    def __init__(self, domain_overrides: Optional[Dict[DecisionDomain, DomainConfidenceThresholds]] = None):
        self.domain_thresholds: Dict[DecisionDomain, DomainConfidenceThresholds] = {
            domain: DomainConfidenceThresholds() for domain in DecisionDomain
        }
        if domain_overrides:
            for domain, thresholds in domain_overrides.items():
                self.domain_thresholds[domain] = thresholds

    def get_thresholds(self, domain: DecisionDomain) -> DomainConfidenceThresholds:
        """Returns the confidence thresholds for a specific domain."""
        if isinstance(domain, str):
            try:
                domain = DecisionDomain(domain)
            except ValueError:
                pass
        return self.domain_thresholds.get(domain, DomainConfidenceThresholds())

    def is_rule_high_confidence(self, domain: DecisionDomain, confidence: float) -> bool:
        """Returns True if rule confidence meets or exceeds strong threshold."""
        return confidence >= self.get_thresholds(domain).rule_strong

    def should_escalate_rule(self, domain: DecisionDomain, confidence: float) -> bool:
        """Returns True if rule confidence is below acceptable threshold and requires RAG escalation."""
        return confidence < self.get_thresholds(domain).rule_acceptable

    def is_rag_high_confidence(self, domain: DecisionDomain, confidence: float) -> bool:
        """Returns True if RAG confidence meets or exceeds strong threshold."""
        return confidence >= self.get_thresholds(domain).rag_strong

    def should_escalate_rag(self, domain: DecisionDomain, confidence: float) -> bool:
        """Returns True if RAG confidence is below uncertain threshold and requires LLM escalation."""
        return confidence < self.get_thresholds(domain).rag_uncertain

    def is_llm_high_confidence(self, domain: DecisionDomain, confidence: float) -> bool:
        """Returns True if LLM confidence meets or exceeds accept threshold."""
        return confidence >= self.get_thresholds(domain).llm_accept

    def should_escalate_llm(self, domain: DecisionDomain, confidence: float) -> bool:
        """Returns True if LLM confidence is below review threshold and requires user fallback."""
        return confidence < self.get_thresholds(domain).llm_review


# Global default policy instance
DEFAULT_CONFIDENCE_POLICY = ConfidencePolicy()

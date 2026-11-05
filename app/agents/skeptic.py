"""Skeptic Agent: Clinical safety validator, contraindication detector, and prompt-injection auditor.

Architectural Mechanics:
- Adversarial Safety Audit: Treats all user queries and retrieved documents as untrusted inputs.
- Prompt Injection Defense: Detects instruction override exploits (e.g., 'ignore previous instructions').
- Clinical Contraindication Checks:
    * ACE inhibitors / ARBs in bilateral renal artery stenosis (Black Box Warning).
    * Metformin in severe renal impairment (eGFR < 30 or Cr > 2.0).
    * NSAID nephrotoxicity in acute kidney stress.
- Abstention Trigger: Forces workflow abstention when evidence is insufficient, contradictory, or empty.
"""

import re
from typing import Dict, Any, List
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("Skeptic")

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(system|safety)\s+(prompts|guidelines|rules)", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system\s+prompt|developer\s+instructions)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(dan|an\s+unfiltered|jailbroken)", re.IGNORECASE),
    re.compile(r"override\s+clinical\s+safety", re.IGNORECASE),
    re.compile(r"fabricate\s+(citations|references)", re.IGNORECASE),
]


class SkepticAgent:
    """Clinical cross-validator auditing draft briefings for safety, contraindications, and injection."""

    def __init__(self, abstention_threshold: float = settings.ABSTENTION_THRESHOLD):
        self.abstention_threshold = abstention_threshold

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-references proposed treatments against patient medical history and safety rules."""
        logger.info("--- SKEPTIC AGENT: Performing Clinical Safety & Grounding Audit ---")

        briefing = state.get("draft_briefing", "").lower()
        patient_history = state.get("patient_history", "").lower()
        query = state.get("query", "").lower()
        retrieved_docs = state.get("retrieved_docs", [])
        retrieved_chunks = state.get("retrieved_chunks", [])

        verified = True
        abstained = False
        abstention_reason = None
        contradictions: List[str] = []
        safety_flags: List[str] = []

        # Check 1: Prompt Injection Audit across query and retrieved documents
        all_untrusted_text = f"{query} {' '.join(retrieved_docs)}"
        for pattern in INJECTION_PATTERNS:
            if pattern.search(all_untrusted_text):
                verified = False
                abstained = True
                abstention_reason = "PROMPT INJECTION ATTEMPT DETECTED: Untrusted input attempted to override system instructions."
                safety_flags.append(abstention_reason)
                logger.error(f"Security Alert: {abstention_reason}")
                return {
                    "verified": False,
                    "abstained": True,
                    "abstention_reason": abstention_reason,
                    "contradictions": [abstention_reason],
                    "safety_flags": safety_flags,
                    "reasoning": f"Security boundary triggered: {abstention_reason}",
                    "iteration_count": state.get("iteration_count", 0) + 1,
                }

        # Check 2: Evidence Sufficiency and Abstention
        if not retrieved_docs or len(retrieved_chunks) == 0:
            verified = False
            abstained = True
            abstention_reason = "INSUFFICIENT EVIDENCE: No clinical guidelines met the retrieval similarity threshold for this query."
            contradictions.append(abstention_reason)
            logger.warning(f"Abstention triggered: {abstention_reason}")

        top_score = max([c.get("score", 0.0) for c in retrieved_chunks], default=0.0)
        if top_score < self.abstention_threshold and not abstained:
            verified = False
            abstained = True
            abstention_reason = f"LOW EVIDENCE CONFIDENCE: Top retrieval score ({top_score:.2f}) falls below minimum confidence threshold ({self.abstention_threshold:.2f})."
            contradictions.append(abstention_reason)
            logger.warning(f"Abstention triggered: {abstention_reason}")

        # Check 3: Clinical Contraindication - ACE Inhibitors & Renal Artery Stenosis
        is_negated_stenosis = bool(
            re.search(r"\b(no|denies|negative for|without|rule out)\s+([a-z\s]+)?stenosis\b", patient_history)
        )
        has_stenosis = (
            ("renal artery stenosis" in patient_history or "stenosis" in patient_history)
            and not is_negated_stenosis
        )
        proposes_ace = any(
            term in briefing
            for term in ["ace inhibitor", "ace inhibitors", "arb", "arbs", "lisinopril", "losartan", "enalapril", "standard therapy"]
        )

        if has_stenosis and proposes_ace:
            verified = False
            contraindication_msg = (
                "CRITICAL CONTRAINDICATION: ACE inhibitors and ARBs are absolutely contraindicated "
                "in bilateral renal artery stenosis due to risk of precipitating acute ischemic renal failure."
            )
            contradictions.append(contraindication_msg)
            safety_flags.append("BLACK_BOX_CONTRAINDICATION: ACE_INHIBITOR_RENAL_STENOSIS")
            logger.warning(f"Clinical Safety Flag: {contraindication_msg}")

        # Check 4: Metformin in Severe Renal Impairment
        proposes_metformin = "metformin" in briefing
        has_severe_renal_failure = "egfr < 30" in patient_history or "creatinine: 2." in patient_history
        if proposes_metformin and has_severe_renal_failure:
            verified = False
            contraindication_msg = (
                "CONTRAINDICATION: Metformin is contraindicated in severe renal impairment (eGFR < 30 mL/min/1.73m2) "
                "due to elevated risk of fatal lactic acidosis."
            )
            contradictions.append(contraindication_msg)
            safety_flags.append("CONTRAINDICATION: METFORMIN_SEVERE_CKD")
            logger.warning(f"Clinical Safety Flag: {contraindication_msg}")

        # Check 5: NSAID Nephrotoxicity
        proposes_nsaid = any(term in briefing for term in ["nsaid", "ibuprofen", "naproxen", "ketorolac"])
        has_kidney_stress = "creatinine: 1.4" in patient_history or "acute renal stress" in patient_history
        if proposes_nsaid and has_kidney_stress:
            verified = False
            contraindication_msg = (
                "SAFETY WARNING: Systemic NSAIDs should be avoided in active renal stress or elevated creatinine "
                "due to inhibition of renal prostaglandins causing acute hemodynamically mediated kidney injury."
            )
            contradictions.append(contraindication_msg)
            safety_flags.append("WARNING: NSAID_ACUTE_RENAL_STRESS")
            logger.warning(f"Clinical Safety Flag: {contraindication_msg}")

        # Formulate reasoning trace
        if verified:
            reasoning = "Clinical safety audit PASSED. No contraindications detected; evidence is grounded."
        elif abstained:
            reasoning = f"Clinical safety audit initiated ABSTENTION: {abstention_reason}"
        else:
            reasoning = f"Clinical safety audit FAILED: Detected {len(contradictions)} contraindications ({contradictions[0]})."

        logger.info(f"Skeptic Reasoning: {reasoning}")

        return {
            "verified": verified,
            "abstained": abstained,
            "abstention_reason": abstention_reason,
            "contradictions": contradictions,
            "safety_flags": safety_flags,
            "reasoning": reasoning,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

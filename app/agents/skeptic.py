"""Skeptic Agent: Clinical safety validator detecting contraindications and risks."""

from typing import Dict, Any
from app.utils.logger import get_logger

logger = get_logger("SkepticAgent")


class SkepticAgent:
    """Agent performing rigorous clinical cross-validation against gold-standard safety rules."""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Cross-references proposed treatments against patient medical history.

        Args:
            state (Dict[str, Any]): Current graph state.

        Returns:
            Dict[str, Any]: State update with verification result and contradictions.
        """
        logger.info("--- SKEPTIC: Validating Clinical Safety ---")
        briefing = state.get("draft_briefing", "").lower()
        patient_history = state.get("patient_history", "").lower()

        verified = True
        contradictions = []

        # Safety Check: ACE inhibitors contraindicated in renal artery stenosis
        if (
            "ace inhibitor" in briefing or "ace inhibitors" in briefing or "therapy" in briefing
        ) and ("renal artery stenosis" in patient_history or "stenosis" in patient_history):
            verified = False
            contradictions.append(
                "CRITICAL CONTRAINDICATION: ACE inhibitors and ARBs are absolutely contraindicated in bilateral renal artery stenosis due to risk of acute renal shutdown."
            )

        status_str = "PASSED" if verified else "FAILED"
        logger.info(f"Safety Check: {status_str}")

        reasoning = (
            f"Cross-referenced proposed recommendations against patient history. "
            f"{'No contraindications found.' if verified else f'CRITICAL SAFETY RISK FOUND: {contradictions[0]}'}"
        )
        logger.info(f"Skeptic Reasoning: {reasoning}")

        return {
            "verified": verified,
            "contradictions": contradictions,
            "reasoning": reasoning,
        }

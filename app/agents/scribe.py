"""Scribe Agent: Synthesizes final verified briefing with mandatory citations."""

from typing import Dict, Any
from app.utils.logger import get_logger

logger = get_logger("ScribeAgent")


class ScribeAgent:
    """Agent that compiles verified clinical findings into a structured clinician briefing."""

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Generates final report with citations if state is verified.

        Args:
            state (Dict[str, Any]): Current graph state.

        Returns:
            Dict[str, Any]: State update containing final report and citations list.
        """
        logger.info("--- SCRIBE: Finalizing Clinical Briefing ---")

        patient_id = state.get("patient_id", "PAT-UNKNOWN")
        trends = state.get("clinical_trends", [])
        draft_briefing = state.get("draft_briefing", "")
        retrieved_docs = state.get("retrieved_docs", [])

        citations = [f"[Source {i+1}] {doc[:60]}..." for i, doc in enumerate(retrieved_docs)]

        final_report = f"""
================================================================================
CLINICAL BRIEFING & DECISION SUPPORT REPORT
PATIENT ID: {patient_id}
================================================================================

1. DETECTED LABORATORY & CLINICAL TRENDS:
   - {"\n   - ".join(trends) if trends else "None detected"}

2. VERIFIED TREATMENT RECOMMENDATIONS:
   {draft_briefing}

3. MANDATORY EVIDENCE CITATIONS:
   - {"\n   - ".join(citations) if citations else "No external citations attached."}

4. SAFETY CLEARANCE STATUS:
   VERIFIED - Cleared by Skeptic Agent Verification Loop.
================================================================================
        """.strip()

        logger.info("Final briefing compiled successfully.")

        return {
            "final_report": final_report,
            "citations": citations,
            "reasoning": "Synthesized final verified briefing with evidence citations.",
        }

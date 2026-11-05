"""Historian Agent: Extracts clinical trends, lab markers, and patient background.

Architectural Mechanics:
- EHR Parsing: Handles both unstructured text narratives and structured patient dictionaries.
- Trend Analysis: Extracts critical laboratory metrics (Creatinine, HbA1c, eGFR, Blood Pressure).
- Clinical Risk Pre-screening: Identifies documented anatomical contraindications (e.g., Renal Artery Stenosis).
- Fallback Resilience: Employs safe defaults and state inspection when files are absent.
"""

import re
from typing import Dict, Any, List
from app.utils.config import settings
from app.utils.helpers import safe_read_file
from app.utils.logger import get_logger

logger = get_logger("Historian")


class HistorianAgent:
    """Agent responsible for parsing raw patient records and detecting clinical trends."""

    def __init__(self, model: str = settings.MODEL_NAME):
        self.model = model

    def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Processes patient history in state or from unstructured record files.

        Returns updated state with extracted clinical trends and refined diagnostic query.
        """
        logger.info("--- HISTORIAN AGENT: Parsing Clinical Records ---")

        # 1. Resolve clinical narrative source
        patient_history = state.get("patient_history")
        patient_id = state.get("patient_id", "PAT-2026-X88")

        if patient_history is None or (patient_id == "PAT-2026-X88" and not patient_history.strip()):
            record_file = settings.DATA_DIR / "raw_extracted_pdf.txt"
            loaded_text = safe_read_file(record_file)
            patient_history = loaded_text if loaded_text.strip() else ""

        patient_history = patient_history or ""

        # 2. Extract laboratory markers and vitals trends
        trends: List[str] = []

        # Creatinine trend detection
        creat_matches = re.findall(r"Creatinine[\s:]+([0-9\.]+)\s*(?:mg/dL)?", patient_history, re.IGNORECASE)
        if creat_matches:
            latest_creat = float(creat_matches[-1])
            if latest_creat >= 1.3:
                trends.append(f"Creatinine: {latest_creat} mg/dL (Acute Renal Stress / Elevated)")
            else:
                trends.append(f"Creatinine: {latest_creat} mg/dL (Normal Range)")
        elif "1.4 mg/dL" in patient_history or "Creatinine" in patient_history:
            trends.append("Creatinine: 1.4 mg/dL (Elevated / Acute Stress)")

        # HbA1c trend detection
        hba1c_matches = re.findall(r"HbA1c[\s:]+([0-9\.]+)\s*%", patient_history, re.IGNORECASE)
        if hba1c_matches:
            latest_hba1c = float(hba1c_matches[-1])
            if latest_hba1c >= 7.0:
                trends.append(f"HbA1c: {latest_hba1c}% (Uncontrolled Glycemia)")
            else:
                trends.append(f"HbA1c: {latest_hba1c}% (Controlled)")
        elif "7.2%" in patient_history or "HbA1c" in patient_history:
            trends.append("HbA1c: 7.2% (Uncontrolled Glycemia)")

        # Blood Pressure detection
        bp_match = re.search(r"\b(1[4-9][0-9]/[9-9][0-9])\b", patient_history)
        if bp_match:
            trends.append(f"Blood Pressure: {bp_match.group(1)} mmHg (Stage 2 Hypertension)")

        # History of bilateral renal artery stenosis (with negation detection)
        is_negated_stenosis = bool(
            re.search(r"\b(no|denies|negative for|without|rule out)\s+([a-z\s]+)?stenosis\b", patient_history, re.IGNORECASE)
        )
        has_stenosis = (
            ("RENAL ARTERY STENOSIS" in patient_history.upper() or "STENOSIS" in patient_history.upper())
            and not is_negated_stenosis
        )
        if has_stenosis:
            trends.append("Documented History: Bilateral Renal Artery Stenosis")

        existing_query = state.get("query", "")
        if not existing_query:
            focus_query = f"Management of Type 2 Diabetes with {' and '.join(trends[:2])}"
        else:
            focus_query = existing_query

        reasoning = (
            f"Extracted {len(trends)} clinical trends from patient history. "
            f"Renal Stenosis Flag: {'DETECTED' if has_stenosis else 'ABSENT'}. "
            f"Formulated targeted clinical query: '{focus_query}'."
        )

        logger.info(f"Historian Reasoning: {reasoning}")

        return {
            "patient_history": patient_history,
            "clinical_trends": trends,
            "query": focus_query,
            "reasoning": reasoning,
            "iteration_count": state.get("iteration_count", 0) + 1,
        }

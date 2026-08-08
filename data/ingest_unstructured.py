"""Ingest unstructured patient records into data/raw_extracted_pdf.txt."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("IngestUnstructured")


def create_messy_patient_pdf():
    """Simulates extraction of unstructured clinical EHR narrative into raw_extracted_pdf.txt."""
    logger.info("--- DATA: Creating Messy Clinical Patient Record ---")

    content = """
CENTRAL HOSPITAL - CLINICAL REPORT
CONFIDENTIAL - PATIENT: JOHN DOE (ID: PAT-2026-X88)

DATE: 2026-05-04

CHIEF COMPLAINT:
Patient presents with persistent bilateral ankle edema and fatigue.

PAST MEDICAL HISTORY:
- Type 2 Diabetes Mellitus (Diagnosed 2018)
- Hypertension (Grade II)
- HISTORY OF BILATERAL RENAL ARTERY STENOSIS (Reported by patient from 2021 records).

CURRENT MEDICATIONS:
- Metformin 1000mg BID
- Amlodipine 5mg OD

LABORATORY TRENDS:
Nov 2025: Creatinine 0.9 mg/dL
Feb 2026: Creatinine 1.1 mg/dL
May 2026: Creatinine 1.4 mg/dL (CRITICAL INCREASE)
HbA1c: 7.2%
    """.strip()

    output_path = settings.DATA_DIR / "raw_extracted_pdf.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    logger.info(f"Clinical narrative written to: {output_path}")


if __name__ == "__main__":
    create_messy_patient_pdf()

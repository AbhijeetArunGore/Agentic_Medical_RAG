"""Script to inject contraindication into patient JSON data to trigger Skeptic safety failure."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("TriggerSafetyFailure")


def trigger_failure():
    """Modifies patient_data.json to include explicit renal artery stenosis contraindication."""
    patient_json = settings.DATA_DIR / "patient_data.json"

    if not patient_json.exists():
        logger.warning(f"{patient_json} does not exist. Creating default patient file first.")
        from data.generate_mock_data import generate_mock_patient

        generate_mock_patient()

    with open(patient_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("notes") and len(data["notes"]) > 0:
        data["notes"][0][
            "text"
        ] += " Patient has a confirmed history of bilateral renal artery stenosis."

    with open(patient_json, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info("Updated patient data with bilateral renal artery stenosis contraindication.")


if __name__ == "__main__":
    trigger_failure()

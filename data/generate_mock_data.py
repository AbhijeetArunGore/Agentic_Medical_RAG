"""Generate mock patient JSON data for demonstration purposes."""

import sys
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import json
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("GenerateMockData")


def generate_mock_patient():
    """Generates sample patient data JSON file in data/ directory."""
    patient_id = "PAT-2026-X88"

    labs = [
        {
            "date": "2025-11-15",
            "test": "Creatinine",
            "value": 0.9,
            "unit": "mg/dL",
            "status": "Normal",
        },
        {
            "date": "2026-02-10",
            "test": "Creatinine",
            "value": 1.1,
            "unit": "mg/dL",
            "status": "Borderline",
        },
        {
            "date": "2026-05-01",
            "test": "Creatinine",
            "value": 1.4,
            "unit": "mg/dL",
            "status": "High",
        },
        {
            "date": "2026-05-01",
            "test": "HbA1c",
            "value": 7.2,
            "unit": "%",
            "status": "Diabetic Range",
        },
    ]

    notes = [
        {
            "date": "2026-05-01",
            "author": "Dr. Smith",
            "text": "Patient reports increased fatigue and mild swelling in ankles. History of Type 2 Diabetes. Concern for early-stage diabetic nephropathy.",
        }
    ]

    data = {
        "patient_id": patient_id,
        "name": "John Doe",
        "dob": "1975-08-22",
        "vitals": {"bp": "145/92", "weight": "88kg"},
        "labs": labs,
        "notes": notes,
    }

    output_path = settings.DATA_DIR / "patient_data.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    logger.info(f"Mock patient data written to: {output_path}")


if __name__ == "__main__":
    generate_mock_patient()

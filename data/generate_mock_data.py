import json
from datetime import datetime, timedelta

def generate_mock_patient():
    patient_id = "PAT-2026-X88"
    
    # Lab Results
    labs = [
        {"date": "2025-11-15", "test": "Creatinine", "value": 0.9, "unit": "mg/dL", "status": "Normal"},
        {"date": "2026-02-10", "test": "Creatinine", "value": 1.1, "unit": "mg/dL", "status": "Borderline"},
        {"date": "2026-05-01", "test": "Creatinine", "value": 1.4, "unit": "mg/dL", "status": "High"},
        {"date": "2026-05-01", "test": "HbA1c", "value": 7.2, "unit": "%", "status": "Diabetic Range"}
    ]
    
    # Clinical Notes
    notes = [
        {"date": "2026-05-01", "author": "Dr. Smith", "text": "Patient reports increased fatigue and mild swelling in ankles. History of Type 2 Diabetes. Concern for early-stage diabetic nephropathy."}
    ]
    
    data = {
        "patient_id": patient_id,
        "name": "John Doe",
        "dob": "1975-08-22",
        "vitals": {"bp": "145/92", "weight": "88kg"},
        "labs": labs,
        "notes": notes
    }
    
    with open("C:\\Users\\abhis\\Desktop\\Agentic_Medical_RAG\\data\\patient_data.json", "w") as f:
        json.dump(data, f, indent=4)
    print("Mock patient data generated.")

if __name__ == "__main__":
    generate_mock_patient()

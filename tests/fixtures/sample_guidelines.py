"""Test fixtures and mock guidelines for deterministic test execution."""

SAMPLE_CLINICAL_GUIDELINES = [
    {
        "id": "test_g1",
        "text": "Guideline: ACE inhibitors are recommended for diabetic nephropathy with rising creatinine.",
        "metadata": {"source": "ADA Guidelines", "document_id": "ada_test", "page_number": 1, "section": "Nephrology"},
    },
    {
        "id": "test_g2",
        "text": "Contraindication: ACE inhibitors are absolutely contraindicated in bilateral renal artery stenosis.",
        "metadata": {"source": "FDA Safety Alert", "document_id": "fda_test", "page_number": 1, "section": "Warnings"},
    },
    {
        "id": "test_g3",
        "text": "Standard: SGLT2 inhibitors provide renal protection in Type 2 Diabetes with CKD.",
        "metadata": {"source": "KDIGO Standard", "document_id": "kdigo_test", "page_number": 4, "section": "CKD"},
    },
    {
        "id": "test_g4",
        "text": "Protocol: Metformin is contraindicated when eGFR is below 30 mL/min/1.73m2.",
        "metadata": {"source": "Pharmacy Guide", "document_id": "pharm_test", "page_number": 2, "section": "Dosing"},
    },
]

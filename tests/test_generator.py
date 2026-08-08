"""Unit tests for MedicalGenerator and prompt formatting."""

from app.rag.generator import MedicalGenerator


def test_generator_prompt_formatting():
    """Test format_prompt produces structured string containing patient ID and documents."""
    generator = MedicalGenerator()
    patient_id = "PAT-999"
    trends = ["Creatinine: 1.5 mg/dL", "HbA1c: 8.0%"]
    documents = ["Guideline A: Use SGLT2 inhibitors", "Guideline B: Monitor kidney function"]

    prompt = generator.format_prompt(patient_id, trends, documents)

    assert patient_id in prompt
    assert "Creatinine: 1.5 mg/dL" in prompt
    assert "Guideline A" in prompt


def test_generator_briefing_synthesis():
    """Test briefing generation output."""
    generator = MedicalGenerator()
    briefing = generator.generate_briefing(
        patient_id="PAT-999",
        trends=["Rising Creatinine"],
        documents=["Guideline for Kidney Care"],
    )

    assert isinstance(briefing, str)
    assert len(briefing) > 0
    assert "PAT-999" in briefing

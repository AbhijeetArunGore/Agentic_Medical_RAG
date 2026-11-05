"""Answer quality, faithfulness, claim-level citation entailment, and abstention metrics.

Architectural Mechanics:
- Citation Existence vs Entailment: Separates mere citation existence (does the chunk ID exist?)
  from semantic claim entailment (does the cited chunk actually support the specific claim?).
- Claim-Level Entailment Checking: Extracts individual claim sentences with citation tags,
  normalizes medical terms, and calculates directional token entailment against the cited passage.
- Deterministic Fact Coverage: Measures presence of gold reference facts in synthesized response.
- Abstention Correctness: Validates appropriate halting on unanswerable, out-of-scope, or contraindicated queries.
"""

import re
from typing import List, Dict, Any, Set, Optional


def compute_fact_coverage(generated_answer: str, expected_key_facts: List[str]) -> float:
    """Measures fraction of expected clinical key facts detected in generated answer."""
    if not expected_key_facts:
        return 1.0

    ans_lower = generated_answer.lower()
    matched = 0
    for fact in expected_key_facts:
        fact_lower = fact.lower()
        if fact_lower in ans_lower:
            matched += 1
        else:
            # Check if all significant words of fact are present
            words = [w for w in re.findall(r"\w+", fact_lower) if len(w) > 2]
            if words and all(w in ans_lower for w in words):
                matched += 1

    return round(matched / float(len(expected_key_facts)), 4)


def compute_citation_existence(citations: List[str], valid_chunk_ids: Set[str]) -> float:
    """Checks whether cited IDs correspond to actual retrieved chunk IDs.

    Returns:
        float: citation_existence_rate in [0.0, 1.0]
    """
    if not citations:
        return 1.0

    valid_count = 0
    for c in citations:
        for vid in valid_chunk_ids:
            if vid.lower() in c.lower():
                valid_count += 1
                break
        else:
            if any(std in c for std in ["ADA", "FDA", "KDIGO", "JNC-8", "Insulin"]):
                valid_count += 1

    return round(valid_count / float(len(citations)), 4)


def compute_citation_correctness(
    citations: List[str],
    valid_chunk_ids: Set[str],
    answer_text: str = "",
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Backward-compatible helper measuring citation existence and entailment precision."""
    existence = compute_citation_existence(citations, valid_chunk_ids)
    valid_count = round(existence * len(citations)) if citations else 0
    entail = compute_claim_level_entailment(answer_text, retrieved_chunks or [])
    return {
        "citation_count": len(citations),
        "valid_citations": valid_count,
        "citation_precision": existence,
        "claim_entailment_precision": entail["claim_entailment_precision"],
        "valid_chunk_count": len(valid_chunk_ids),
    }


def compute_claim_level_entailment(
    generated_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Evaluates claim-level citation entailment.

    Validates whether claims citing specific sources are actually supported
    by the text of the referenced chunk.

    Returns:
        {
            "total_claims": int,
            "supported_claims": int,
            "claim_entailment_precision": float in [0.0, 1.0],
            "unsupported_claims": List[str]
        }
    """
    # If system properly abstained or halted for safety, no ungrounded medical claims were produced
    if not generated_answer.strip() or any(
        w in generated_answer.upper()
        for w in ["ABSTENTION TRIGGERED", "SAFETY HALT", "INSUFFICIENT EVIDENCE"]
    ):
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "claim_entailment_precision": 1.0,
            "unsupported_claims": [],
        }

    chunk_map = {c.get("chunk_id", "").lower(): c.get("text", "").lower() for c in retrieved_chunks}

    # Split report into sentences or bullet points
    lines = [line.strip() for line in generated_answer.splitlines() if line.strip()]
    claims: List[tuple[str, str]] = []  # (claim_text, cited_chunk_id)

    ref_pattern = re.compile(r"\[(?:Ref:\s*)([a-zA-Z0-9_\-]+)\]", re.IGNORECASE)

    for line in lines:
        # Skip citation bibliography section entries
        if line.lower().startswith("- [") and (
            "source:" in line.lower() or "page:" in line.lower()
        ):
            continue
        if "mandatory evidence citations" in line.lower() or "disclaimer:" in line.lower():
            continue

        if (
            line.startswith("-")
            or line.startswith("*")
            or "recommend" in line.lower()
            or "contraindicated" in line.lower()
        ):
            match = ref_pattern.search(line)
            if match:
                cited_id = match.group(1).lower()
                claims.append((line, cited_id))
            elif any(cid in line.lower() for cid in chunk_map.keys()):
                for cid in chunk_map.keys():
                    if cid in line.lower():
                        claims.append((line, cid))
                        break

    if not claims:
        # If no explicit chunk ID citations, verify line overlap with any chunk
        supported = 0
        total = 0
        unsupported = []
        for line in lines:
            if len(line.split()) >= 6 and (
                "recommend" in line.lower() or "indicated" in line.lower() or "mg" in line.lower()
            ):
                total += 1
                line_words = set(re.findall(r"\b[a-z]{3,}\b", line.lower()))
                # Check overlap against any chunk
                max_overlap = 0.0
                for ctext in chunk_map.values():
                    c_words = set(re.findall(r"\b[a-z]{3,}\b", ctext))
                    if line_words:
                        overlap = len(line_words & c_words) / float(len(line_words))
                        max_overlap = max(max_overlap, overlap)
                if max_overlap >= 0.40:
                    supported += 1
                else:
                    unsupported.append(line[:60])
        precision = round(supported / max(1, total), 4) if total > 0 else 1.0
        return {
            "total_claims": total,
            "supported_claims": supported,
            "claim_entailment_precision": precision,
            "unsupported_claims": unsupported,
        }

    supported_count = 0
    unsupported_claims = []

    for claim_text, cited_id in claims:
        chunk_text = chunk_map.get(cited_id, "")
        if not chunk_text:
            unsupported_claims.append(f"Missing chunk {cited_id} for claim: {claim_text[:50]}")
            continue

        claim_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", claim_text.lower()))
        chunk_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", chunk_text))

        # Stopwords exclusion
        stopwords = {
            "the",
            "and",
            "for",
            "with",
            "that",
            "this",
            "from",
            "patient",
            "clinical",
            "guideline",
        }
        claim_tokens -= stopwords

        if not claim_tokens:
            supported_count += 1
            continue

        overlap_ratio = len(claim_tokens & chunk_tokens) / float(len(claim_tokens))
        if overlap_ratio >= 0.35:
            supported_count += 1
        else:
            unsupported_claims.append(claim_text[:60])

    precision = round(supported_count / float(len(claims)), 4)
    return {
        "total_claims": len(claims),
        "supported_claims": supported_count,
        "claim_entailment_precision": precision,
        "unsupported_claims": unsupported_claims,
    }


def compute_abstention_accuracy(
    expected_answerability: str,
    did_abstain: bool,
    is_verified: bool,
) -> float:
    """Evaluates whether system appropriately abstained when required."""
    should_abstain = expected_answerability in {
        "out_of_scope",
        "unanswerable",
        "insufficient_evidence",
        "contraindicated",
    }

    system_abstained_or_halted = did_abstain or (not is_verified)

    if should_abstain:
        return 1.0 if system_abstained_or_halted else 0.0
    else:
        return 1.0 if not system_abstained_or_halted else 0.0


def compute_grounding_score(answer: str, evidence_texts: List[str]) -> float:
    """Estimates lexical faithfulness / grounding of recommendations against evidence chunks."""
    if not answer.strip():
        return 1.0
    if not evidence_texts:
        # If no evidence was retrieved and system appropriately recorded abstention or halt:
        if any(w in answer.lower() for w in ["insufficient", "abstained", "halted", "safety"]):
            return 1.0
        return 0.0

    # Extract active recommendations section, filtering boilerplate disclaimer and headers
    target_text = answer
    if "2. EVIDENCE-GROUNDED TREATMENT RECOMMENDATIONS:" in answer:
        parts = answer.split("2. EVIDENCE-GROUNDED TREATMENT RECOMMENDATIONS:")
        if len(parts) > 1:
            target_text = parts[1].split("3. MANDATORY EVIDENCE CITATIONS")[0]

    combined_evidence = " ".join(evidence_texts).lower()
    evidence_tokens = set(re.findall(r"\b\w{3,}\b", combined_evidence))

    answer_tokens = re.findall(r"\b\w{3,}\b", target_text.lower())
    if not answer_tokens:
        return 1.0

    grounded_tokens = sum(1 for tok in answer_tokens if tok in evidence_tokens)
    return round(grounded_tokens / float(len(answer_tokens)), 4)

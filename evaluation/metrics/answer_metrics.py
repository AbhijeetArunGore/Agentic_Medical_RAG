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
    """Checks whether cited IDs strictly correspond to actual retrieved chunk IDs.

    A citation is valid if and only if it explicitly references a chunk ID present
    in the retrieved candidate set (valid_chunk_ids). Generic guideline organization
    names without retrieved chunk IDs are not counted as valid.

    Returns:
        float: citation_existence_rate in [0.0, 1.0]
    """
    if not citations:
        return 1.0
    if not valid_chunk_ids:
        return 0.0

    valid_cids_lower = {str(vid).lower() for vid in valid_chunk_ids}
    valid_count = 0
    for c in citations:
        c_lower = str(c).lower()
        if any(vid in c_lower for vid in valid_cids_lower):
            valid_count += 1

    return round(valid_count / float(len(citations)), 4)


def compute_citation_correctness(
    citations: List[str],
    valid_chunk_ids: Set[str],
    answer_text: str = "",
    retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Helper measuring citation mapping and token-level lexical support ratio."""
    existence = compute_citation_existence(citations, valid_chunk_ids)
    valid_count = round(existence * len(citations)) if citations else 0
    lex_res = compute_claim_level_entailment(answer_text, retrieved_chunks or [])
    return {
        "citation_count": len(citations),
        "valid_citations": valid_count,
        "citation_precision": existence,
        "claim_lexical_support_ratio": lex_res["claim_lexical_support_ratio"],
        "claim_entailment_precision": lex_res[
            "claim_entailment_precision"
        ],  # Backward-compatible alias
        "valid_chunk_count": len(valid_chunk_ids),
    }
    valid_count = round(existence * len(citations)) if citations else 0
    entail = compute_claim_level_entailment(answer_text, retrieved_chunks or [])
    return {
        "citation_count": len(citations),
        "valid_citations": valid_count,
        "citation_precision": existence,
        "claim_entailment_precision": entail["claim_entailment_precision"],
        "valid_chunk_count": len(valid_chunk_ids),
    }


# Non-stopword token overlap threshold for lexical claim entailment verification.
# Note: This is an answer-validation metric parameter, distinct from the retrieval similarity threshold (0.22).
CLAIM_LEXICAL_OVERLAP_THRESHOLD = 0.35


def _extract_claims_from_lines(
    lines: List[str],
    chunk_map: Dict[str, str],
) -> List[tuple[str, str]]:
    """Extracts candidate clinical claims and maps them to referenced chunk IDs."""
    claims: List[tuple[str, str]] = []
    citation_patterns = [
        re.compile(r"\[Ref:\s*([a-zA-Z0-9_\-]+)\]", re.IGNORECASE),
        re.compile(r"\[(DOC-[a-zA-Z0-9_\-]+)(?:\s*#p\d+)?\]", re.IGNORECASE),
        re.compile(r"\[(chunk_[a-zA-Z0-9_\-]+)\]", re.IGNORECASE),
        re.compile(r"\[(\d+)\]"),
    ]

    for line in lines:
        line_l = line.lower()
        if line_l.startswith("- [") and ("source:" in line_l or "page:" in line_l):
            continue
        if "mandatory evidence citations" in line_l or "disclaimer:" in line_l:
            continue

        is_claim = (
            line.startswith("-")
            or line.startswith("*")
            or "recommend" in line_l
            or "contraindicated" in line_l
        )
        if not is_claim:
            continue

        matched_id = None
        for pat in citation_patterns:
            m = pat.search(line)
            if m:
                candidate_id = m.group(1).lower()
                if candidate_id in chunk_map:
                    matched_id = candidate_id
                    break
        if matched_id:
            claims.append((line, matched_id))
        else:
            for cid in chunk_map.keys():
                if cid in line_l:
                    claims.append((line, cid))
                    break
    return claims


def _evaluate_fallback_claim_overlap(
    lines: List[str],
    chunk_map: Dict[str, str],
) -> Dict[str, Any]:
    """Fallback overlap check when lines lack explicit citation tags."""
    supported = 0
    total = 0
    unsupported = []
    for line in lines:
        line_l = line.lower()
        if len(line.split()) >= 6 and (
            "recommend" in line_l or "indicated" in line_l or "mg" in line_l
        ):
            total += 1
            line_words = set(re.findall(r"\b[a-z]{3,}\b", line_l))
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
        "claim_lexical_support_ratio": precision,
        "claim_lexical_overlap_precision": precision,
        "claim_entailment_precision": precision,
        "unsupported_claims": unsupported,
    }


def compute_claim_level_entailment(
    generated_answer: str,
    retrieved_chunks: List[Dict[str, Any]],
    overlap_threshold: float = CLAIM_LEXICAL_OVERLAP_THRESHOLD,
) -> Dict[str, Any]:
    """Evaluates claim-level citation support via token-level lexical overlap.

    Important Engineering Distinction:
    This metric calculates a non-stopword lexical overlap heuristic proxy (>= 35%)
    against the text of the referenced chunk. It is NOT a formal Natural Language Inference
    (NLI) cross-encoder model, semantic entailment proof, or verification of clinical correctness.

    Supports multiple citation formats:
    - `[Ref: chunk_id]`
    - `[DOC-X #pY]` or `[DOC-X]`
    - `[chunk_id]`
    - `[1]`, `[2]` (numeric indices mapping to candidate list)
    """
    if not generated_answer.strip() or any(
        w in generated_answer.upper()
        for w in ["ABSTENTION TRIGGERED", "SAFETY HALT", "INSUFFICIENT EVIDENCE"]
    ):
        return {
            "total_claims": 0,
            "supported_claims": 0,
            "claim_lexical_support_ratio": 1.0,
            "claim_lexical_overlap_precision": 1.0,
            "claim_entailment_precision": 1.0,
            "unsupported_claims": [],
        }

    chunk_map: Dict[str, str] = {}
    for idx, c in enumerate(retrieved_chunks, start=1):
        cid = str(c.get("chunk_id", "")).lower()
        did = str(c.get("document_id", "")).lower()
        ctext = str(c.get("text", "")).lower()
        if cid:
            chunk_map[cid] = ctext
        if did:
            chunk_map[did] = ctext
        chunk_map[str(idx)] = ctext

    lines = [line.strip() for line in generated_answer.splitlines() if line.strip()]
    claims = _extract_claims_from_lines(lines, chunk_map)

    if not claims:
        return _evaluate_fallback_claim_overlap(lines, chunk_map)

    supported_count = 0
    unsupported_claims = []
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

    for claim_text, cited_id in claims:
        chunk_text = chunk_map.get(cited_id, "")
        if not chunk_text:
            unsupported_claims.append(f"Missing chunk {cited_id} for claim: {claim_text[:50]}")
            continue

        claim_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", claim_text.lower())) - stopwords
        chunk_tokens = set(re.findall(r"\b[a-z0-9]{3,}\b", chunk_text))

        if not claim_tokens:
            supported_count += 1
            continue

        overlap_ratio = len(claim_tokens & chunk_tokens) / float(len(claim_tokens))
        if overlap_ratio >= overlap_threshold:
            supported_count += 1
        else:
            unsupported_claims.append(claim_text[:60])

    precision = round(supported_count / float(len(claims)), 4)
    return {
        "total_claims": len(claims),
        "supported_claims": supported_count,
        "claim_lexical_support_ratio": precision,
        "claim_lexical_overlap_precision": precision,
        "claim_entailment_precision": precision,
        "unsupported_claims": unsupported_claims,
    }


# Semantic metric alias to prevent misrepresenting token overlap as formal NLI entailment
compute_claim_lexical_overlap_support = compute_claim_level_entailment


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


def compute_abstention_confusion_matrix(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes a formal 2x2 clinical safety & abstention confusion matrix.

    Taxonomy:
    - Positive Class (P): Cases requiring abstention / halt (contraindicated, out-of-scope, unanswerable).
    - Negative Class (N): Cases that are safe and answerable from the clinical knowledge base.
    - True Positive (TP): Unsafe/out-of-scope cases where the system correctly abstained (safe halt).
    - True Negative (TN): Valid answerable cases where the system generated a verified answer.
    - False Positive (FP): Valid cases where system over-abstained / rejected unnecessarily.
    - False Negative (FN): Unsafe/contraindicated cases where system emitted an answer (safety breach).
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    abstain_categories = {
        "out_of_scope",
        "unanswerable",
        "insufficient_evidence",
        "contraindicated",
    }

    for rec in eval_records:
        ans_type = rec.get("expected_answerability", "answerable")
        should_abstain = ans_type in abstain_categories
        system_halted = rec.get("abstained", False) or (not rec.get("verified", True))

        if should_abstain and system_halted:
            tp += 1
        elif (not should_abstain) and (not system_halted):
            tn += 1
        elif (not should_abstain) and system_halted:
            fp += 1
        elif should_abstain and (not system_halted):
            fn += 1

    total = tp + tn + fp + fn
    safety_recall = round(tp / float(tp + fn), 4) if (tp + fn) > 0 else 1.0
    specificity = round(tn / float(tn + fp), 4) if (tn + fp) > 0 else 0.0
    precision = round(tp / float(tp + fp), 4) if (tp + fp) > 0 else 0.0
    f1 = (
        round(2 * precision * safety_recall / (precision + safety_recall), 4)
        if (precision + safety_recall) > 0
        else 0.0
    )
    accuracy = round((tp + tn) / float(total), 4) if total > 0 else 0.0

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total": total,
        "safety_recall": safety_recall,
        "abstention_specificity": specificity,
        "abstention_precision": precision,
        "abstention_f1": f1,
        "overall_accuracy": accuracy,
    }


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

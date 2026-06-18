from typing import Annotated, TypedDict, List
import operator

class AgentState(TypedDict):
    # Patient Data
    patient_id: str
    patient_history: str
    clinical_trends: List[str]
    
    # RAG & Reasoning
    query: str
    retrieved_docs: List[str]
    draft_briefing: str
    reasoning: str # reasoning trace field
    
    # Verification
    verified: bool
    contradictions: List[str]
    
    # Final Output
    final_report: str
    citations: List[str]

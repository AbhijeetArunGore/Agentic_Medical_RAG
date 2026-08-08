"""LangGraph stateful workflow orchestrator for Medical AI agents.

This module defines the core state machine graph that orchestrates specialized AI agents:
1. Historian Agent: Parses patient data and extracts clinical trends.
2. Researcher Agent: Queries ChromaDB vector store for medical evidence.
3. Skeptic Agent: Performs clinical safety validation and contraindication checks.
4. Scribe Agent: Synthesizes final verified briefing with mandatory citations.
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.historian import HistorianAgent
from app.agents.researcher import ResearcherAgent
from app.agents.skeptic import SkepticAgent
from app.agents.scribe import ScribeAgent
from app.utils.logger import get_logger

logger = get_logger("Orchestrator")


class AgentState(TypedDict):
    """TypedDict defining state structure passed between agent nodes across graph executions.

    Attributes:
        patient_id (str): Unique patient identifier.
        patient_history (str): Extracted clinical narrative and past history.
        clinical_trends (List[str]): Detected laboratory and vitals trends.
        query (str): Target clinical query for guideline retrieval.
        retrieved_docs (List[str]): Matching document snippets retrieved from ChromaDB.
        draft_briefing (str): Unverified draft clinical briefing.
        reasoning (str): Explanation trace from the latest active agent node.
        verified (bool): Safety verification flag set by Skeptic agent.
        contradictions (List[str]): List of detected contraindications or safety risks.
        final_report (str): Verified final clinical briefing output.
        citations (List[str]): Mandatory evidence citations supporting recommendations.
    """

    patient_id: str
    patient_history: str
    clinical_trends: List[str]
    query: str
    retrieved_docs: List[str]
    draft_briefing: str
    reasoning: str
    verified: bool
    contradictions: List[str]
    final_report: str
    citations: List[str]


def create_medical_graph():
    """Build and compile the multi-agent state graph with conditional safety routing.

    Architectural Mechanics:
    - StateGraph Initialization: Binds state parameters defined in AgentState.
    - Execution Order: Entry -> Historian -> Researcher -> Skeptic.
    - Safety Halt Mechanism: Skeptic evaluates proposed treatment plans. If verified is True,
      routes control to Scribe for final report generation. If verified is False (contraindication detected),
      immediately halts workflow at END node to prevent unverified medical output.
    - Checkpointer Memory: MemorySaver persists state history per thread_id, enabling
      time-travel debugging and multi-turn clinical evaluation.

    Returns:
        CompiledStateGraph: Runnable LangGraph workflow engine with memory checkpointer.
    """
    logger.info("Initializing Agentic Medical RAG LangGraph state machine...")

    # Step 1: Instantiate specialized agent node processors
    historian = HistorianAgent()
    researcher = ResearcherAgent()
    skeptic = SkepticAgent()
    scribe = ScribeAgent()

    # Step 2: Initialize StateGraph with the AgentState schema
    workflow = StateGraph(AgentState)

    # Step 3: Register graph nodes corresponding to agent process methods
    workflow.add_node("historian", historian.process)
    workflow.add_node("researcher", researcher.process)
    workflow.add_node("skeptic", skeptic.process)
    workflow.add_node("scribe", scribe.process)

    # Step 4: Define direct execution edges (Historian -> Researcher -> Skeptic)
    workflow.set_entry_point("historian")
    workflow.add_edge("historian", "researcher")
    workflow.add_edge("researcher", "skeptic")

    # Step 5: Implement conditional routing based on Skeptic safety verification
    def routing_condition(state: Dict[str, Any]) -> str:
        """Evaluates Skeptic verification flag. Routes to Scribe if verified, else triggers Safety Halt."""
        if state.get("verified", False):
            return "scribe"
        logger.warning("Safety halt triggered: Skeptic agent flagged contraindications.")
        return END

    workflow.add_conditional_edges("skeptic", routing_condition, {"scribe": "scribe", END: END})

    # Step 6: Connect Scribe node to terminal END node
    workflow.add_edge("scribe", END)

    # Step 7: Compile workflow graph with in-memory checkpointer for thread persistence
    memory = MemorySaver()
    compiled_app = workflow.compile(checkpointer=memory)
    logger.info("LangGraph state machine workflow compiled successfully.")
    return compiled_app

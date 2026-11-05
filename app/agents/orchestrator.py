"""LangGraph stateful workflow orchestrator for Medical AI agents.

Architectural Mechanics:
- Strongly Typed AgentState: Coordinates clinical data across all specialized agents.
- Maximum Iteration Safeguard: Enforces max_iterations boundary to strictly prevent graph loops.
- Safety Routing Logic: Skeptic evaluates proposed treatment plans. Whether verified or flagged
  for contraindication/abstention, Scribe formats an authoritative, citation-grounded response.
- In-Memory Checkpointing: MemorySaver provides thread-isolated state inspection and debugging.
"""

from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.agents.historian import HistorianAgent
from app.agents.researcher import ResearcherAgent
from app.agents.skeptic import SkepticAgent
from app.agents.scribe import ScribeAgent
from app.utils.config import settings
from app.utils.logger import get_logger

logger = get_logger("Orchestrator")


class AgentState(TypedDict):
    """TypedDict defining state structure passed between agent nodes across graph executions."""

    patient_id: str
    patient_history: str
    clinical_trends: List[str]
    query: str
    retrieved_docs: List[str]
    retrieved_chunks: List[Dict[str, Any]]
    draft_briefing: str
    reasoning: str
    verified: bool
    abstained: bool
    abstention_reason: Optional[str]
    contradictions: List[str]
    safety_flags: List[str]
    final_report: str
    citations: List[str]
    validated_citations: List[Dict[str, Any]]
    iteration_count: int
    max_iterations: int


def create_medical_graph(
    retriever: Optional[Any] = None,
    max_iterations: int = settings.MAX_AGENT_ITERATIONS,
):
    """Build and compile the multi-agent state graph with clinical guardrails.

    Execution Flow:
    Entry -> Historian -> Researcher -> Skeptic -> Scribe -> END.
    """
    logger.info("Initializing Agentic Medical RAG LangGraph state machine...")

    # Step 1: Instantiate specialized agent node processors
    historian = HistorianAgent()
    researcher = ResearcherAgent(retriever=retriever)
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

    # Step 5: Implement iteration-guarded routing from Skeptic to Scribe
    def check_iteration_and_safety(state: Dict[str, Any]) -> str:
        iter_count = state.get("iteration_count", 0)
        max_iters = state.get("max_iterations", max_iterations)

        # Allow full multi-agent pass (4 nodes) before halting
        if iter_count > (max_iters * 4):
            logger.warning(f"Maximum graph step limit ({max_iters * 4}) exceeded. Routing to terminal END.")
            return END

        # Route to Scribe for report synthesis (either verified briefing or structured safety notice)
        return "scribe"

    workflow.add_conditional_edges(
        "skeptic",
        check_iteration_and_safety,
        {"scribe": "scribe", END: END},
    )

    # Step 6: Connect Scribe node to terminal END node
    workflow.add_edge("scribe", END)

    # Step 7: Compile workflow graph with in-memory checkpointer
    memory = MemorySaver()
    compiled_app = workflow.compile(checkpointer=memory)
    logger.info("LangGraph state machine workflow compiled successfully.")
    return compiled_app

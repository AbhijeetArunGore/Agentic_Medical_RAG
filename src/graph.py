from langgraph.graph import StateGraph, END
from src.state import AgentState
from src.agents.historian import HistorianAgent
from src.agents.researcher import ResearcherAgent
from src.agents.skeptic import SkepticAgent
from src.agents.scribe import ScribeAgent

from langgraph.checkpoint.memory import MemorySaver

def create_medical_graph():
    # 1. Initialize Agents
    historian = HistorianAgent()
    researcher = ResearcherAgent()
    skeptic = SkepticAgent()
    scribe = ScribeAgent()

    # 2. Define the Graph
    workflow = StateGraph(AgentState)

    # 3. Add Nodes
    workflow.add_node("historian", historian.process)
    workflow.add_node("researcher", researcher.process)
    workflow.add_node("skeptic", skeptic.process)
    workflow.add_node("scribe", scribe.process)

    # 4. Define Edges (The Industry-Grade Logic)
    workflow.set_entry_point("historian")
    workflow.add_edge("historian", "researcher")
    workflow.add_edge("researcher", "skeptic")

    # Conditional Edge: If verified -> Scribe, else -> END (or back to researcher)
    def should_continue(state):
        if state["verified"]:
            return "scribe"
        return END

    workflow.add_conditional_edges(
        "skeptic",
        should_continue,
        {
            "scribe": "scribe",
            END: END
        }
    )
    
    workflow.add_edge("scribe", END)

    # 5. Compile with Persistence (Checkpointer)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    app = create_medical_graph()
    print("LangGraph Medical Workflow Compiled Successfully.")

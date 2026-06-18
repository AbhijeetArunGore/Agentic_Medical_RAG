from src.graph import create_medical_graph
import uuid

def run_industry_grade_pipeline():
    print("=== STARTING INDUSTRY-GRADE AGENTIC MEDICAL RAG (LangGraph) ===")
    
    # Compile the graph
    app = create_medical_graph()
    
    # Initial State
    initial_state = {
        "patient_id": "PAT-2026-X88",
        "patient_history": "",
        "clinical_trends": [],
        "query": "Treatment for rising creatinine and high HbA1c",
        "retrieved_docs": [],
        "draft_briefing": "",
        "reasoning": "Starting diagnostic flow.",
        "verified": False,
        "contradictions": [],
        "final_report": "",
        "citations": []
    }
    
    # Execute the graph
    # thread_id for persistence (time travel)
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    
    for event in app.stream(initial_state, config):
        for node_name, state_update in event.items():
            print(f"\n[NODE COMPLETED: {node_name.upper()}]")
            if "reasoning" in state_update:
                print(f"Agent Reasoning: {state_update['reasoning']}")

    # Final Result
    final_state = app.get_state(config).values
    if final_state.get("final_report"):
        print("\n=== FINAL GENERATED BRIEFING ===")
        print(final_state["final_report"])
    else:
        print("\n=== PIPELINE TERMINATED: Safety Halt ===")
        print(f"Safety Logs: {final_state.get('contradictions')}")

if __name__ == "__main__":
    run_industry_grade_pipeline()

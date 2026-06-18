class ScribeAgent:
    def process(self, state):
        print("--- SCRIBE: Finalizing Clinical Briefing ---")
        
        final_report = f"""
        CLINICAL BRIEFING: {state['patient_id']}
        -----------------------------------
        TRENDS DETECTED:
        {', '.join(state['clinical_trends'])}

        RECOMMENDATIONS (VERIFIED):
        {state['draft_briefing']}

        SOURCES:
        {', '.join([f"Source {i+1}" for i in range(len(state['retrieved_docs']))])}
        """
        
        return {
            "final_report": final_report
        }

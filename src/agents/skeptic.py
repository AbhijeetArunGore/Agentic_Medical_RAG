class SkepticAgent:
    def process(self, state):
        print("--- SKEPTIC: Validating Clinical Safety ---")
        briefing = state.get("draft_briefing", "")
        
        # Mock safety check logic
        # In real world, this would use an LLM to find contradictions between patient history and recommendations
        verified = True
        contradictions = []
        
        if "ACE inhibitors" in briefing and "bilateral renal artery stenosis" in state.get("patient_history", ""):
            verified = False
            contradictions.append("Contraindication: ACE inhibitors should be avoided in bilateral renal artery stenosis.")
            
        print(f"Safety Check: {'PASSED' if verified else 'FAILED'}")
        
        reasoning = f"Cross-referenced proposed ACE inhibitors against patient history. {'No contraindications found.' if verified else f'CRITICAL SAFETY RISK FOUND: {contradictions[0]}'}"
        print(f"Skeptic Reasoning: {reasoning}")
        
        return {
            "verified": verified,
            "contradictions": contradictions,
            "reasoning": reasoning
        }

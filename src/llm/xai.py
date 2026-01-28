from .llm_interface import LLMInterface, MockLLM

class XAIExplainer:
    def __init__(self, llm: LLMInterface = None):
        self.llm = llm if llm else MockLLM()
        
    def explain_prediction(self, state_dict, action_label, true_label=None):
        """
        Generates an explanation for the agent's decision.
        """
        prompt = f"""
        Role: Cybersecurity Expert.
        Task: Explain why the IDS classified this flow as '{action_label}'.
        Context: 
        - Features: {state_dict}
        - True Label: {true_label if true_label else 'Unknown'}
        
        Provide a concise technical explanation.
        """
        return self.llm.generate(prompt)

from .llm_interface import LLMInterface, MockLLM
import numpy as np

class RewardShaper:
    def __init__(self, llm: LLMInterface = None):
        self.llm = llm if llm else MockLLM()
        self.history = []
        self.class_weights = {}
        
    def update_history(self, episode_stats):
        self.history.append(episode_stats)
        
    def compute_weights(self):
        """
        Analyzes history and returns new reward weights per class.
        """
        if len(self.history) < 5:
            return {}
            
        # Summarize recent performance
        recent = self.history[-5:]
        summary = f"Recent Precision: {[h['precision'] for h in recent]}"
        
        prompt = f"""
        Role: RL Optimization Expert.
        Task: Suggest reward scaling weights to improve detection of rare attacks.
        Data: {summary}
        Format: JSON mapping 'class_name': weight_float.
        """
        
        response = self.llm.generate(prompt)
        # Parse response (Mock logic here)
        # In real impl, we would parse JSON.
        
        # Mock adaptive logic
        return {'Backdoor': 1.5, 'Analysis': 1.2}

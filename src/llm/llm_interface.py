from abc import ABC, abstractmethod
import random
import time

class LLMInterface(ABC):
    @abstractmethod
    def generate(self, prompt, max_tokens=100):
        pass

class MockLLM(LLMInterface):
    """
    Simulates LLM responses for development/no-GPU environments.
    """
    def generate(self, prompt, max_tokens=100):
        # Simulate latency
        # time.sleep(0.1) 
        
        if "Explain" in prompt:
            return "Analysis: The traffic flow exhibits high packet rate (DoS signature) and anomalous port usage. Classified as DoS."
        elif "Reward" in prompt:
            return "Scaling: Increase penalty for False Negatives on 'Backdoor' class by 1.5x."
        elif "Adversarial" in prompt:
            return "Modification: Increase 'duration' by 0.5s and change 'proto' to TCP to mimic benign web traffic."
        else:
            return "LLM Response Placeholder"

class LocalLLM(LLMInterface):
    def __init__(self, model_name="microsoft/Phi-3-mini-4k-instruct"):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
        except ImportError:
            raise ImportError("Please install 'transformers', 'torch' and 'accelerate' to use LocalLLM.")
            
        print(f"Loading Local LLM: {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load Model & Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto", 
            trust_remote_code=True,
            attn_implementation="eager"
        )
        
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
        print("LLM Loaded Successfully.")
        
    def generate(self, prompt, max_tokens=100):
        # Format prompt for Phi-3 or generic instruct
        formatted_prompt = f"<|user|>\n{prompt}<|end|>\n<|assistant|>"
        
        output = self.pipe(formatted_prompt, max_new_tokens=max_tokens)
        generated_text = output[0]['generated_text']
        
        # Extract only the assistant response
        if "<|assistant|>" in generated_text:
            return generated_text.split("<|assistant|>")[-1].strip()
        return generated_text.strip()

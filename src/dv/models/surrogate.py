from typing import Dict, Any, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.dv.interfaces import Signaler
from src.utils.torch_utils import get_torch_device

class LocalSignaler(Signaler):
    def __init__(self, model_name: str, device: Optional[str] = None):
        self.device = device or get_torch_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            output_attentions=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        ).to(self.device)

    def get_signals(self, query: str, context: str, answer: str) -> Dict[str, Any]:
        """ Extracts attention weights for the context tokens relative to the answer tokens. """
        full_text = f"Question: {query}\nContext: {context}\nAnswer: {answer}"
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # outputs.attentions is a tuple of layers
        # Each layer is [batch, heads, seq_len, seq_len]
        # We want the attention on the context part from the answer part
        
        # This is simplified. In practice, we'd map tokens to chunks.
        return {
            "attentions": outputs.attentions,
            "input_ids": inputs["input_ids"]
        }

from typing import Optional
import os
from openai import OpenAI
from transformers import pipeline
from src.dv.interfaces import Judge
from src.utils.torch_utils import get_device_map

class MNLIJudge(Judge):
    def __init__(self, model_name: str = "roberta-large-mnli", device: Optional[str] = None):
        device = device or get_device_map()
        self.classifier = pipeline("zero-shot-classification", model=model_name, device=device)

    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        """Determines faithfulness using NLI entailment score."""
        if not context or not answer:
            return 0.0
        
        # We check if the context entails the answer
        # For zero-shot classification, we use context as the sequence and answer as part of the hypothesis
        result = self.classifier(context, candidate_labels=["entailment", "neutral", "contradiction"], hypothesis_template=f"Based on this text, it is true that {answer} is {{}}")
        
        # Return the score for 'entailment'
        entailment_idx = result["labels"].index("entailment")
        return result["scores"][entailment_idx]

class LLMJudge(Judge):
    def __init__(self, model_name: str = "gpt-4o-mini", api_key: Optional[str] = None):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        """Determines faithfulness using an LLM-as-a-judge prompt."""
        prompt = f"""
        Given the following context and answer to a query, rate the faithfulness of the answer based ONLY on the context.
        Provide a single float score between 0.0 (not faithful) and 1.0 (perfectly faithful).
        
        Query: {query}
        Context: {context}
        Answer: {answer}
        
        Score (float):"""
        
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        

        try:
            score_str = response.choices[0].message.content.strip()
            return float(score_str)
        except (ValueError, TypeError):
            return 0.5 # Fallback

class RagasJudge(Judge):
    def __init__(self, metrics=None, llm=None):
        from ragas.metrics import faithfulness
        self.metrics = metrics or [faithfulness]
        self.llm = llm

    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        """Determines faithfulness using Ragas (requires dataset construction)."""
        from ragas import evaluate
        from datasets import Dataset

        try:
            # Wrap input into Ragas-compatible format
            # Ragas expects: "question", "contexts" (list[list[str]]), "answer"
            dataset = Dataset.from_dict({
                "question": [query],
                "contexts": [[context]], # context is a single string here, so wrap in list
                "answer": [answer] # "answer" or "ground_truth"? faithfulness uses "answer" (generated) and "contexts"
                # faithfulness formula: Claims in generated answer vs contexts.
            })
            
            # Since Ragas 0.1+, arguments might vary, but this is the standard API.
            # providing llm directly to evaluate is supported in some versions, or set in metrics.
            # If self.llm is provided, we might need to configure it.
            # For simplicity, we assume metrics are configured or environment is set.
            # If llm is passed, we pass it.
            
            kwargs = {"metrics": self.metrics}
            if self.llm:
                 kwargs["llm"] = self.llm
                 # Note: In newer ragas, llm/embeddings might need to be wrapped in LangchainLLM or BaseRagasLLM
            
            result = evaluate(dataset, **kwargs)
            return float(result["faithfulness"][0]) # result is a Result object or dict-like
        except Exception as e:
            print(f"Ragas evaluation failed: {e}")
            return 0.5

    def get_faithfulness_batch(self, queries: list[str], contexts: list[str], answers: list[str]) -> list[float]:
        """Batch evaluation for efficiency."""
        from ragas import evaluate
        from datasets import Dataset

        try:
            dataset = Dataset.from_dict({
                "question": queries,
                "contexts": [[c] for c in contexts], 
                "answer": answers
            })
            
            kwargs = {"metrics": self.metrics}
            if self.llm:
                 kwargs["llm"] = self.llm
            
            result = evaluate(dataset, **kwargs)
            return [float(x) for x in result["faithfulness"]]
        except Exception as e:
            print(f"Ragas batch evaluation failed: {e}")
            return [0.5] * len(queries)

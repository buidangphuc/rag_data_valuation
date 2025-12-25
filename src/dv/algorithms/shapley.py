import itertools
import math
from typing import List
from src.dv.interfaces import Valuator, Judge
from src.dv.models.entities import Chunk, ValuationResult, ValuationMethod

class ShapleyValuator(Valuator):
    def __init__(self, judge: Judge, mc_samples: int = 100):
        self.judge = judge
        self.mc_samples = mc_samples

    def evaluate(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        n = len(chunks)
        if n <= 10:
            return self._evaluate_exact(query, chunks, answer)
        else:
            return self._evaluate_mc(query, chunks, answer)

    def _evaluate_exact(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        n = len(chunks)
        scores = {}
        
        # Prepare all queries/contexts/answers for batching if supported
        subset_definitions = []
        for r in range(n + 1):
            for subset_indices in itertools.combinations(range(n), r):
                subset_chunks = [chunks[i] for i in subset_indices]
                context = " ".join([c.text for c in subset_chunks])
                subset_definitions.append((subset_indices, context))
        
        if hasattr(self.judge, "get_faithfulness_batch"):
            contexts = [ctx for _, ctx in subset_definitions]
            queries = [query] * len(contexts)
            answers = [answer] * len(contexts)
            batch_scores = self.judge.get_faithfulness_batch(queries, contexts, answers)
            for (subset_indices, _), score in zip(subset_definitions, batch_scores):
                scores[subset_indices] = score
        else:
            for subset_indices, context in subset_definitions:
                scores[subset_indices] = self.judge.get_faithfulness(query, context, answer)
        
        shapley_values = [0.0] * n
        for i in range(n):
            for subset_indices in scores:
                if i not in subset_indices:
                    # S is a subset not containing i
                    # S_with_i is S + {i}
                    S = subset_indices
                    S_with_i = tuple(sorted(subset_indices + (i,)))
                    
                    weight = (math.factorial(len(S)) * math.factorial(n - len(S) - 1)) / math.factorial(n)
                    shapley_values[i] += weight * (scores[S_with_i] - scores[S])
        
        return [
            ValuationResult(chunk_id=chunks[i].id, method=ValuationMethod.SHAPLEY, score=shapley_values[i])
            for i in range(n)
        ]

    def _evaluate_mc(self, query: str, chunks: List[Chunk], answer: str) -> List[ValuationResult]:
        # Simple Monte Carlo approximation via random permutations
        import random
        n = len(chunks)
        shapley_values = [0.0] * n
        
        for _ in range(self.mc_samples):
            perm = list(range(n))
            random.shuffle(perm)
            
            prev_score = self.judge.get_faithfulness(query, "", answer) # Score of empty context
            current_context_list = []
            
            for idx in perm:
                current_context_list.append(chunks[idx].text)
                current_context = " ".join(current_context_list)
                current_score = self.judge.get_faithfulness(query, current_context, answer)
                
                shapley_values[idx] += (current_score - prev_score)
                prev_score = current_score
        
        return [
            ValuationResult(chunk_id=chunks[i].id, method=ValuationMethod.SHAPLEY, score=shapley_values[i] / self.mc_samples)
            for i in range(n)
        ]

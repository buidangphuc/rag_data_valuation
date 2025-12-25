from src.dv.models.entities import Chunk, ValuationMethod
from src.dv.algorithms.loo import LOOValuator
from src.dv.algorithms.shapley import ShapleyValuator
from src.dv.core import ValuationSuite
from src.dv.interfaces import Judge

class MockJudge(Judge):
    def get_faithfulness(self, query: str, context: str, answer: str) -> float:
        # Simple mock logic: faithfulness is proportional to number of words in context
        if not context:
            return 0.0
        return min(1.0, len(context.split()) / 20.0)

def test_valuation_suite_flow():
    # Setup
    judge = MockJudge()
    loo_valuator = LOOValuator(judge)
    shapley_valuator = ShapleyValuator(judge, mc_samples=10)
    
    suite = ValuationSuite({
        "loo": loo_valuator,
        "shapley": shapley_valuator
    })
    
    query = "What is the capital of France?"
    answer = "The capital of France is Paris."
    chunks = [
        Chunk(id="c1", text="Paris is the capital of France."),
        Chunk(id="c2", text="France is a country in Europe."),
        Chunk(id="c3", text="Random unrelated text here.")
    ]
    
    # Execute
    results = suite.evaluate_all(query, chunks, answer)
    
    # Verify
    assert len(results) == 6 # 3 chunks * 2 methods
    
    loo_results = [r for r in results if r.method == ValuationMethod.LOO]
    shapley_results = [r for r in results if r.method == ValuationMethod.SHAPLEY]
    
    assert len(loo_results) == 3
    assert len(shapley_results) == 3
    
    # Verify scores are present
    for r in results:
        assert isinstance(r.score, float)
    
    # Simple check: c1 should have positive value (contains the answer)
    c1_loo = next(r for r in loo_results if r.chunk_id == "c1")
    assert c1_loo.score > 0


import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.dv.models.entities import Chunk
from src.dv.algorithms.loo import LOOValuator
from src.dv.evaluation.judges import MNLIJudge
from src.dv.evaluation.filtering import filter_negative_chunks

def run_experiment():
    print("Running Hallucination Filtering Experiment...")
    # 1. Setup Data
    query = "What is the capital of France?"
    answer = "Paris is the capital of France."
    chunks = [
        Chunk(id="c1", text="Paris is the capital of France."),
        Chunk(id="c2", text="France is a country in Europe."),
        Chunk(id="c3", text="The moon is made of cheese.") # Hallucination source or irrelevant
    ]
    
    # 2. Valuation
    print("Valuating chunks...")
    # Using MNLIJudge for fast local execution without API keys
    try:
        judge = MNLIJudge() 
    except Exception as e:
        print(f"Could not initialize MNLIJudge: {e}")
        return

    valuator = LOOValuator(judge)
    results = valuator.evaluate(query, chunks, answer)
    
    for res in results:
        print(f"Chunk {res.chunk_id}: {res.score}")
        
    # 3. Filtering
    print("Filtering negative/low value chunks...")
    filtered_chunks = filter_negative_chunks(chunks, results, threshold=0.0)
    print(f"Original count: {len(chunks)}, Filtered count: {len(filtered_chunks)}")
    
    # 4. Measure Improvement (Simulated)
    # In a real scenario, we would re-generate the answer or re-evaluate faithfulness using the filtered context.
    # Here we check faithfulness of the EXISTING answer against the FILTERED context.
    
    original_context = " ".join([c.text for c in chunks])
    filtered_context = " ".join([c.text for c in filtered_chunks])
    
    score_original = judge.get_faithfulness(query, original_context, answer)
    score_filtered = judge.get_faithfulness(query, filtered_context, answer)
    
    print(f"Faithfulness (Original Context): {score_original}")
    print(f"Faithfulness (Filtered Context): {score_filtered}")
    
    improvement = score_filtered - score_original
    print(f"Improvement: {improvement}")
    
    if improvement >= 0:
        print("SUCCESS: Faithfulness maintained or improved.")
    else:
        print("WARNING: Faithfulness dropped.")

if __name__ == "__main__":
    run_experiment()

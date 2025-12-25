import argparse
import json
from src.dv.models.entities import Chunk, ExperimentRun
from src.dv.algorithms.loo import LOOValuator
from src.dv.evaluation.judges import MNLIJudge, LLMJudge
from src.dv.evaluation.filtering import filter_negative_chunks
from src.dv.core import ValuationSuite
from src.utils.io import save_json

def run_experiment(query: str, chunks_file: str, answer: str, judge_type: str = "mnli"):
    # Load chunks
    with open(chunks_file, "r") as f:
        chunks_data = json.load(f)
    chunks = [Chunk(id=c["id"], text=c["text"], metadata=c.get("metadata", {})) for c in chunks_data]
    
    # Initialize Judge
    judge = MNLIJudge() if judge_type == "mnli" else LLMJudge()
    
    # Initialize Valuators
    loo = LOOValuator(judge)
    suite = ValuationSuite({"loo": loo})
    
    # Initial Evaluation
    results = suite.evaluate_all(query, chunks, answer)
    initial_faithfulness = judge.get_faithfulness(query, " ".join([c.text for c in chunks]), answer)
    
    # Filter Chunks
    filtered_chunks = filter_negative_chunks(chunks, results)
    post_filter_faithfulness = judge.get_faithfulness(query, " ".join([c.text for c in filtered_chunks]), answer)
    
    # Save Run
    run = ExperimentRun(
        query=query,
        retrieved_chunks=chunks,
        generated_answer=answer,
        ground_truth_faithfulness=initial_faithfulness,
        valuation_reports=results
    )
    
    run_dir = f"experiments/{run.id}"
    save_json(run.__dict__, f"{run_dir}/metadata.json")
    
    print(f"Experiment {run.id} completed.")
    print(f"Initial Faithfulness: {initial_faithfulness:.4f}")
    print(f"Post-Filter Faithfulness: {post_filter_faithfulness:.4f}")
    print(f"Improvement: {(post_filter_faithfulness - initial_faithfulness):.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True)
    parser.add_argument("--chunks-file", required=True)
    parser.add_argument("--answer", required=True)
    parser.add_argument("--judge", default="mnli")
    args = parser.parse_args()
    
    run_experiment(args.query, args.chunks_file, args.answer, args.judge)

import argparse
import json
from src.dv.core import ValuationSuite
from src.dv.algorithms.loo import LOOValuator
from src.dv.algorithms.shapley import ShapleyValuator
from src.dv.evaluation.judges import MNLIJudge, LLMJudge
from src.dv.models.entities import Chunk
from src.utils.io import save_valuation_results_csv

def main():
    parser = argparse.ArgumentParser(prog="rag-dv")
    subparsers = parser.add_subparsers(dest="command", help="sub-command help")

    # evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate chunks for a query/answer")
    eval_parser.add_argument("--query", required=True)
    eval_parser.add_argument("--chunks-file", required=True)
    eval_parser.add_argument("--answer", required=True)
    eval_parser.add_argument("--methods", default="loo", help="Comma-separated methods (loo, shapley)")
    eval_parser.add_argument("--judge", default="mnli", choices=["mnli", "llm"])

    args = parser.parse_args()

    if args.command == "evaluate":
        # Load chunks
        with open(args.chunks_file, "r") as f:
            chunks_data = json.load(f)
        chunks = [Chunk(id=c["id"], text=c["text"], metadata=c.get("metadata", {})) for c in chunks_data]
        
        # Initialize Judge
        judge = MNLIJudge() if args.judge == "mnli" else LLMJudge()
        
        # Initialize Valuators
        methods = args.methods.split(",")
        valuators = {}
        if "loo" in methods:
            valuators["loo"] = LOOValuator(judge)
        if "shapley" in methods:
            valuators["shapley"] = ShapleyValuator(judge)
            
        suite = ValuationSuite(valuators)
        
        # Execute
        results = suite.evaluate_all(args.query, chunks, args.answer)
        
        # Output
        save_valuation_results_csv(results, "experiments/latest/scores.csv")
        print("Evaluation complete. Results saved to experiments/latest/scores.csv")
        for res in results:
            print(f"Chunk: {res.chunk_id} | Method: {res.method.value} | Score: {res.score:.4f}")

if __name__ == "__main__":
    main()

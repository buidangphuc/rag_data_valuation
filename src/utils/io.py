import json
import csv
import os
from typing import List, Any
from ..dv.models.entities import ValuationResult

def save_json(data: Any, filepath: str):
    """Saves data to a JSON file, creating directories if needed."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, default=str)

def load_json(filepath: str) -> Any:
    """Loads data from a JSON file."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_valuation_results_csv(results: List[ValuationResult], filepath: str):
    """Saves valuation results to a CSV file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    if not results:
        return
    
    keys = ["chunk_id", "method", "score", "timestamp"]
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for res in results:
            writer.writerow({
                "chunk_id": res.chunk_id,
                "method": res.method.value,
                "score": res.score,
                "timestamp": res.timestamp.isoformat()
            })

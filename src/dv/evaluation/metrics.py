from typing import List
from scipy.stats import kendalltau

def calculate_agreement_kendall(scores1: List[float], scores2: List[float]) -> float:
    """Calculates Kendall's Tau correlation between two sets of scores."""
    if len(scores1) != len(scores2) or len(scores1) < 2:
        return 0.0
    
    correlation, _ = kendalltau(scores1, scores2)
    return correlation

def calculate_mrr(relevance_labels: List[int]) -> float:
    """Calculates Mean Reciprocal Rank."""
    for i, rel in enumerate(relevance_labels):
        if rel > 0:
            return 1.0 / (i + 1)
    return 0.0

def calculate_mae(y_true: List[float], y_pred: List[float]) -> float:
    """Calculates Mean Absolute Error between two sets of scores."""
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        return 0.0
    
    # Use numpy if available for speed, else loop
    try:
        import numpy as np
        return float(np.mean(np.abs(np.array(y_true) - np.array(y_pred))))
    except ImportError:
        error_sum = sum(abs(t - p) for t, p in zip(y_true, y_pred))
        return error_sum / len(y_true)

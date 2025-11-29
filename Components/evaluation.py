"""
Evaluation Metrics for Information Retrieval

Based on: "Introduction to Information Retrieval" by Manning, Raghavan & Schütze
Chapter 8: Evaluation in Information Retrieval

This module provides standard IR evaluation metrics for Approximate Nearest Neighbor (ANN) search.
In our ANN context:
- "Relevant documents" = the k exact nearest neighbors (ground truth from brute-force search)
- R_total = k (there are exactly k "relevant" documents)
- Therefore: Precision@k = Recall@k in our specific case
"""

import numpy as np
from typing import List, Dict, Tuple, Any


# =============================================================================
# BASIC METRICS (Chapter 8.1-8.3)
# =============================================================================

def compute_precision_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """
    Precision@k: Fraction of retrieved documents that are relevant.
    
    Formula: P@k = |relevant ∩ retrieved_top_k| / k
    
    Reference: Manning et al., Section 8.4
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        k: Number of top results to consider
        
    Returns:
        Precision@k value in [0, 1]
    """
    predicted_set = set(predicted[:k])
    actual_set = set(actual[:k])
    return len(predicted_set & actual_set) / k


def compute_recall_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """
    Recall@k: Fraction of relevant documents that are retrieved.
    
    Formula: R@k = |relevant ∩ retrieved_top_k| / R_total
    
    In ANN search, R_total = k (the k exact neighbors), so R@k = P@k.
    
    Reference: Manning et al., Section 8.4
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        k: Number of top results to consider
        
    Returns:
        Recall@k value in [0, 1]
    """
    predicted_set = set(predicted[:k])
    actual_set = set(actual[:k])
    # In ANN context, R_total = k (we define the k nearest as "all relevant")
    return len(predicted_set & actual_set) / k


def compute_f1_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """
    F1@k: Harmonic mean of Precision@k and Recall@k.
    
    Formula: F1 = 2 * P * R / (P + R)
    
    In ANN search where P@k = R@k, F1@k = P@k = R@k.
    
    Reference: Manning et al., Section 8.3
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        k: Number of top results to consider
        
    Returns:
        F1@k value in [0, 1]
    """
    p = compute_precision_at_k(predicted, actual, k)
    r = compute_recall_at_k(predicted, actual, k)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


# =============================================================================
# RANKED METRICS (Chapter 8.4)
# =============================================================================

def compute_average_precision(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Average Precision (AP): Mean precision at each relevant position.
    
    Formula: AP = (1/R) * Σ P(k) * rel(k)
    
    Where:
    - R = number of relevant documents
    - P(k) = precision at position k
    - rel(k) = 1 if document at position k is relevant, 0 otherwise
    
    Reference: Manning et al., Section 8.4
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        
    Returns:
        Average Precision value in [0, 1]
    """
    relevant = set(actual)
    precisions = []
    hits = 0
    
    for i, doc in enumerate(predicted):
        if doc in relevant:
            hits += 1
            precisions.append(hits / (i + 1))
    
    if not precisions:
        return 0.0
    return np.mean(precisions)


def compute_map(all_predictions: List[np.ndarray], all_actuals: List[np.ndarray]) -> float:
    """
    Mean Average Precision (MAP): Mean of AP over all queries.
    
    Formula: MAP = (1/Q) * Σ AP(q)
    
    Where Q is the number of queries.
    
    Reference: Manning et al., Section 8.4
    
    Args:
        all_predictions: List of predicted arrays for each query
        all_actuals: List of actual arrays for each query
        
    Returns:
        MAP value in [0, 1]
    """
    aps = [compute_average_precision(p, a) for p, a in zip(all_predictions, all_actuals)]
    return np.mean(aps) if aps else 0.0


def compute_dcg_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """
    Discounted Cumulative Gain at k.
    
    Formula: DCG@k = Σ rel_i / log2(i + 2)
    
    Where rel_i = 1 if document at position i is in actual top-k, else 0.
    
    Reference: Manning et al., Section 8.4
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        k: Number of top results to consider
        
    Returns:
        DCG@k value
    """
    actual_set = set(actual[:k])
    dcg = sum([1 / np.log2(i + 2) for i, doc in enumerate(predicted[:k]) if doc in actual_set])
    return dcg


def compute_ndcg_at_k(predicted: np.ndarray, actual: np.ndarray, k: int) -> float:
    """
    Normalized Discounted Cumulative Gain at k.
    
    Formula: nDCG@k = DCG@k / IDCG@k
    
    Where IDCG@k is the ideal DCG (when ranking is perfect).
    
    Reference: Manning et al., Section 8.4
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        k: Number of top results to consider
        
    Returns:
        nDCG@k value in [0, 1]
    """
    dcg = compute_dcg_at_k(predicted, actual, k)
    # Ideal DCG: all relevant documents at top positions
    idcg = sum([1 / np.log2(i + 2) for i in range(min(len(actual), k))])
    
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_reciprocal_rank(predicted: np.ndarray, actual: np.ndarray) -> float:
    """
    Reciprocal Rank: 1 / position of first relevant document.
    
    Reference: Manning et al., Section 8.4
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        
    Returns:
        Reciprocal Rank value in (0, 1]
    """
    actual_set = set(actual)
    for i, doc in enumerate(predicted):
        if doc in actual_set:
            return 1.0 / (i + 1)
    return 0.0


def compute_mrr(all_predictions: List[np.ndarray], all_actuals: List[np.ndarray]) -> float:
    """
    Mean Reciprocal Rank (MRR): Mean of RR over all queries.
    
    Formula: MRR = (1/Q) * Σ RR(q)
    
    Reference: Manning et al., Section 8.4
    
    Args:
        all_predictions: List of predicted arrays for each query
        all_actuals: List of actual arrays for each query
        
    Returns:
        MRR value in (0, 1]
    """
    rrs = [compute_reciprocal_rank(p, a) for p, a in zip(all_predictions, all_actuals)]
    return np.mean(rrs) if rrs else 0.0


# =============================================================================
# PRECISION-RECALL CURVE (Chapter 8.4)
# =============================================================================

def compute_precision_recall_curve(predicted: np.ndarray, actual: np.ndarray, 
                                    max_k: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve by varying k.
    
    Reference: Manning et al., Section 8.4, Figure 8.1
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        max_k: Maximum k to consider (default: length of predicted)
        
    Returns:
        Tuple of (recall_values, precision_values) arrays
    """
    if max_k is None:
        max_k = len(predicted)
    
    actual_set = set(actual)
    R_total = len(actual_set)
    
    recalls = []
    precisions = []
    hits = 0
    
    for k in range(1, max_k + 1):
        if predicted[k-1] in actual_set:
            hits += 1
        recalls.append(hits / R_total)
        precisions.append(hits / k)
    
    return np.array(recalls), np.array(precisions)


def compute_interpolated_precision_at_recall(recalls: np.ndarray, precisions: np.ndarray,
                                              recall_levels: np.ndarray = None) -> np.ndarray:
    """
    Interpolated precision at standard recall levels.
    
    The interpolated precision at recall level r is the maximum precision
    at any recall level >= r.
    
    Reference: Manning et al., Section 8.4
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        recall_levels: Standard recall levels (default: 0.0, 0.1, ..., 1.0)
        
    Returns:
        Array of interpolated precision values
    """
    if recall_levels is None:
        recall_levels = np.linspace(0, 1, 11)
    
    interpolated = []
    for r in recall_levels:
        # Max precision at recall >= r
        mask = recalls >= r
        if np.any(mask):
            interpolated.append(np.max(precisions[mask]))
        else:
            interpolated.append(0.0)
    
    return np.array(interpolated)


# =============================================================================
# EFFICIENCY METRICS (for ANN benchmarking)
# =============================================================================

def compute_candidate_ratio(n_candidates: int, n_total: int) -> float:
    """
    Candidate Ratio: Fraction of corpus examined.
    
    Lower is better (more efficient).
    
    Args:
        n_candidates: Number of candidate documents examined
        n_total: Total number of documents in corpus
        
    Returns:
        Candidate ratio in [0, 1]
    """
    return n_candidates / n_total


def compute_speedup(baseline_time: float, method_time: float) -> float:
    """
    Speedup: How much faster a method is compared to baseline.
    
    Args:
        baseline_time: Time for baseline method (e.g., brute-force)
        method_time: Time for the method being evaluated
        
    Returns:
        Speedup factor (> 1 means faster)
    """
    if method_time == 0:
        return float('inf')
    return baseline_time / method_time


# =============================================================================
# COMPREHENSIVE EVALUATION
# =============================================================================

def evaluate_retrieval(predicted: np.ndarray, actual: np.ndarray, k: int = 10,
                       query_time: float = None, n_candidates: int = None,
                       n_total: int = None) -> Dict[str, float]:
    """
    Comprehensive evaluation of a single query result.
    
    Args:
        predicted: Array of predicted document indices (ranked)
        actual: Array of actual relevant document indices (ground truth)
        k: Number of top results to consider
        query_time: Optional query execution time
        n_candidates: Optional number of candidates examined
        n_total: Optional total corpus size
        
    Returns:
        Dictionary with all computed metrics
    """
    results = {
        'precision_at_k': compute_precision_at_k(predicted, actual, k),
        'recall_at_k': compute_recall_at_k(predicted, actual, k),
        'f1_at_k': compute_f1_at_k(predicted, actual, k),
        'ndcg_at_k': compute_ndcg_at_k(predicted, actual, k),
        'average_precision': compute_average_precision(predicted, actual),
        'reciprocal_rank': compute_reciprocal_rank(predicted, actual),
    }
    
    if query_time is not None:
        results['query_time'] = query_time
    
    if n_candidates is not None and n_total is not None:
        results['candidate_ratio'] = compute_candidate_ratio(n_candidates, n_total)
    
    return results


def evaluate_batch(all_predictions: List[np.ndarray], all_actuals: List[np.ndarray],
                   k: int = 10, query_times: List[float] = None) -> Dict[str, float]:
    """
    Evaluate a batch of queries and compute aggregate metrics.
    
    Args:
        all_predictions: List of predicted arrays for each query
        all_actuals: List of actual arrays for each query
        k: Number of top results to consider
        query_times: Optional list of query times
        
    Returns:
        Dictionary with mean metrics across all queries
    """
    n_queries = len(all_predictions)
    
    # Compute per-query metrics
    metrics = {
        'precision_at_k': [],
        'recall_at_k': [],
        'f1_at_k': [],
        'ndcg_at_k': [],
        'average_precision': [],
        'reciprocal_rank': [],
    }
    
    for pred, actual in zip(all_predictions, all_actuals):
        metrics['precision_at_k'].append(compute_precision_at_k(pred, actual, k))
        metrics['recall_at_k'].append(compute_recall_at_k(pred, actual, k))
        metrics['f1_at_k'].append(compute_f1_at_k(pred, actual, k))
        metrics['ndcg_at_k'].append(compute_ndcg_at_k(pred, actual, k))
        metrics['average_precision'].append(compute_average_precision(pred, actual))
        metrics['reciprocal_rank'].append(compute_reciprocal_rank(pred, actual))
    
    # Aggregate
    results = {
        'mean_precision_at_k': np.mean(metrics['precision_at_k']),
        'mean_recall_at_k': np.mean(metrics['recall_at_k']),
        'mean_f1_at_k': np.mean(metrics['f1_at_k']),
        'mean_ndcg_at_k': np.mean(metrics['ndcg_at_k']),
        'map': np.mean(metrics['average_precision']),
        'mrr': np.mean(metrics['reciprocal_rank']),
        'n_queries': n_queries,
    }
    
    if query_times is not None:
        results['mean_query_time'] = np.mean(query_times)
        results['total_query_time'] = np.sum(query_times)
    
    return results


# =============================================================================
# VISUALIZATION HELPERS
# =============================================================================

def plot_precision_recall_curve(recalls: np.ndarray, precisions: np.ndarray,
                                 ax=None, label: str = None, **kwargs):
    """
    Plot a Precision-Recall curve.
    
    Args:
        recalls: Array of recall values
        precisions: Array of precision values
        ax: Matplotlib axis (if None, uses current axis)
        label: Legend label
        **kwargs: Additional arguments for plt.plot()
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
    
    ax.plot(recalls, precisions, label=label, **kwargs)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title('Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    
    if label:
        ax.legend()


def plot_interpolated_precision_recall(recall_levels: np.ndarray, 
                                        interpolated_precisions: np.ndarray,
                                        ax=None, label: str = None, **kwargs):
    """
    Plot interpolated Precision-Recall curve (11-point).
    
    Reference: Manning et al., Section 8.4, Figure 8.2
    
    Args:
        recall_levels: Standard recall levels (0.0 to 1.0)
        interpolated_precisions: Interpolated precision at each level
        ax: Matplotlib axis (if None, uses current axis)
        label: Legend label
        **kwargs: Additional arguments for plt.step()
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
    
    ax.step(recall_levels, interpolated_precisions, where='post', label=label, **kwargs)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Interpolated Precision')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_title('11-Point Interpolated Precision-Recall Curve')
    ax.grid(True, alpha=0.3)
    
    if label:
        ax.legend()


import numpy as np

try:
    import ot
except ImportError:
    ot = None

# ------------------------------
# Metrics
# ------------------------------
def compute_metrics_from_scores(scores: np.ndarray, hot_coded_correct: np.ndarray, ks=(1, 3, 5)):
    """
    Computes retrieval metrics (Acc@1, Hit@K, MRR) given similarity scores and a binary truth array.
    - scores: 1D array of similarity scores for all candidates.
    - hot_coded_correct: 1D binary array (1 for correct matches, 0 for incorrect).
    - ks: Tuple of K values for Hit@K metrics.
    
    returns:
        - out: Dictionary of computed metrics.
        - order: Indices of candidates sorted by descending similarity.
        - rank: The rank position of the *first* correct match (1-based). Returns None if no match.
    """
    order = np.argsort(-scores)
    ranked_hits = hot_coded_correct[order]
    correct_ranks = np.nonzero(ranked_hits)[0]
    
    out = {}
    
    if len(correct_ranks) == 0:
        for k in ks:
            out["Acc" if k == 1 else f"Hit@{k}"] = 0.0
        out["MRR"] = 0.0
        return out, order, None
        
    rank = int(correct_ranks[0]) + 1
    
    for k in ks:
        out["Acc" if k == 1 else f"Hit@{k}"] = float(rank <= k)
    
    out["MRR"] = 1.0 / rank
    
    return out, order, rank
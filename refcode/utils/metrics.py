"""Metric helpers for code-search retrieval evaluation."""

import numpy as np


def mrr_from_ranks(ranks):
    """Compute mean reciprocal rank from one-indexed rank positions."""
    return float(np.mean([1.0 / r if r > 0 else 0.0 for r in ranks]))


def recall_from_ranks(ranks, k):
    """Compute Recall@K from one-indexed rank positions."""
    return float(np.mean([1.0 if r > 0 and r <= k else 0.0 for r in ranks]))

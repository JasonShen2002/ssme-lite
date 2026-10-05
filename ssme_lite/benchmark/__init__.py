"""Reproducible comparisons, with labels hidden before fitting the evaluator."""
from .datasets import gaussian_scores, official_scores, synthetic_scores, ten_models
from .runner import benchmark, plot_comparison, selection_stats

__all__ = [
    "benchmark",
    "gaussian_scores",
    "official_scores",
    "plot_comparison",
    "selection_stats",
    "synthetic_scores",
    "ten_models",
]

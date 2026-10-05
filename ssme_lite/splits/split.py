"""Evaluation-pool splits shared by every dataset experiment."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SemiSupervisedSplit:
    """Labeled estimation rows, unlabeled estimation rows, and held-out evaluation.

    The draw is one uniform permutation of the evaluation pool. Labels are not
    used to repair a draw that misses a class. The estimation pool is the
    permutation prefix, with labeled rows first, so a larger ``n_labeled`` on
    the same seed extends the labeled prefix.
    """

    n_labeled: int
    n_unlabeled: int
    random_state: int = 0

    def __post_init__(self):
        if isinstance(self.n_labeled, bool) or isinstance(self.n_unlabeled, bool):
            raise ValueError("sample counts must be positive integers")
        if int(self.n_labeled) != self.n_labeled or int(self.n_unlabeled) != self.n_unlabeled:
            raise ValueError("sample counts must be positive integers")
        if self.n_labeled < 1 or self.n_unlabeled < 1:
            raise ValueError("sample counts must be positive integers")
        object.__setattr__(self, "n_labeled", int(self.n_labeled))
        object.__setattr__(self, "n_unlabeled", int(self.n_unlabeled))
        object.__setattr__(self, "random_state", int(self.random_state))

    def indices(self, n_samples):
        n = int(n_samples)
        if self.n_labeled + self.n_unlabeled >= n:
            raise ValueError("reserve a nonempty held-out evaluation set")
        order = np.random.default_rng(self.random_state).permutation(n)
        labeled_end = self.n_labeled
        estimation_end = self.n_labeled + self.n_unlabeled
        return {
            "labeled": order[:labeled_end],
            "unlabeled": order[labeled_end:estimation_end],
            "estimation": order[:estimation_end],
            "heldout": order[estimation_end:],
        }

    def partial_labels(self, y, parts):
        """Labels aligned with the estimation pool: observed prefix, then -1."""
        y = np.asarray(y)
        estimation = np.asarray(parts["estimation"])
        labeled = np.asarray(parts["labeled"])
        if estimation.shape != (self.n_labeled + self.n_unlabeled,):
            raise ValueError("estimation pool has an unexpected length")
        if not np.array_equal(estimation[: self.n_labeled], labeled):
            raise ValueError("estimation pool must start with the labeled rows")
        partial = np.full(len(estimation), -1, dtype=int)
        partial[: self.n_labeled] = y[labeled]
        return partial

    def labeled_class_count(self, y, parts):
        return int(len(np.unique(np.asarray(y)[parts["labeled"]])))

"""SSME-Lite: evaluation from shared classifier probabilities."""
from .prediction import PredictionMatrixGenerator
from .estimator import SSMEEstimator
from .report import EvaluationReport
from .splits import SemiSupervisedSplit

__all__ = ["PredictionMatrixGenerator", "SSMEEstimator", "EvaluationReport", "SemiSupervisedSplit"]
__version__ = "0.1.0"


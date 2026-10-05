"""SSME-Lite: evaluation from shared classifier probabilities."""
from .estimation import SSMEEstimator
from .prediction import PredictionMatrixGenerator
from .reporting import EvaluationReport
from .splits import SemiSupervisedSplit

__all__ = ["PredictionMatrixGenerator", "SSMEEstimator", "EvaluationReport", "SemiSupervisedSplit"]
__version__ = "0.1.0"


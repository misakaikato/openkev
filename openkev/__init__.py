"""openkev — calibration and decision-quality evaluation for typed-decision models."""

from .calibrate import TemperatureScaler, fit_temperature, holdout_split
from .metrics import Report, brier, ece, evaluate, reliability, risk_coverage, select_threshold
from .scorer import LETTERS, MLXScorer, Scorer

__version__ = "0.1.0"
__all__ = [
	"LETTERS",
	"MLXScorer",
	"Report",
	"Scorer",
	"TemperatureScaler",
	"brier",
	"ece",
	"evaluate",
	"fit_temperature",
	"holdout_split",
	"reliability",
	"risk_coverage",
	"select_threshold",
]

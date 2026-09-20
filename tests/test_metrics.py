import numpy as np
import pytest

from openkev import brier, ece, evaluate, reliability, risk_coverage
from openkev.metrics import as_probabilities


def test_ece_is_near_zero_for_a_calibrated_predictor():
	rng = np.random.default_rng(0)
	conf = rng.uniform(0.5, 1.0, size=20000)
	correct = rng.uniform(size=20000) < conf
	assert ece(conf, correct) < 0.02


def test_ece_is_large_for_a_constantly_overconfident_predictor():
	rng = np.random.default_rng(0)
	correct = rng.uniform(size=20000) < 0.6
	assert ece(np.full(20000, 0.99), correct) > 0.3


def test_brier_bounds():
	P = np.array([[1.0, 0.0], [0.0, 1.0]])
	assert brier(P, [0, 1]) == 0.0
	assert brier(P, [1, 0]) == 2.0


def test_reliability_bins_partition_the_rows():
	conf = np.array([0.95, 0.85, 0.6, 0.2])
	correct = np.array([True, True, False, False])
	bins = reliability(conf, correct)
	assert [b.count for b in bins] == [1, 1, 1, 1]
	assert bins[0].accuracy == 1.0


def test_risk_coverage_improves_when_confidence_ranks_errors_last():
	conf = np.linspace(1.0, 0.0, 100)
	correct = conf > 0.3  # the least confident rows are the wrong ones
	points = {p.coverage: p.accuracy for p in risk_coverage(conf, correct)}
	assert points[1.0] < points[0.9] < points[0.7] <= points[0.5] == 1.0


def test_evaluate_reports_accuracy_and_shape():
	P = np.array([[0.9, 0.1], [0.2, 0.8], [0.6, 0.4]])
	report = evaluate(P, [0, 1, 1])
	assert report.n == 3 and report.n_classes == 2
	assert report.accuracy == pytest.approx(2 / 3)
	assert "acc=" in str(report)


@pytest.mark.parametrize(
	"bad",
	[
		np.array([[0.5, 0.4]]),			  # does not sum to 1
		np.array([[1.0]]),				  # fewer than two options
		np.array([[0.5, 0.5, np.nan]]),	  # not finite
		np.array([[1.5, -0.5]]),		  # negative entry
	],
)
def test_as_probabilities_rejects_malformed_input(bad):
	with pytest.raises(ValueError):
		as_probabilities(bad)


def test_evaluate_rejects_mismatched_labels():
	with pytest.raises(ValueError):
		evaluate(np.array([[0.5, 0.5]]), [0, 1])

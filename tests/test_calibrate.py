import numpy as np
import pytest

from openkev import TemperatureScaler, ece, evaluate, fit_temperature, holdout_split
from openkev.calibrate import _scale


def test_fit_recovers_a_known_sharpening_factor(overconfident):
	P, y, sharpness = overconfident()
	assert fit_temperature(P, y) == pytest.approx(sharpness, rel=0.25)


def test_scaling_reduces_calibration_error(overconfident):
	P, y, _ = overconfident()
	fit, test = holdout_split(len(y))
	scaler = TemperatureScaler()
	scaler.fit(P[fit], y[fit])
	before = evaluate(P[test], y[test])
	after = evaluate(scaler.transform(P[test]), y[test])
	assert after.ece < before.ece / 2
	assert after.accuracy == before.accuracy  # temperature never moves the argmax


def test_temperature_above_one_lowers_confidence():
	P = np.array([[0.99, 0.01]])
	assert _scale(P, 3.0)[0, 0] < 0.99
	assert _scale(P, 0.5)[0, 0] > 0.99


def test_keys_separate_question_shapes():
	scaler = TemperatureScaler()
	scaler.temperatures[TemperatureScaler.key("choice", 2)] = 2.0
	assert "choice:2" in scaler
	assert "choice:15" not in scaler


def test_unknown_key_passes_through_with_a_warning():
	scaler = TemperatureScaler()
	P = np.array([[0.7, 0.3]])
	with pytest.warns(UserWarning):
		assert np.allclose(scaler.transform(P), P)
	with pytest.raises(KeyError):
		scaler.transform(P, strict=True)


def test_save_and_load_roundtrip(tmp_path, overconfident):
	P, y, _ = overconfident(n=200)
	scaler = TemperatureScaler()
	T = scaler.fit(P, y)
	path = tmp_path / "calibration.json"
	scaler.save(path)
	assert TemperatureScaler.load(path).temperatures == {"choice:3": T}


def test_small_fits_warn(overconfident):
	P, y, _ = overconfident(n=20)
	with pytest.warns(UserWarning):
		fit_temperature(P, y)


def test_invalid_temperature_rejected():
	with pytest.raises(ValueError):
		_scale(np.array([[0.5, 0.5]]), 0.0)


def test_holdout_split_is_disjoint_and_complete():
	fit, test = holdout_split(10, 0.3)
	assert len(fit) == 3 and len(test) == 7
	assert set(fit).isdisjoint(test)

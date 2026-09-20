import numpy as np
import pytest


def softmax(L: np.ndarray) -> np.ndarray:
	L = L - L.max(axis=1, keepdims=True)
	E = np.exp(L)
	return E / E.sum(axis=1, keepdims=True)


@pytest.fixture
def overconfident():
	"""Reported probabilities that are exactly `sharpness` times too sharp.

	The true generating distribution is softmax(logits); the backend reports
	softmax(logits * sharpness). The optimal temperature is therefore `sharpness`,
	which gives the calibration tests a known answer to hit.
	"""

	def build(n: int = 4000, k: int = 3, sharpness: float = 3.0, seed: int = 0):
		rng = np.random.default_rng(seed)
		logits = rng.normal(size=(n, k))
		truth = softmax(logits)
		y = np.array([rng.choice(k, p=row) for row in truth])
		return softmax(logits * sharpness), y, sharpness

	return build

"""Temperature scaling for typed-decision models.

Reading option logits out of a model gives probabilities that are conditional on
the options you supplied. They are usually far too confident to threshold on.
Temperature scaling fixes that with a single scalar per question shape, fitted on
a few hundred labelled rows, and it cannot change which option wins.

Dividing log-probabilities by T and re-normalising is exactly temperature scaling
on the original logits: log p = logit - logsumexp(logits), and the constant term
cancels in the softmax.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .metrics import as_labels, as_probabilities

MIN_FIT_ROWS = 50
GRID = (0.25, 12.0, 0.05)
EPS = 1e-9


def _scale(P: np.ndarray, T: float) -> np.ndarray:
	if T <= 0:
		raise ValueError(f"temperature must be positive, got {T}")
	L = np.log(np.clip(P, EPS, None)) / T
	L -= L.max(axis=1, keepdims=True)
	E = np.exp(L)
	return E / E.sum(axis=1, keepdims=True)


def _nll(P: np.ndarray, y: np.ndarray, T: float) -> float:
	Q = _scale(P, T)
	return float(-np.log(np.clip(Q[np.arange(len(y)), y], EPS, None)).mean())


def fit_temperature(
	P: Sequence[Sequence[float]] | np.ndarray,
	y: Sequence[int],
	*,
	grid: tuple[float, float, float] = GRID,
) -> float:
	"""Return the temperature minimising negative log-likelihood.

	Coarse grid followed by a local refinement, which is deterministic and needs
	no optimiser dependency. T > 1 means the model was over-confident.
	"""
	A = as_probabilities(P)
	v = as_labels(y, A.shape[1])
	if len(v) != len(A):
		raise ValueError(f"got {len(A)} rows of probabilities but {len(v)} labels")
	if len(v) < MIN_FIT_ROWS:
		warnings.warn(
			f"fitting a temperature on {len(v)} rows; below {MIN_FIT_ROWS} the estimate is noisy",
			stacklevel=2,
		)
	lo, hi, step = grid
	coarse = np.arange(lo, hi + step / 2, step)
	best = min(coarse, key=lambda T: _nll(A, v, float(T)))
	fine = np.arange(max(lo, best - step), best + step + step / 20, step / 10)
	return round(float(min(fine, key=lambda T: _nll(A, v, float(T)))), 4)


class TemperatureScaler:
	"""A temperature per question shape.

	Calibration does not transfer between question shapes: a two-way choice and a
	fifteen-way choice are over-confident by different amounts, so each gets its
	own scalar. `key` defaults to "<kind>:<n_options>".
	"""

	def __init__(self, temperatures: dict[str, float] | None = None) -> None:
		self.temperatures: dict[str, float] = dict(temperatures or {})

	@staticmethod
	def key(kind: str, n_options: int) -> str:
		return f"{kind}:{n_options}"

	def __contains__(self, key: str) -> bool:
		return key in self.temperatures

	def __repr__(self) -> str:
		return f"TemperatureScaler({self.temperatures!r})"

	def fit(
		self,
		P: Sequence[Sequence[float]] | np.ndarray,
		y: Sequence[int],
		*,
		kind: str = "choice",
		key: str | None = None,
	) -> float:
		A = as_probabilities(P)
		k = key or self.key(kind, A.shape[1])
		T = fit_temperature(A, y)
		self.temperatures[k] = T
		return T

	def transform(
		self,
		P: Sequence[Sequence[float]] | np.ndarray,
		*,
		kind: str = "choice",
		key: str | None = None,
		strict: bool = False,
	) -> np.ndarray:
		"""Apply the fitted temperature. Unknown shapes pass through untouched."""
		A = as_probabilities(P)
		k = key or self.key(kind, A.shape[1])
		T = self.temperatures.get(k)
		if T is None:
			if strict:
				raise KeyError(f"no temperature fitted for {k!r}; call fit() or pass strict=False")
			warnings.warn(f"no temperature fitted for {k!r}; returning raw probabilities", stacklevel=2)
			return A
		return _scale(A, T)

	def fit_transform(
		self,
		P: Sequence[Sequence[float]] | np.ndarray,
		y: Sequence[int],
		*,
		kind: str = "choice",
		key: str | None = None,
	) -> np.ndarray:
		"""Convenience for exploration only. Report numbers on a held-out split instead."""
		self.fit(P, y, kind=kind, key=key)
		return self.transform(P, kind=kind, key=key)

	def save(self, path: str | Path) -> None:
		Path(path).write_text(json.dumps({"temperatures": self.temperatures}, indent=2) + "\n")

	@classmethod
	def load(cls, path: str | Path) -> "TemperatureScaler":
		data = json.loads(Path(path).read_text())
		return cls(data["temperatures"])


def holdout_split(n: int, fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
	"""Deterministic first/second split. Fit on the first half, report on the second."""
	if not 0.0 < fraction < 1.0:
		raise ValueError(f"fraction must lie in (0, 1), got {fraction}")
	cut = max(1, int(n * fraction))
	idx = np.arange(n)
	return idx[:cut], idx[cut:]

"""Calibration and decision-quality metrics.

A typed-decision model returns a probability per option. Two questions matter and
they are not the same one:

- Is the number honest? (calibration: ECE, Brier, reliability table)
- Does the ordering pick out the rows worth escalating? (risk-coverage)

A model can be badly calibrated and still rank perfectly, which is why both are
reported side by side.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Iterable, Sequence

import numpy as np

DEFAULT_BINS = 15
DEFAULT_EDGES: tuple[tuple[float, float], ...] = ((0.9, 1.0), (0.7, 0.9), (0.5, 0.7), (0.0, 0.5))
DEFAULT_COVERAGES: tuple[float, ...] = (1.0, 0.9, 0.7, 0.5)


def as_probabilities(P: Sequence[Sequence[float]] | np.ndarray, *, tol: float = 1e-3) -> np.ndarray:
	"""Validate an (n, k) probability matrix. Raises rather than silently renormalising."""
	A = np.asarray(P, dtype=float)
	if A.ndim != 2 or A.shape[0] == 0 or A.shape[1] < 2:
		raise ValueError(f"expected an (n, k>=2) probability matrix, got shape {A.shape}")
	if not np.isfinite(A).all():
		raise ValueError("probability matrix contains NaN or inf")
	if (A < -tol).any():
		raise ValueError("probability matrix contains negative entries")
	sums = A.sum(axis=1)
	if not np.allclose(sums, 1.0, atol=tol):
		bad = int(np.argmax(np.abs(sums - 1.0)))
		raise ValueError(f"row {bad} sums to {sums[bad]:.6f}, not 1")
	return A


def as_labels(y: Sequence[int] | np.ndarray, n_classes: int) -> np.ndarray:
	v = np.asarray(y, dtype=int)
	if v.ndim != 1:
		raise ValueError("labels must be one-dimensional")
	if v.size and (v.min() < 0 or v.max() >= n_classes):
		raise ValueError(f"labels must lie in [0, {n_classes - 1}]")
	return v


def ece(confidence: Sequence[float], correct: Sequence[bool], bins: int = DEFAULT_BINS) -> float:
	"""Expected calibration error: mean |accuracy - confidence| over equal-width bins."""
	c = np.asarray(confidence, dtype=float)
	ok = np.asarray(correct, dtype=float)
	if c.shape != ok.shape:
		raise ValueError("confidence and correct must have the same shape")
	if c.size == 0:
		return float("nan")
	edges = np.linspace(0.0, 1.0, bins + 1)
	total = 0.0
	for lo, hi in zip(edges[:-1], edges[1:]):
		m = (c > lo) & (c <= hi) if lo > 0 else (c >= lo) & (c <= hi)
		if m.any():
			total += m.mean() * abs(ok[m].mean() - c[m].mean())
	return float(total)


def brier(P: np.ndarray, y: np.ndarray) -> float:
	"""Multiclass Brier score: mean squared error against the one-hot outcome."""
	A = as_probabilities(P)
	v = as_labels(y, A.shape[1])
	onehot = np.eye(A.shape[1])[v]
	return float(((A - onehot) ** 2).sum(axis=1).mean())


@dataclass(frozen=True)
class Bin:
	low: float
	high: float
	count: int
	accuracy: float
	mean_confidence: float


def reliability(
	confidence: Sequence[float],
	correct: Sequence[bool],
	edges: Iterable[tuple[float, float]] = DEFAULT_EDGES,
) -> list[Bin]:
	"""Accuracy grouped by the confidence the model attached to its own answer."""
	c = np.asarray(confidence, dtype=float)
	ok = np.asarray(correct, dtype=float)
	out: list[Bin] = []
	for lo, hi in edges:
		m = (c >= lo) & (c < hi) if hi < 1.0 else (c >= lo) & (c <= 1.0)
		n = int(m.sum())
		out.append(
			Bin(
				low=float(lo),
				high=float(hi),
				count=n,
				accuracy=float(ok[m].mean()) if n else float("nan"),
				mean_confidence=float(c[m].mean()) if n else float("nan"),
			)
		)
	return out


@dataclass(frozen=True)
class CoveragePoint:
	coverage: float
	kept: int
	accuracy: float


def risk_coverage(
	confidence: Sequence[float],
	correct: Sequence[bool],
	coverages: Iterable[float] = DEFAULT_COVERAGES,
) -> list[CoveragePoint]:
	"""Accuracy when only the most confident fraction is kept and the rest is escalated.

	This is the metric that decides an escalation threshold. It depends on the
	ordering of the confidences, not on their absolute values, so it stays
	meaningful on an uncalibrated model.
	"""
	c = np.asarray(confidence, dtype=float)
	ok = np.asarray(correct, dtype=float)
	order = np.argsort(-c, kind="stable")
	out: list[CoveragePoint] = []
	for cov in coverages:
		if not 0.0 < cov <= 1.0:
			raise ValueError(f"coverage must lie in (0, 1], got {cov}")
		k = max(1, int(round(len(ok) * cov)))
		out.append(CoveragePoint(coverage=float(cov), kept=k, accuracy=float(ok[order[:k]].mean())))
	return out


@dataclass(frozen=True)
class Report:
	n: int
	n_classes: int
	accuracy: float
	ece: float
	brier: float
	mean_confidence: float
	reliability: list[Bin] = field(default_factory=list)
	coverage: list[CoveragePoint] = field(default_factory=list)

	def to_dict(self) -> dict:
		return asdict(self)

	def to_json(self, indent: int = 2) -> str:
		return json.dumps(self.to_dict(), indent=indent)

	def __str__(self) -> str:
		head = (
			f"n={self.n}  classes={self.n_classes}  acc={self.accuracy:.3f}  "
			f"ECE={self.ece:.3f}  Brier={self.brier:.3f}  mean conf={self.mean_confidence:.3f}"
		)
		rel = "  ".join(
			f"[{b.low:.1f},{b.high:.1f}) n={b.count} acc={b.accuracy:.3f}"
			for b in self.reliability
			if b.count
		)
		cov = "  ".join(f"keep {int(p.coverage * 100)}%={p.accuracy:.3f}" for p in self.coverage)
		return f"{head}\n  reliability  {rel}\n  coverage     {cov}"


def evaluate(
	P: np.ndarray,
	y: Sequence[int],
	*,
	confidence: Sequence[float] | None = None,
	bins: int = DEFAULT_BINS,
) -> Report:
	"""Score a batch of decisions.

	`confidence` defaults to the winning option's probability. Pass the backend's
	own confidence statistic when it reports one that is not the max probability.
	"""
	A = as_probabilities(P)
	v = as_labels(y, A.shape[1])
	if len(v) != len(A):
		raise ValueError(f"got {len(A)} rows of probabilities but {len(v)} labels")
	ok = A.argmax(axis=1) == v
	conf = A.max(axis=1) if confidence is None else np.asarray(confidence, dtype=float)
	return Report(
		n=len(v),
		n_classes=A.shape[1],
		accuracy=float(ok.mean()),
		ece=ece(conf, ok, bins=bins),
		brier=brier(A, v),
		mean_confidence=float(conf.mean()),
		reliability=reliability(conf, ok),
		coverage=risk_coverage(conf, ok),
	)

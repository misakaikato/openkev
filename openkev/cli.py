"""Command line entry point: openkev fit | eval."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np

from .calibrate import TemperatureScaler, holdout_split
from .metrics import evaluate


def load_rows(path: str | Path) -> list[dict]:
	rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
	if not rows:
		raise SystemExit(f"{path} contains no rows")
	for i, r in enumerate(rows):
		if "label" not in r:
			raise SystemExit(f"row {i} has no 'label'")
		if "probabilities" not in r and not {"state", "question", "options"} <= r.keys():
			raise SystemExit(
				f"row {i} needs either 'probabilities' or all of 'state', 'question', 'options'"
			)
	return rows


def score_rows(rows: list[dict], model: str | None) -> np.ndarray:
	"""Use precomputed probabilities when present, otherwise run the MLX backend."""
	if all("probabilities" in r for r in rows):
		widths = {len(r["probabilities"]) for r in rows}
		if len(widths) != 1:
			raise SystemExit(f"rows mix option counts {sorted(widths)}; split them per shape")
		return np.asarray([r["probabilities"] for r in rows], dtype=float)
	if model is None:
		raise SystemExit("rows have no 'probabilities'; pass --model to score them locally")
	from .scorer import MLXScorer

	scorer = MLXScorer(model)
	return np.asarray(
		[scorer.choice(r["state"], r["question"], r["options"]) for r in rows], dtype=float
	)


def _report(P: np.ndarray, y: np.ndarray, title: str) -> None:
	print(f"{title}\n  {evaluate(P, y)}")


def cmd_fit(args: argparse.Namespace) -> int:
	rows = load_rows(args.data)
	P = score_rows(rows, args.model)
	y = np.asarray([r["label"] for r in rows], dtype=int)
	scaler = TemperatureScaler.load(args.out) if args.out and Path(args.out).exists() else TemperatureScaler()
	key = args.key or TemperatureScaler.key(args.kind, P.shape[1])
	T = scaler.fit(P, y, key=key)
	scaler.save(args.out)
	print(f"fitted {key} -> T={T}  (T>1 means the backend was over-confident)")
	print(f"wrote {args.out}")
	return 0


def cmd_eval(args: argparse.Namespace) -> int:
	rows = load_rows(args.data)
	P = score_rows(rows, args.model)
	y = np.asarray([r["label"] for r in rows], dtype=int)
	key = args.key or TemperatureScaler.key(args.kind, P.shape[1])

	if args.holdout:
		fit_idx, test_idx = holdout_split(len(y), args.holdout)
		scaler = TemperatureScaler()
		T = scaler.fit(P[fit_idx], y[fit_idx], key=key)
		_report(P[test_idx], y[test_idx], f"held-out, uncalibrated (n={len(test_idx)})")
		_report(scaler.transform(P[test_idx], key=key), y[test_idx], f"held-out, T={T}")
		if args.save:
			scaler.save(args.save)
			print(f"wrote {args.save}")
		return 0

	_report(P, y, f"uncalibrated (n={len(y)})")
	if args.calibration:
		scaler = TemperatureScaler.load(args.calibration)
		_report(scaler.transform(P, key=key, strict=True), y, f"calibrated with {args.calibration}")
	return 0


def build_parser() -> argparse.ArgumentParser:
	p = argparse.ArgumentParser(prog="openkev", description=__doc__)
	sub = p.add_subparsers(dest="command", required=True)

	common = argparse.ArgumentParser(add_help=False)
	common.add_argument("--data", required=True, help="JSONL with 'label' plus either 'probabilities' or state/question/options")
	common.add_argument("--model", default=None, help="MLX model id, when rows need scoring")
	common.add_argument("--kind", default="choice", help="question shape name for the temperature key")
	common.add_argument("--key", default=None, help="explicit temperature key, overrides --kind")

	f = sub.add_parser("fit", parents=[common], help="fit a temperature and write it to disk")
	f.add_argument("--out", required=True, help="calibration JSON to create or update")
	f.set_defaults(func=cmd_fit)

	e = sub.add_parser("eval", parents=[common], help="report accuracy, calibration and coverage")
	e.add_argument("--calibration", default=None, help="apply a saved calibration before reporting")
	e.add_argument("--holdout", type=float, default=None, metavar="FRACTION",
				   help="fit on the first FRACTION of rows and report on the rest")
	e.add_argument("--save", default=None, help="with --holdout, also write the fitted calibration")
	e.set_defaults(func=cmd_eval)
	return p


def main(argv: Sequence[str] | None = None) -> int:
	args = build_parser().parse_args(argv)
	return args.func(args)


if __name__ == "__main__":
	sys.exit(main())

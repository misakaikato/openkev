import json

import numpy as np

from openkev.cli import main


def write_rows(path, P, y):
	with path.open("w") as fh:
		for probs, label in zip(P, y):
			fh.write(json.dumps({"probabilities": list(map(float, probs)), "label": int(label)}) + "\n")


def test_fit_then_eval_roundtrip(tmp_path, capsys, overconfident):
	P, y, _ = overconfident(n=600)
	data = tmp_path / "rows.jsonl"
	calib = tmp_path / "calibration.json"
	write_rows(data, P, y)

	assert main(["fit", "--data", str(data), "--out", str(calib)]) == 0
	assert "choice:3" in json.loads(calib.read_text())["temperatures"]

	assert main(["eval", "--data", str(data), "--calibration", str(calib)]) == 0
	out = capsys.readouterr().out
	assert "uncalibrated" in out and "calibrated with" in out


def test_holdout_reports_both_sides(tmp_path, capsys, overconfident):
	P, y, _ = overconfident(n=600)
	data = tmp_path / "rows.jsonl"
	write_rows(data, P, y)
	assert main(["eval", "--data", str(data), "--holdout", "0.5"]) == 0
	out = capsys.readouterr().out
	assert "held-out, uncalibrated" in out and "held-out, T=" in out


def test_rows_without_probabilities_need_a_model(tmp_path):
	data = tmp_path / "rows.jsonl"
	data.write_text(json.dumps({"state": "s", "question": "q", "options": ["a", "b"], "label": 0}) + "\n")
	try:
		main(["eval", "--data", str(data)])
	except SystemExit as exc:
		assert "--model" in str(exc)
	else:
		raise AssertionError("expected SystemExit")

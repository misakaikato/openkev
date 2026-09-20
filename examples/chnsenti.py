"""End to end on a Chinese sentiment set: score, evaluate, calibrate, compare prompts.

	pip install 'openkev[mlx]' datasets
	python examples/chnsenti.py
"""

import numpy as np
from datasets import load_dataset

from openkev import MLXScorer, TemperatureScaler, evaluate, holdout_split

N = 400
PROMPTS = {
	"chinese": ("判断这条评论的整体情感倾向", ["负面", "正面"]),
	"english": ("What is the overall sentiment of this review?", ["negative", "positive"]),
}


def main() -> None:
	rows = load_dataset("lansinuote/ChnSentiCorp", split=f"test[:{N}]")
	texts = [r["text"][:400] for r in rows]
	y = np.array([int(r["label"]) for r in rows])  # 0 = negative, 1 = positive
	scorer = MLXScorer()

	for name, (question, options) in PROMPTS.items():
		P = np.array([scorer.choice(t, question, options) for t in texts])
		fit, test = holdout_split(len(y))
		scaler = TemperatureScaler()
		T = scaler.fit(P[fit], y[fit])
		print(f"\n[{name}] raw")
		print(f"  {evaluate(P[test], y[test])}")
		print(f"[{name}] calibrated, T={T}")
		print(f"  {evaluate(scaler.transform(P[test]), y[test])}")


if __name__ == "__main__":
	main()

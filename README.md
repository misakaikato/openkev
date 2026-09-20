# openkev

**Calibration and decision-quality evaluation for typed-decision models.**

A typed-decision model answers a multiple-choice question with a probability per
option, in one forward pass, generating no text. Jev popularised the interface;
Laya, SemIf and a dozen weekend projects reproduce it. All of them hand you a
number between 0 and 1, and almost none of them tell you whether that number
means anything.

openkev is the layer that answers this. It wraps any scorer, measures how honest
its probabilities are, fits the correction, and reports the two things you
actually need before putting a decision model in front of real traffic.

## What this is not

Not another decision runtime. [SemIf](https://github.com/TheoLeeCJ/SemIf) already
does that part well — CUDA and MLX backends, shared-state execution, quantisation,
frozen reproducible benchmarks — and its documentation is explicit that calibrated
probabilities suitable for operational thresholds are *not* among the things it
establishes. That gap is what openkev fills. The bundled `MLXScorer` exists so the
package works standalone on a Mac; for serious serving, point openkev at SemIf.

## Install

```bash
pip install openkev                 # calibration and metrics only, numpy
pip install 'openkev[mlx]'          # plus the local Apple Silicon scorer
```

## Use

Score, evaluate, calibrate:

```python
from openkev import MLXScorer, TemperatureScaler, evaluate, holdout_split
import numpy as np

scorer = MLXScorer("mlx-community/Qwen3-1.7B-bf16")
P = np.array([scorer.choice(text, "What is the sentiment of this review?",
                            ["negative", "positive"]) for text in texts])

fit, test = holdout_split(len(labels))
scaler = TemperatureScaler()
scaler.fit(P[fit], np.array(labels)[fit])

print(evaluate(P[test], np.array(labels)[test]))                      # before
print(evaluate(scaler.transform(P[test]), np.array(labels)[test]))    # after
scaler.save("calibration.json")
```

Already have probabilities from somewhere else — SemIf, a hosted endpoint, your
own model? Skip the scorer entirely. Write one JSON object per row with
`probabilities` and `label`, then:

```bash
openkev eval --data rows.jsonl --holdout 0.5 --save calibration.json
```

```
held-out, uncalibrated (n=200)
  n=200  classes=2  acc=0.910  ECE=0.089  Brier=0.176  mean conf=0.998
  reliability  [0.9,1.0) n=198 acc=0.914  [0.7,0.9) n=2 acc=0.500
  coverage     keep 100%=0.910  keep 90%=0.917  keep 70%=0.936  keep 50%=0.930
held-out, T=8.835
  n=200  classes=2  acc=0.910  ECE=0.052  Brier=0.161  mean conf=0.872
  reliability  [0.9,1.0) n=117 acc=0.940  [0.7,0.9) n=74 acc=0.878  [0.5,0.7) n=9 acc=0.778
  coverage     keep 100%=0.910  keep 90%=0.917  keep 70%=0.943  keep 50%=0.950
```

Note the shape of the uncalibrated run: 198 of 200 rows land in the top
confidence bin. The number carries almost no information until it is scaled, yet
the *ordering* underneath it already separates the errors.

Read that output as two separate findings. `ECE` says whether 0.9 means 90%.
`coverage` says what happens if you escalate the least confident rows to a human.
They move independently.

## What the metrics are for

| | Question it answers | When it matters |
|---|---|---|
| `accuracy` | Is the decision right? | Always |
| `ece`, `brier` | Is the stated probability honest? | When a downstream step multiplies by it |
| `reliability` | Honest *where*? | Thresholds differ per confidence band |
| `risk_coverage` | Does the ordering find the rows worth escalating? | Human-in-the-loop routing |

Temperature scaling never changes which option wins, so accuracy is invariant
under calibration. If you only need escalation routing, you may not need to
calibrate at all — see below.

## Measured

Apple M5 Max. `mlx-community/Qwen3-1.7B-bf16` through `MLXScorer`, and
`aac6fef/laya-multilingual-mlx` through its own API, on 400 rows each of
ChnSentiCorp, TNEWS and XNLI-zh. Single datasets, single runs, small samples:
read these as the shape of an effect, not as benchmark results.

**Every backend tested was over-confident out of the box, and one scalar fixed
it.** Every fitted temperature came out well above 1. Calibration error is
reported on a held-out half, never on the rows the temperature was fitted on.

| Backend / task | ECE before | fitted T | ECE after |
|---|---:|---:|---:|
| Qwen3-1.7B, ChnSentiCorp | 0.089 | 8.84 | **0.052** |
| Laya 322M, ChnSentiCorp | 0.132 | 3.45 | **0.054** |
| Laya 322M, XNLI-zh | 0.191 | 2.10 | **0.069** |
| Laya 322M, TNEWS (15-way) | 0.336 | 2.15 | **0.076** |

**Prompt language can matter more than anything else you tune — on some
backends.** Same Chinese input, only the language of the question and the option
labels changed:

| Task | Backend | Chinese prompt | English prompt |
|---|---|---:|---:|
| ChnSentiCorp, sentiment | Laya 322M | 0.713 | **0.838** |
| XNLI-zh, inference | Laya 322M | 0.642 | **0.767** |
| TNEWS, 15 news channels | Laya 322M | **0.445** | 0.403 |
| ChnSentiCorp, sentiment | Qwen3-1.7B | 0.895 | **0.910** |

Two things to take from this. English wins where the labels are generic semantic
categories and loses where they are culture-specific — a Chinese news desk
separates 股票 from 财经 and the English rendering collapses that. And the size of
the effect is a property of the backend, not a law: 12.5 points on the small
multilingual encoder, 1.5 points on the decoder. So measure it on your own label
set rather than copying a rule of thumb.

**Uncalibrated probabilities still rank well.** On the run above, keeping only the
most confident 70% of rows lifted accuracy from 0.910 to 0.943 using the raw,
wildly over-confident numbers. Calibrate when a downstream step consumes the
probability as a probability; for escalation routing the ordering is enough.

**Put the state before the question.** Both orderings read naturally, but only
state-first lets the state's key-value cache be shared across questions — and in
development the question-first layout also produced confidently wrong answers on
questions the state-first layout got right. `render_parts` is the single place
this prompt is built, and `choice` and `multi` are asserted to agree byte for
byte.

**Two debiasing tricks were a net loss, so they are not shipped.** Option-order
rotation and contextual prior correction each cost 2–4x the forward passes and
lowered accuracy on a binary task (0.895 to 0.877 and 0.877, both together
0.870), measured during development on a prototype of `MLXScorer`. They may well
pay off with more options. Measure before adopting — which is what this library
is for.

## Limitations

- Temperature scaling is one scalar per question shape. It corrects over-confidence; it cannot fix a model that is wrong, and it does not reorder anything.
- The reported probabilities are conditional on the options you supplied. They are not a probability that the answer is correct in any absolute sense.
- A temperature fitted on one workload does not transfer to another. Refit per task, per label set, per question shape.
- `MLXScorer` is a reference implementation: single sequence, no batching, macOS arm64 only.
- All numbers above come from one machine and one run each. The repository ships the tests, not frozen benchmark fixtures.

## Development

```bash
uv venv && uv pip install -e '.[test]'
pytest -q
```

MIT licensed. Independent project; not affiliated with TypeSafe, Convai
Innovations or SemIf. Jev, TypeSafe and other names are the property of their
respective owners.

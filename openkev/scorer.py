"""Backends that turn a text state plus a question into option probabilities.

`Scorer` is the only thing openkev's calibration and evaluation layers need, so a
TypeSafe endpoint, a SemIf runner or a local model all plug in the same way.

`MLXScorer` is a small reference backend for Apple Silicon: it prompts a normal
decoder model up to the position where the answer token goes, runs one forward
pass, and reads the logits of the candidate letters. No token is ever sampled.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-bf16"


@runtime_checkable
class Scorer(Protocol):
	"""Anything that answers a multiple-choice question with a probability per option."""

	def choice(self, state: str, question: str, options: Sequence[str]) -> np.ndarray: ...


def render_parts(state: str, question: str, options: Sequence[str]) -> tuple[str, str]:
	"""Split the prompt at the state/question boundary.

	The state comes first so that its key-value cache can be computed once and
	reused across questions. `choice` and `multi` therefore build byte-identical
	prompts and must agree; `tests` and the MLX smoke check assert exactly that.
	"""
	body = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(options))
	head = f"请阅读以下输入并回答问题。\n\n输入：{state}\n\n"
	tail = f"{question}\n\n{body}\n\n只回答一个字母。答案："
	return head, tail


def render(state: str, question: str, options: Sequence[str]) -> str:
	return "".join(render_parts(state, question, options))


class MLXScorer:
	"""Single forward pass, zero output tokens, on macOS arm64.

	For CUDA, parallel shared-state execution or a reproducible benchmark harness,
	use SemIf (https://github.com/TheoLeeCJ/SemIf) as the backend instead; this
	class exists so openkev is usable on its own on a Mac.
	"""

	def __init__(self, model_id: str = DEFAULT_MODEL, *, enable_thinking: bool = False) -> None:
		try:
			import mlx.core as mx
			from mlx_lm import load
		except ImportError as exc:	# pragma: no cover - depends on the host
			raise ImportError("MLXScorer needs the mlx extra: pip install 'openkev[mlx]'") from exc
		self._mx = mx
		self.model_id = model_id
		self.enable_thinking = enable_thinking
		self.model, self.tokenizer = load(model_id)

	# -- prompt plumbing -------------------------------------------------
	def _candidate_ids(self, tokens: Sequence[str]):
		ids = []
		for t in tokens:
			enc = self.tokenizer.encode(t, add_special_tokens=False)
			if len(enc) != 1:
				raise ValueError(
					f"candidate {t!r} is not a single token ({enc}); use letter or digit "
					"codes and put the wording in the option list"
				)
			ids.append(enc[0])
		return self._mx.array(ids)

	def _prompt_ids(self, body: str):
		return self.tokenizer.apply_chat_template(
			[{"role": "user", "content": body}],
			add_generation_prompt=True,
			enable_thinking=self.enable_thinking,
		)

	def _read_logits(self, prompt_ids, candidate_ids) -> np.ndarray:
		mx = self._mx
		logits = self.model(mx.array([prompt_ids]))[0, -1]
		p = mx.softmax(logits[candidate_ids].astype(mx.float32))
		mx.eval(p)
		return np.array(p, dtype=float)

	# -- primitives ------------------------------------------------------
	def choice(self, state: str, question: str, options: Sequence[str]) -> np.ndarray:
		if not 2 <= len(options) <= len(LETTERS):
			raise ValueError(f"need between 2 and {len(LETTERS)} options, got {len(options)}")
		body = render(state, question, options)
		return self._read_logits(self._prompt_ids(body), self._candidate_ids(LETTERS[: len(options)]))

	def noul_options(self) -> list[str]:
		return ["是", "否"]

	def noul(self, state: str, question: str) -> float:
		"""Probability that the proposition holds."""
		return float(self.choice(state, question, ["是", "否"])[0])

	def score(self, state: str, question: str, levels: int = 5) -> tuple[np.ndarray, float]:
		"""Ordered rating. Returns the level distribution and its expected value.

		The expectation carries more information than the winning level: 3.2 and
		3.8 are both argmax 3.
		"""
		if not 2 <= levels <= 9:
			raise ValueError(f"levels must lie in [2, 9] to stay single-token, got {levels}")
		body = (
			f"请阅读以下输入并回答问题。\n\n输入：{state}\n\n{question}\n\n"
			f"用 1-{levels} 打分（1=最低，{levels}=最高）。只回答一个数字。答案："
		)
		p = self._read_logits(self._prompt_ids(body), self._candidate_ids([str(i + 1) for i in range(levels)]))
		return p, float((np.arange(1, levels + 1) * p).sum())

	# -- shared prefix ---------------------------------------------------
	def multi(self, state: str, questions: Sequence[tuple[str, Sequence[str]]]) -> list[np.ndarray]:
		"""Ask several questions about one state, prefilling the state once.

		Worth it roughly in proportion to state length times question count; on a
		short state with two or three questions the bookkeeping costs more than it
		saves. Every prompt is built by `render_parts`, so each result is identical
		to the matching `choice` call.
		"""
		from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache, trim_prompt_cache

		mx = self._mx
		if not questions:
			return []
		parts = [render_parts(state, q, o) for q, o in questions]
		head, tails = parts[0][0], [t for _, t in parts]
		rendered = self.tokenizer.apply_chat_template(
			[{"role": "user", "content": head + tails[0]}],
			add_generation_prompt=True,
			enable_thinking=self.enable_thinking,
			tokenize=False,
		)
		cut = rendered.index(tails[0])
		prefix = self.tokenizer.encode(rendered[:cut], add_special_tokens=False)
		suffix = rendered[cut + len(tails[0]) :]

		cache = make_prompt_cache(self.model)
		if not can_trim_prompt_cache(cache):
			return [self.choice(state, q, o) for q, o in questions]
		self.model(mx.array([prefix]), cache=cache)
		out: list[np.ndarray] = []
		for (_, options), tail in zip(questions, tails):
			ids = self.tokenizer.encode(tail + suffix, add_special_tokens=False)
			logits = self.model(mx.array([ids]), cache=cache)[0, -1]
			p = mx.softmax(logits[self._candidate_ids(LETTERS[: len(options)])].astype(mx.float32))
			mx.eval(p)
			out.append(np.array(p, dtype=float))
			trim_prompt_cache(cache, len(ids))
		return out


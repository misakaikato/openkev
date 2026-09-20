"""分类速递 — a falling-item game where every decision is semantic and timed.

Items drop one at a time and must land in one of four waste bins before they hit
the floor. Deciding "沾油的披萨盒" is not geometry: it is a judgement about a short
piece of text, made in the low tens of milliseconds, which is exactly the shape of
problem a typed-decision model is for.

Calibration is part of the rules, not decoration. The temperature and the
escalation threshold are both fitted on a held-out split of the item list: the
threshold is the lowest confidence cut at which the retained rows still hit
`TARGET_ACCURACY`. At play time anything below it goes to 人工复核 for partial
credit instead of being guessed. An over-confident model never escalates and eats
the penalties; a well calibrated one knows which items it should not answer.

	pip install 'openkev[mlx]' pillow
	python examples/sortgame.py
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from openkev import (
	MLXScorer,
	TemperatureScaler,
	evaluate,
	holdout_split,
	select_threshold,
)

BINS = ["可回收物", "厨余垃圾", "有害垃圾", "其他垃圾"]
QUESTION = "这件垃圾应该投放到哪个垃圾桶？"
DEFAULT_MODEL = "mlx-community/Qwen3-1.7B-bf16"
SCORE = {"correct": 10, "wrong": -5, "review": 2}

# (物品, 正确的桶下标). 顺序固定：前 50 条拟合温度，后 14 条才是比赛用的题目。
ITEMS: list[tuple[str, int]] = [
	("旧报纸", 0), ("易拉罐", 0), ("塑料矿泉水瓶", 0), ("旧衣服", 0), ("玻璃啤酒瓶", 0),
	("快递纸箱", 0), ("废铁锅", 0), ("旧手机充电器", 0), ("硬纸板", 0), ("空洗发水瓶", 0),
	("牛奶盒", 0), ("旧书本", 0), ("铝合金窗框", 0),
	("剩饭剩菜", 1), ("鱼骨头", 1), ("菜叶子", 1), ("西瓜皮", 1), ("茶叶渣", 1),
	("鸡蛋壳", 1), ("苹果核", 1), ("过期的牛奶", 1), ("剩面条", 1), ("烂掉的草莓", 1),
	("虾壳", 1), ("玉米须", 1), ("咖啡渣", 1),
	("废旧电池", 2), ("过期药品", 2), ("水银温度计", 2), ("杀虫剂罐", 2), ("油漆桶", 2),
	("废弃的指甲油", 2), ("废灯泡", 2), ("过期化妆品", 2), ("废胶片", 2), ("消毒剂瓶", 2),
	("废弃的农药瓶", 2),
	("用过的纸巾", 3), ("一次性筷子", 3), ("烟头", 3), ("陶瓷碗碎片", 3), ("旧毛巾", 3),
	("创可贴", 3), ("笔芯", 3), ("干燥剂", 3), ("防尘口罩", 3), ("污损的信封", 3),
	("尿不湿", 3), ("宠物粪便", 3), ("碎玻璃镜子", 3),
	# ---- 比赛题目从这里开始：混了几个经典易错项 ----
	("沾满油污的披萨盒", 3), ("大骨头", 3), ("用过的猫砂", 3), ("坚果壳", 3),
	("荧光灯管", 2), ("废弃的体温计", 2), ("过期的维生素片", 2),
	("香蕉皮", 1), ("发霉的面包", 1), ("排骨上的肉", 1),
	("空的玻璃罐头瓶", 0), ("废旧自行车", 0), ("泡沫快递箱", 0), ("旧充电宝", 2),
]
PLAY_FROM = 50

# ---- 画面 ----
W, H = 680, 400
FONT = "/System/Library/Fonts/STHeiti Medium.ttc"
SUBTITLE = "openkev"
BG = (18, 20, 26)
FG = (232, 234, 240)
DIM = (120, 126, 140)
BIN_COLOR = [(64, 132, 214), (86, 170, 96), (206, 92, 84), (128, 134, 148)]
GOOD, BAD, WARN = (86, 190, 110), (222, 92, 88), (226, 172, 72)


@dataclass
class Turn:
	item: str
	truth: int
	picked: int
	probs: np.ndarray
	confidence: float
	escalated: bool
	ms: float
	score: int


def font(size: int) -> ImageFont.FreeTypeFont:
	return ImageFont.truetype(FONT, size)


def save_trace(turns: list[Turn], threshold: float, subtitle: str, path: Path) -> None:
	path.write_text(json.dumps({
		"threshold": threshold, "subtitle": subtitle,
		"turns": [{**vars(t), "probs": [float(x) for x in t.probs]} for t in turns],
	}, ensure_ascii=False, indent=1))


def load_trace(path: Path) -> tuple[list[Turn], float, str]:
	d = json.loads(path.read_text())
	turns = [Turn(**{**t, "probs": np.array(t["probs"])}) for t in d["turns"]]
	return turns, d["threshold"], d["subtitle"]


def play(scorer: MLXScorer, scaler: TemperatureScaler, threshold: float) -> list[Turn]:
	turns, total = [], 0
	for item, truth in ITEMS[PLAY_FROM:]:
		t0 = time.perf_counter()
		raw = scorer.choice(item, QUESTION, BINS)
		ms = (time.perf_counter() - t0) * 1000
		probs = scaler.transform(raw.reshape(1, -1))[0]
		picked = int(probs.argmax())
		confidence = float(probs.max())
		escalated = confidence < threshold
		if escalated:
			gained = SCORE["review"]
		else:
			gained = SCORE["correct"] if picked == truth else SCORE["wrong"]
		total += gained
		turns.append(Turn(item, truth, picked, probs, confidence, escalated, ms, total))
	return turns


# ---- 渲染 ----
def bin_boxes() -> list[tuple[int, int, int, int]]:
	pad, top, h = 18, H - 96, 64
	w = (W - pad * (len(BINS) + 1)) // len(BINS)
	return [(pad + i * (w + pad), top, pad + i * (w + pad) + w, top + h) for i in range(len(BINS))]


def frame(turn: Turn, phase: str, y: int, index: int, total_items: int, threshold: float) -> Image.Image:
	img = Image.new("RGB", (W, H), BG)
	d = ImageDraw.Draw(img)

	d.text((18, 14), "分类速递", font=font(24), fill=FG)
	d.text((18, 48), SUBTITLE, font=font(13), fill=DIM)
	d.text((W - 18, 14), f"{turn.score:+d} 分", font=font(24), fill=FG, anchor="ra")
	d.text((W - 18, 48), f"第 {index + 1}/{total_items} 件", font=font(13), fill=DIM, anchor="ra")

	boxes = bin_boxes()
	for i, (x0, y0, x1, y1) in enumerate(boxes):
		active = phase in ("decide", "result") and i == turn.picked and not turn.escalated
		color = BIN_COLOR[i] if active else tuple(c // 3 for c in BIN_COLOR[i])
		d.rounded_rectangle((x0, y0, x1, y1), 10, fill=color)
		d.text(((x0 + x1) // 2, y1 - 22), BINS[i], font=font(15), fill=FG, anchor="ma")
		if phase in ("decide", "result"):
			p = float(turn.probs[i])
			bar = int((x1 - x0 - 20) * p)
			d.rectangle((x0 + 10, y0 + 12, x0 + 10 + bar, y0 + 22), fill=FG if active else DIM)
			d.text((x0 + 10, y0 + 26), f"{p:.2f}", font=font(12), fill=FG if active else DIM)

	if phase in ("fall", "decide", "result"):
		label = turn.item
		tw = d.textlength(label, font=font(22))
		cx = (boxes[turn.picked][0] + boxes[turn.picked][2]) // 2 if phase == "decide" else W // 2
		y = 104 if phase == "result" else y
		d.rounded_rectangle((cx - tw // 2 - 16, y - 24, cx + tw // 2 + 16, y + 24), 12,
							fill=(38, 42, 52), outline=DIM)
		d.text((cx, y), label, font=font(22), fill=FG, anchor="mm")

	if phase == "result":
		cut = "无解" if threshold == float("inf") else f"{threshold:.2f}"
		if turn.escalated:
			tone = WARN
			verdict = f"置信度 {turn.confidence:.2f} 低于出手线 {cut} → 人工复核  +2"
		elif turn.picked == turn.truth:
			tone, verdict = GOOD, f"投进{BINS[turn.picked]}，正确  +10"
		else:
			tone, verdict = BAD, f"投进{BINS[turn.picked]}，错了，应为{BINS[turn.truth]}  -5"
		d.rounded_rectangle((18, 158, W - 18, 222), 12, fill=(30, 34, 42), outline=tone, width=2)
		d.text((W // 2, 176), verdict, font=font(17), fill=tone, anchor="ma")
		d.text((W // 2, 200), f"决策耗时 {turn.ms:.0f} ms", font=font(13), fill=DIM, anchor="ma")
	return img


def render(turns: list[Turn], path: Path, threshold: float) -> None:
	frames, durations = [], []
	for i, turn in enumerate(turns):
		for y in (110, 160, 210):
			frames.append(frame(turn, "fall", y, i, len(turns), threshold))
			durations.append(110)
		frames.append(frame(turn, "decide", 250, i, len(turns), threshold))
		durations.append(320)
		frames.append(frame(turn, "result", 250, i, len(turns), threshold))
		durations.append(760)
	frames[0].save(path, save_all=True, append_images=frames[1:], duration=durations,
				   loop=0, optimize=True)


def main() -> None:
	ap = argparse.ArgumentParser(description=__doc__)
	ap.add_argument("--model", default=DEFAULT_MODEL)
	ap.add_argument("--target", type=float, default=0.95,
					help="出手的那批要达到的准确率，阈值由校准集解出来")
	ap.add_argument("--out", default=None)
	ap.add_argument("--replay", default=None, help="从存好的轨迹重渲染，不加载模型")
	args = ap.parse_args()

	out = Path(args.out) if args.out else Path(__file__).resolve().parents[1] / "docs" / "sortgame.gif"
	trace_path = out.with_suffix(".json")
	if args.replay:
		turns, threshold, subtitle = load_trace(Path(args.replay))
		globals()["SUBTITLE"] = subtitle
		render(turns, out, threshold)
		print(f"重渲染 {out}  ({out.stat().st_size / 1024:.0f} KB)")
		return

	global SUBTITLE
	SUBTITLE = f"openkev · {args.model.split('/')[-1]} · 单次前向，0 输出 token"
	scorer = MLXScorer(args.model)
	fit_idx, _ = holdout_split(len(ITEMS), PLAY_FROM / len(ITEMS))
	fit_items = [ITEMS[i] for i in fit_idx]
	P = np.array([scorer.choice(name, QUESTION, BINS) for name, _ in fit_items])
	y = np.array([b for _, b in fit_items])

	scaler = TemperatureScaler()
	T = scaler.fit(P, y)
	print(f"温度在 {len(y)} 件校准样本上拟合：T={T}")
	print(f"  校准前 {evaluate(P, y)}")
	calibrated = scaler.transform(P)
	report = evaluate(calibrated, y)
	print(f"  校准后 {report}")
	threshold = select_threshold(calibrated.max(axis=1), calibrated.argmax(axis=1) == y,
								 target_accuracy=args.target, min_coverage=0.2)
	if threshold == float("inf"):
		print(f"出手阈值：无解——{args.model} 在校准集上任何切点都到不了 {args.target}，全部转人工\n")
	else:
		print(f"出手阈值：置信度 >= {threshold:.2f}（校准集上这批的准确率 >= {args.target}）\n")

	turns = play(scorer, scaler, threshold)
	for t in turns:
		mark = "复核" if t.escalated else ("对" if t.picked == t.truth else "错")
		print(f"  {t.item:<12s} → {BINS[t.picked]}  {t.confidence:.2f}  {t.ms:5.1f} ms  {mark}")

	decided = [t for t in turns if not t.escalated]
	correct = sum(t.picked == t.truth for t in decided)
	print(f"\n最终 {turns[-1].score:+d} 分 | 出手 {len(decided)}/{len(turns)} 件，"
		  f"其中对 {correct} 件 | 复核 {len(turns) - len(decided)} 件 | "
		  f"中位决策 {np.median([t.ms for t in turns]):.1f} ms")

	save_trace(turns, threshold, SUBTITLE, trace_path)
	render(turns, out, threshold)
	print(f"动图写入 {out}  ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
	main()

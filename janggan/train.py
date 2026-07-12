"""4단계 — v1/v2 를 동일 조건으로 각각 from-scratch 학습.

변수는 인코딩 하나뿐: 데이터·모델 크기·스텝·시드·lr 전부 고정.
체크포인트(runs/<scheme>.pt)와 loss 이력(runs/<scheme>.loss.json) 저장.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset import build
from model import TinyGPT

# ── 고정 하이퍼파라미터 (v1/v2 공통) ─────────────────────────────────────────
SEED = 1337
D, H, LAYERS, DROPOUT = 64, 4, 2, 0.1
STEPS, LR, LOG_EVERY = 2000, 3e-3, 200

DATA = Path(__file__).with_name("data") / "patterns.json"
RUNS = Path(__file__).with_name("runs")


def device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_scheme(scheme, patterns, dev):
    torch.manual_seed(SEED)
    vocab, stoi, itos, data, maxlen, ids = build(patterns, scheme)
    data = data.to(dev)
    inputs, targets = data[:, :-1], data[:, 1:]

    model = TinyGPT(len(vocab), maxlen, ids["pad"], D, H, LAYERS, DROPOUT).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)

    history = []
    model.train()
    for step in range(1, STEPS + 1):
        _, loss = model(inputs, targets)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % LOG_EVERY == 0 or step == 1:
            history.append({"step": step, "loss": round(loss.item(), 4)})
            print(f"  [{scheme}] step {step:4d}  loss {loss.item():.4f}")

    RUNS.mkdir(exist_ok=True)
    torch.save(
        {"state": model.state_dict(), "vocab": vocab, "maxlen": maxlen, "ids": ids,
         "cfg": {"d": D, "h": H, "layers": LAYERS, "dropout": DROPOUT}},
        RUNS / f"{scheme}.pt",
    )
    (RUNS / f"{scheme}.loss.json").write_text(json.dumps(history, indent=2))
    return vocab, history[-1]["loss"]


if __name__ == "__main__":
    dev = device()
    patterns = json.loads(DATA.read_text(encoding="utf-8"))["patterns"]
    print(f"device={dev}  patterns={len(patterns)}  steps={STEPS} seed={SEED}\n")

    results = {}
    for scheme in ("v1", "v2"):
        vocab, final = train_scheme(scheme, patterns, dev)
        results[scheme] = {"vocab": len(vocab), "final_loss": final}
        print(f"  → {scheme}: vocab={len(vocab)} final_loss={final}\n")

    print("done:", json.dumps(results, ensure_ascii=False))

"""5단계 — 두 모델 생성물의 구조 유효성 비교 (연구 ④).

각 모델에서 온도별 N개 생성 → 디코딩 → 유효성 측정:
  parse   토큰이 문법적으로 파싱되나
  cycle   총 길이가 학습 장단 길이(9/10/12)에 드나        ← 양쪽 공통 지표
  shape   박 구조가 학습 장단 shape 에 드나                 ← v2 전용(v1은 박 경계 부재)
  novel   학습셋에 없는 새 시퀀스인가                        ← 암기 아닌 생성 확인

핵심 가설(3단계서 수정): v1 은 박 경계를 토큰에 못 담아 cycle 유효율이 더 낮을 것.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from dataset import BOS, EOS, PAD, ENCODERS, build
from model import TinyGPT

RUNS = Path(__file__).with_name("runs")
DATA = Path(__file__).with_name("data") / "patterns.json"
GU = {"deong", "kung", "deok", "gideok", "roll"}
N, TEMPS, SEED = 200, [1.0, 1.5, 2.0], 7


# ── 안전 파서 (생성물은 malformed 일 수 있음) ────────────────────────────────

def v1_parse(toks):
    if len(toks) == 0 or len(toks) % 2 != 0:
        return None
    flat = []
    for a in range(0, len(toks), 2):
        sym, dur = toks[a], toks[a + 1]
        if sym not in GU and sym != "rest":
            return None
        if not (dur.startswith("d") and dur[1:].isdigit() and int(dur[1:]) >= 1):
            return None
        flat.append(None if sym == "rest" else sym)
        flat += [None] * (int(dur[1:]) - 1)
    return flat


def v2_parse(toks):
    if not toks or toks[0][0] != "|":
        return None
    grid, beat, i = [], None, 0
    while i < len(toks):
        t = toks[i]
        if t[0] == "|":
            if beat is not None:
                grid.append(beat)
            if not t[1:].isdigit():
                return None
            beat = [None] * int(t[1:])
            i += 1
        elif t[0] == "p":
            if not t[1:].isdigit():
                return None
            pos = int(t[1:])
            if pos >= len(beat) or i + 1 >= len(toks) or toks[i + 1] not in GU:
                return None
            if beat[pos] is not None:
                return None
            beat[pos] = toks[i + 1]
            i += 2
        else:
            return None
    grid.append(beat)
    return grid


# ── 생성·측정 ────────────────────────────────────────────────────────────────

def load(scheme, dev):
    ck = torch.load(RUNS / f"{scheme}.pt", map_location=dev, weights_only=False)
    cfg = ck["cfg"]
    m = TinyGPT(len(ck["vocab"]), ck["maxlen"], ck["ids"]["pad"],
                cfg["d"], cfg["h"], cfg["layers"], cfg["dropout"]).to(dev)
    m.load_state_dict(ck["state"])
    m.eval()
    stoi = {t: i for i, t in enumerate(ck["vocab"])}
    return m, ck["vocab"], stoi, ck["ids"]


def measure(scheme, patterns, dev, valid_totals, valid_shapes):
    m, vocab, stoi, ids = load(scheme, dev)
    parse = v1_parse if scheme == "v1" else v2_parse
    train_set = {tuple(ENCODERS[scheme](p["grid"])) for p in patterns}

    rows = []
    for temp in TEMPS:
        torch.manual_seed(SEED)
        ok = cyc = shp = nov = 0
        samples = []
        for _ in range(N):
            idx = torch.tensor([[ids["bos"]]], device=dev)
            out = m.generate(idx, max_new=m.block_size, eos_id=ids["eos"], temperature=temp)
            toks = [vocab[i] for i in out[0].tolist()]
            toks = [t for t in toks if t not in (BOS, EOS, PAD)]
            parsed = parse(toks)
            if parsed is None:
                continue
            ok += 1
            total = len(parsed) if scheme == "v1" else sum(len(b) for b in parsed)
            if total in valid_totals:
                cyc += 1
            if scheme == "v2" and tuple(len(b) for b in parsed) in valid_shapes:
                shp += 1
            if tuple(toks) not in train_set:
                nov += 1
            samples.append(toks)
        rows.append({
            "temp": temp,
            "parse_%": round(100 * ok / N, 1),
            "cycle_%": round(100 * cyc / N, 1),
            "shape_%": round(100 * shp / N, 1) if scheme == "v2" else None,
            "novel_%": round(100 * nov / N, 1),
        })
    return rows, samples


if __name__ == "__main__":
    dev = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    patterns = json.loads(DATA.read_text(encoding="utf-8"))["patterns"]
    valid_totals = {sum(len(b) for b in p["grid"]) for p in patterns}
    valid_shapes = {tuple(len(b) for b in p["grid"]) for p in patterns}
    print(f"device={dev}  N={N}/temp  valid_totals={sorted(valid_totals)}  "
          f"valid_shapes={sorted(valid_shapes)}\n")

    report = {}
    for scheme in ("v1", "v2"):
        rows, samples = measure(scheme, patterns, dev, valid_totals, valid_shapes)
        report[scheme] = rows
        print(f"[{scheme}]")
        for r in rows:
            shp = f" shape={r['shape_%']:5}" if r["shape_%"] is not None else ""
            print(f"  temp {r['temp']}  parse={r['parse_%']:5}%  "
                  f"cycle={r['cycle_%']:5}%{shp}  novel={r['novel_%']:5}%")
        print(f"  예시 생성(temp {TEMPS[-1]}): {' '.join(samples[0]) if samples else '—'}\n")

    (RUNS / "compare.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print("saved → runs/compare.json")

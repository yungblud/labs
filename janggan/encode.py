"""장단 그리드(중립 IR) → 두 가지 토큰 인코딩.

논문(arXiv 2408.01096)의 핵심 축을 재현한다:
  v1 (flat / duration) — 길이를 토큰으로. 박 구조 없이 평탄화. 어휘에 길이 토큰이 들어감.
  v2 (positional / 정간보식) — 길이를 위치로. 칸 경계 |N + 위치 + 구음. 어휘가 위치 등분에 bounded.

두 인코딩 모두 그리드로 왕복(round-trip)한다. 단 v1 은 박 경계를 토큰에 담지 않으므로
그리드로 되돌리려면 박별 길이(beat_lens)라는 외부 정보가 필요하다 — 이 비대칭 자체가
"길이식 인코딩은 박 구조를 잃는다"는 논문 주장의 축소판이다.

토큰 표기:
  구음     deong kung deok gideok roll
  v1 길이  d1 d2 d3 ...        (지속 셀 수)
  v1 쉼    rest                (구음 자리의 빈 시작)
  v2 칸    |2 |3 ...            (칸 경계 + 등분수)
  v2 위치  p0 p1 p2 ...        (칸 안 위치 index)
"""

from __future__ import annotations

import json
from pathlib import Path

DATA = Path(__file__).with_name("data") / "patterns.json"


# ── 공통 ────────────────────────────────────────────────────────────────────

def flatten(grid):
    """grid → (셀 평탄 리스트, 박별 길이)."""
    flat = [cell for beat in grid for cell in beat]
    beat_lens = [len(beat) for beat in grid]
    return flat, beat_lens


def reshape(flat, beat_lens):
    """평탄 셀 리스트 → 박별 그리드 (박 길이 외부 주입)."""
    out, i = [], 0
    for length in beat_lens:
        out.append(flat[i:i + length])
        i += length
    return out


# ── v1: flat / duration ─────────────────────────────────────────────────────

def v1_encode(grid):
    flat, _ = flatten(grid)
    n = len(flat)
    stroke_idx = [k for k, c in enumerate(flat) if c is not None]
    toks = []
    if not stroke_idx:                       # 전부 쉼(방어)
        return ["rest", f"d{n}"]
    if stroke_idx[0] > 0:                     # 선행 쉼
        toks += ["rest", f"d{stroke_idx[0]}"]
    for j, k in enumerate(stroke_idx):
        end = stroke_idx[j + 1] if j + 1 < len(stroke_idx) else n
        toks += [flat[k], f"d{end - k}"]      # 구음 + 지속 길이
    return toks


def v1_decode(toks):
    """→ 평탄 셀 리스트. (박 그리드는 reshape 로, beat_lens 필요)"""
    flat = []
    for a in range(0, len(toks), 2):
        sym, dur = toks[a], int(toks[a + 1][1:])
        flat.append(None if sym == "rest" else sym)
        flat += [None] * (dur - 1)
    return flat


# ── v2: positional / 정간보식 ────────────────────────────────────────────────

def v2_encode(grid):
    toks = []
    for beat in grid:
        toks.append(f"|{len(beat)}")          # 칸 경계 + 등분수
        for pos, cell in enumerate(beat):
            if cell is not None:
                toks += [f"p{pos}", cell]     # 위치 + 구음 (빈 셀은 생략)
    return toks


def v2_decode(toks):
    """→ 박 그리드. 자기완결(외부 정보 불필요)."""
    grid, beat, i = [], None, 0
    while i < len(toks):
        t = toks[i]
        if t[0] == "|":
            if beat is not None:
                grid.append(beat)
            beat = [None] * int(t[1:])
            i += 1
        elif t[0] == "p":
            beat[int(t[1:])] = toks[i + 1]
            i += 2
        else:
            i += 1
    if beat is not None:
        grid.append(beat)
    return grid


# ── 검증·리포트 ──────────────────────────────────────────────────────────────

def _load():
    return json.loads(DATA.read_text(encoding="utf-8"))["patterns"]


def verify(patterns):
    """모든 패턴에 대해 두 인코딩 왕복 검증. 실패 리스트 반환."""
    fails = []
    for p in patterns:
        grid = p["grid"]
        _, beat_lens = flatten(grid)
        rt1 = reshape(v1_decode(v1_encode(grid)), beat_lens)
        rt2 = v2_decode(v2_encode(grid))
        if rt1 != grid:
            fails.append((p["name"], p["variant"], "v1"))
        if rt2 != grid:
            fails.append((p["name"], p["variant"], "v2"))
    return fails


def vocab(patterns, encode):
    v = set()
    for p in patterns:
        v.update(encode(p["grid"]))
    return v


if __name__ == "__main__":
    patterns = _load()

    fails = verify(patterns)
    print(f"round-trip: {len(patterns)} patterns × 2 인코딩 → 실패 {len(fails)}")
    for f in fails:
        print("  FAIL", f)

    v1v, v2v = vocab(patterns, v1_encode), vocab(patterns, v2_encode)
    dur = sorted(t for t in v1v if t.startswith("d") and t[1:].isdigit())
    print(f"\nvocab  v1(flat/duration) = {len(v1v):2d}   v2(positional) = {len(v2v):2d}")
    print(f"  v1 길이 토큰: {dur}  (길이 범위가 어휘로 들어감)")
    print(f"  v2 어휘     : {sorted(v2v)}")

    sample = next(p for p in patterns if p["name"] == "eotmori" and p["variant"] == "basic")
    print(f"\n엇모리 basic (비대칭 3+2+3+2):")
    print("  v1:", " ".join(v1_encode(sample["grid"])))
    print("  v2:", " ".join(v2_encode(sample["grid"])))

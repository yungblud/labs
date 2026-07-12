"""장단 데이터셋 빌더.

각 변주를 박별 shorthand 문자열(한 글자=한 셀)로 손저작하고, 중립 IR(grid)로
펼쳐 patterns.json 을 생성한다. 변형은 사람이 고른 것이며 랜덤이 아니다.
표기 규칙 정본: NOTATION.md.

실행: python build_dataset.py  (→ patterns.json 덮어씀)
"""

import json
from pathlib import Path

# shorthand 글자 → 구음 (또는 null)
LEGEND = {"D": "deong", "K": "kung", "T": "deok", "G": "gideok", "R": "roll", ".": None}
GU = ["deong", "kung", "deok", "gideok", "roll"]

# 장단별 변주 — 각 변주는 박(칸) 문자열 리스트. 박 길이(=위치 등분)는 장단 정체성으로 고정.
#   굿거리 4박×3 / 자진모리 4박×3 / 세마치 3박×3 / 엇모리 3+2+3+2
JANGDAN = {
    "gutgeori": [
        ["D.G", "K.R", "D.G", "K.R"],  # basic
        ["D.G", "K.R", "D.G", "KR."],
        ["DKG", "K.R", "D.G", "K.R"],
        ["D.G", "K.T", "D.G", "K.R"],
        ["D.G", "K.R", "DKG", "K.R"],
        ["D..", "K.R", "D.G", "K.R"],
        ["D.G", "K.R", "D.T", "K.R"],
        ["DTG", "K.R", "DTG", "K.R"],
        ["D.G", "K.R", "D.G", "KTR"],
    ],
    "jajinmori": [
        ["DTK", "T.K", "DTK", "TKT"],  # basic
        ["DTK", "TTK", "DTK", "TKT"],
        ["DTK", "T.K", "DTK", "TKK"],
        ["DTG", "T.K", "DTK", "TKT"],
        ["DTK", "T.K", "DGK", "TKT"],
        ["DKK", "T.K", "DTK", "TKT"],
        ["DTK", "TTK", "DTK", "TKG"],
        ["DTK", "T.K", "DTK", "T.T"],
        ["DTK", "TGK", "DTK", "TKT"],
    ],
    "semachi": [
        ["D.K", "T.K", "DT."],  # basic
        ["D.K", "T.K", "DTG"],
        ["DGK", "T.K", "DT."],
        ["D.K", "TTK", "DT."],
        ["D.K", "T.K", "DTK"],
        ["D.G", "T.K", "DT."],
        ["DKK", "T.K", "DT."],
        ["D.K", "TGK", "DT."],
    ],
    "eotmori": [
        ["D.K", "TK", "D.G", "TK"],  # basic (3+2+3+2)
        ["D.K", "TK", "D.G", "TT"],
        ["DGK", "TK", "D.G", "TK"],
        ["D.K", "GK", "D.G", "TK"],
        ["D.K", "TK", "DGG", "TK"],
        ["D.K", "TK", "D.K", "TK"],
        ["DKK", "TK", "D.G", "TK"],
        ["D.K", "TK", "D.G", "KT"],
    ],
}


def expand(beat_strs):
    """박 문자열 리스트 → grid(비대칭이면 박마다 길이 다름)."""
    grid = []
    for bs in beat_strs:
        cells = []
        for ch in bs:
            if ch not in LEGEND:
                raise ValueError(f"unknown shorthand char: {ch!r} in {bs!r}")
            cells.append(LEGEND[ch])
        grid.append(cells)
    return grid


def subdivisions(grid):
    """대칭이면 int, 비대칭이면 박별 리스트."""
    lens = [len(b) for b in grid]
    return lens[0] if len(set(lens)) == 1 else lens


def build():
    patterns = []
    for name, variants in JANGDAN.items():
        for i, beat_strs in enumerate(variants):
            grid = expand(beat_strs)
            patterns.append({
                "name": name,
                "variant": "basic" if i == 0 else f"v{i}",
                "beats": len(grid),
                "subdivisions": subdivisions(grid),
                "grid": grid,
            })
    return {
        "meta": {
            "gu": GU,
            "note": "연구용 양식화 근사치. 정확한 채보 아님. build_dataset.py 로 생성. 규칙=NOTATION.md.",
        },
        "patterns": patterns,
    }


if __name__ == "__main__":
    out = Path(__file__).with_name("patterns.json")
    data = build()
    out.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(data['patterns'])} patterns → {out.name}")

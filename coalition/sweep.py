"""위협현저성 스윕 — fracture point & 이탈 순서 측정.

스펙 ③④: s 를 연속으로 낮추며 매 스텝 연합을 재수렴, 언제/누가 깨지는지 기록.
- 기저 스윕:   두 위협을 함께 1.0→0.0
- 비대칭 스윕: china 만 낮춤 / cultural 만 낮춤 (다른 적은 1.0 고정)
결과는 사람이 읽는 JSON 으로 덤프 + 콘솔 요약.

실행: python coalition/sweep.py   (기본 출력: coalition/sweep_result.json)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import AXES, THREATS, Faction, build_factions, resolve_coalition


def _steps(n: int = 51) -> list[float]:
    """1.0 → 0.0 을 n개 점으로 (기본 0.02 간격)."""
    return [round(float(x), 4) for x in np.linspace(1.0, 0.0, n)]


def run_sweep(
    factions: list[Faction],
    axis: str,            # "joint" | "china" | "cultural"
    w: float = 1.5,
    win_threshold: float = 0.5,
    n: int = 51,
) -> dict:
    """한 종류의 스윕을 돌려 스텝별 연합 상태 + 파생 지표를 반환.

    axis="joint"     : s_china = s_cultural = s (함께 낮춤)
    axis="china"     : s_china = s,  s_cultural = 1.0 (문화위협 상존, 중국위협만 소멸)
    axis="cultural"  : s_cultural = s, s_china = 1.0 (중국위협 상존, 문화위협만 소멸)
    """
    ci, cu = THREATS.index("china"), THREATS.index("cultural")
    records = []
    for s in _steps(n):
        vec = np.ones(len(THREATS))
        if axis == "joint":
            vec[ci] = vec[cu] = s
        elif axis == "china":
            vec[ci] = s
        elif axis == "cultural":
            vec[cu] = s
        else:
            raise ValueError(f"unknown axis: {axis}")
        r = resolve_coalition(factions, vec, w=w, win_threshold=win_threshold)
        records.append({
            "s": s,
            "members": r.members,
            "n_members": len(r.members),
            "total_mass": round(r.total_mass, 3),
            "winning": r.winning,
        })

    all_names = [f.name for f in factions]
    # 파생: 각 파벌이 이탈하는 s (내려가며 처음 사라지는 s). 끝까지 있으면 None.
    exit_s = {}
    for name in all_names:
        gone_at = None
        for rec in records:  # s 내림차순
            if name not in rec["members"]:
                gone_at = rec["s"]
                break
        exit_s[name] = gone_at
    # 이탈 순서: 먼저(높은 s에서) 나간 순. None(끝까지 생존)은 뒤로.
    exit_order = sorted(all_names, key=lambda x: (exit_s[x] is None, -(exit_s[x] or -1)))
    # fracture point: 내려가며 winning 이 처음 False 가 되는 s.
    fracture_s = next((rec["s"] for rec in records if not rec["winning"]), None)

    return {
        "axis": axis,
        "fracture_s": fracture_s,
        "exit_order": [{"name": n, "exit_s": exit_s[n]} for n in exit_order],
        "records": records,
    }


def _fmt_exit(order: list[dict]) -> str:
    parts = []
    for e in order:
        tag = "생존" if e["exit_s"] is None else f"s={e['exit_s']:.2f}"
        parts.append(f"{e['name']}({tag})")
    return " → ".join(parts)


def main() -> None:
    factions = build_factions()
    total = sum(f.mass for f in factions)
    params = {
        "w": 1.5,
        "win_threshold": 0.5,
        "total_mass": total,
        "win_mass_needed": round(0.5 * total, 2),
        "axes": list(AXES),
        "threats": list(THREATS),
    }

    out = {"params": params}
    print(f"총 mass={total:.1f}, 승리 필요 mass={0.5 * total:.1f} (전체의 50%)\n")

    for axis, label in [("joint", "기저(두 적 함께 ↓)"),
                        ("china", "비대칭: 중국위협만 ↓ (문화위협 상존)"),
                        ("cultural", "비대칭: 문화위협만 ↓ (중국위협 상존)")]:
        res = run_sweep(factions, axis)
        out[axis] = res
        fs = res["fracture_s"]
        print(f"── {label}")
        print(f"   fracture s* = {fs if fs is not None else '(안 깨짐)'}"
              f"   {'← 이 지점서 연합이 승리크기 아래로' if fs is not None else ''}")
        print(f"   이탈 순서: {_fmt_exit(res['exit_order'])}\n")

    dst = Path(__file__).with_name("sweep_result.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"→ 전체 스텝 기록 저장: {dst.relative_to(Path.cwd()) if dst.is_relative_to(Path.cwd()) else dst}")


if __name__ == "__main__":
    main()

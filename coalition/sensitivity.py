"""감도 분석 — 결론의 견고성 확인.

검증할 두 결론:
  (A) 공동위협만으로 묶인 과대연합은 s→0 에서 붕괴한다 (joint fracture 존재).
  (B) 거울상 비대칭: 중국위협만 끄면 테크우파가 떨어지고 문화연합이 남고,
      문화위협만 끄면 그 반대. (두 날개가 서로 다른 적에 묶임)

이게 손으로 박은 값(w=1.5, 이상점 배치)의 우연이 아닌지 흔들어 본다:
  ① 마찰강도 w 그리드
  ② 이상점·stake 에 가우시안 노이즈 준 몬테카를로

실행: python coalition/sensitivity.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import THREATS, Faction, build_factions, resolve_coalition

TECH_WING = {"Thiel/NatCon", "Musk/DOGE", "Karp/Palantir", "Crypto"}
CULTURE_WING = {"Evangelical", "MAGA-populist"}
S_LOW = 0.2  # 붕괴 후를 보는 저(低)현저성 지점


def _vec(china: float, cultural: float) -> np.ndarray:
    v = np.ones(len(THREATS))
    v[THREATS.index("china")] = china
    v[THREATS.index("cultural")] = cultural
    return v


def _survival(members: list[str], wing: set[str]) -> float:
    """wing 중 몇 %가 살아남았나."""
    return len(wing & set(members)) / len(wing)


def joint_collapses(factions, w) -> bool:
    """(A) 두 적을 함께 0으로 내리면 승리크기 아래로 붕괴하는가."""
    r = resolve_coalition(factions, _vec(0.0, 0.0), w=w)
    return not r.winning


def mirror_holds(factions, w) -> tuple[bool, dict]:
    """(B) 거울상 비대칭이 성립하는가.

    china↓ 에선 문화날개 생존율 > 테크날개 생존율,
    cultural↓ 에선 테크날개 생존율 > 문화날개 생존율 이면 성립.
    """
    m_china = resolve_coalition(factions, _vec(S_LOW, 1.0), w=w).members
    m_cult = resolve_coalition(factions, _vec(1.0, S_LOW), w=w).members
    tc, cc = _survival(m_china, TECH_WING), _survival(m_china, CULTURE_WING)
    tk, ck = _survival(m_cult, TECH_WING), _survival(m_cult, CULTURE_WING)
    ok = (cc > tc) and (tk > ck)
    detail = {
        "china_down": {"tech_surv": tc, "culture_surv": cc, "members": m_china},
        "cultural_down": {"tech_surv": tk, "culture_surv": ck, "members": m_cult},
    }
    return ok, detail


def jitter(factions: list[Faction], sigma: float, rng: np.random.Generator) -> list[Faction]:
    """이상점·stake 에 N(0,sigma) 노이즈 후 [0,1] 클립. mass 는 유지."""
    out = []
    for f in factions:
        ideal = np.clip(f.ideal + rng.normal(0, sigma, f.ideal.shape), 0, 1)
        stake = np.clip(f.stake + rng.normal(0, sigma, f.stake.shape), 0, 1)
        out.append(Faction(f.name, ideal, stake, f.mass, f.note))
    return out


def main() -> None:
    base = build_factions()
    out: dict = {}

    # ── ① w 그리드 ──────────────────────────────────────────────────────────
    print("① 마찰강도 w 흔들기 (이상점·stake 는 기본값 고정)\n")
    w_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    grid_rows = []
    for w in w_grid:
        A = joint_collapses(base, w)
        B, det = mirror_holds(base, w)
        grid_rows.append({"w": w, "joint_collapses": A, "mirror_holds": B,
                          "china_down_survivors": det["china_down"]["members"],
                          "cultural_down_survivors": det["cultural_down"]["members"]})
        print(f"   w={w:<4}  붕괴(A)={'O' if A else 'X'}  거울상(B)={'O' if B else 'X'}"
              f"   | china↓생존={det['china_down']['members']}"
              f"   cultural↓생존={det['cultural_down']['members']}")
    out["w_grid"] = grid_rows

    # ── ② 몬테카를로 노이즈 ──────────────────────────────────────────────────
    print("\n② 이상점·stake 노이즈 몬테카를로 (w=1.5 고정, seed=0)\n")
    rng = np.random.default_rng(0)
    mc_rows = []
    for sigma in [0.05, 0.10, 0.15, 0.20]:
        n = 1000
        a_cnt = b_cnt = 0
        for _ in range(n):
            fj = jitter(base, sigma, rng)
            if joint_collapses(fj, 1.5):
                a_cnt += 1
            if mirror_holds(fj, 1.5)[0]:
                b_cnt += 1
        row = {"sigma": sigma, "n": n,
               "joint_collapse_rate": round(a_cnt / n, 3),
               "mirror_hold_rate": round(b_cnt / n, 3)}
        mc_rows.append(row)
        print(f"   σ={sigma:<5} (n={n})  붕괴(A) {a_cnt/n:6.1%}   거울상(B) {b_cnt/n:6.1%}")
    out["monte_carlo"] = mc_rows

    dst = Path(__file__).with_name("sensitivity_result.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 저장: {dst.name}")


if __name__ == "__main__":
    main()

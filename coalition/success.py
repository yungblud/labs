"""기술공화국 기획 성공확률 — 2요인 몬테카를로 + 분산분해.

성공 = 정책 포획(capture) ≥ θ. 두 확률 노브를 각각/함께 흔들어 P(성공)을 재고,
"성공을 지배하는 건 창길이인가 구조인가"를 분산분해로 가른다.
스펙: specs/coalition-success.md

    (W) 궤적-only : s(t) 랜덤, 구조 고정        → 창길이가 성공을 흔드는 정도
    (S) 구조-only : 궤적 고정(기저), w·이상점·stake 랜덤 → 구조가 흔드는 정도
    (B) 둘다      : P(성공) 기저 추정치

실행: python coalition/success.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from model import build_factions
from sensitivity import jitter  # 구조 노이즈(이상점·stake) 재사용
from trajectory import (
    TECH_REPUBLIC,
    baseline_trajectory,
    capture_over,
    proximity,
    sample_trajectory,
)

N = 500
HORIZON = 40
# θ — 포획 성공 임계. 기저 궤적 capture=7.54 를 살짝 밑도는 값으로 잡아, "기저보다 창이 짧거나
# 구조가 험하면 실패"가 갈리게 캘리브레이션(아래 θ-스윕으로 곡선도 함께 본다).
THETA = 6.5


def _sample_w(rng: np.random.Generator) -> float:
    """구조 노브의 마찰항 — w ~ N(1.5, 0.4) 클립."""
    return float(np.clip(rng.normal(1.5, 0.4), 0.5, 3.0))


def run_once(rng, *, random_traj: bool, random_struct: bool):
    """한 판. 노브별로 궤적/구조를 랜덤화하거나 기저 고정."""
    traj = sample_trajectory(rng, HORIZON) if random_traj else baseline_trajectory(HORIZON)
    factions = jitter(build_factions(), 0.10, rng) if random_struct else build_factions()
    w = _sample_w(rng) if random_struct else 1.5
    run = capture_over(traj, factions, w=w)
    peak_prox = max((row["proximity"] for row in run.timeline if row["winning"]), default=0.0)
    win_steps = sum(1 for row in run.timeline if row["winning"])
    return {"capture": run.capture, "fracture_t": run.fracture_t,
            "peak_prox": peak_prox, "win_steps": win_steps}


def regime(name: str, *, random_traj: bool, random_struct: bool, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = [run_once(rng, random_traj=random_traj, random_struct=random_struct) for _ in range(N)]
    caps = np.array([r["capture"] for r in rows])
    p = float(np.mean(caps >= THETA))
    return {
        "regime": name, "n": N,
        "p_success": round(p, 3),
        "var_success": round(p * (1 - p), 4),      # 이진 결과의 분산
        "capture": {"min": round(float(caps.min()), 2), "median": round(float(np.median(caps)), 2),
                    "mean": round(float(caps.mean()), 2), "max": round(float(caps.max()), 2)},
        "peak_prox_mean": round(float(np.mean([r["peak_prox"] for r in rows])), 3),
        "win_steps_mean": round(float(np.mean([r["win_steps"] for r in rows])), 1),
        "_caps": caps,  # 내부용(θ-스윕), JSON 덤프 전 제거
    }


def theta_curve(caps_by_regime: dict[str, np.ndarray]) -> list[dict]:
    """P(성공) vs θ 곡선 — 단일 θ의 자의성을 줄이려 곡선으로 본다."""
    rows = []
    for th in [4.0, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0]:
        rows.append({"theta": th, **{name: round(float(np.mean(c >= th)), 3)
                                     for name, c in caps_by_regime.items()}})
    return rows


def main() -> None:
    out: dict = {"theta": THETA, "n": N, "horizon": HORIZON,
                 "target": dict(zip(("religion", "nativism", "market_dereg", "state_order"),
                                    [float(x) for x in TECH_REPUBLIC]))}

    # ── 기저 궤적 스냅샷 ─────────────────────────────────────────────────────
    base = capture_over(baseline_trajectory(HORIZON), build_factions())
    out["baseline"] = {
        "capture": round(base.capture, 2),
        "fracture_t": base.fracture_t,
        "success": base.capture >= THETA,
        "peak_prox": round(max(r["proximity"] for r in base.timeline if r["winning"]), 3),
        "timeline": base.timeline,
    }
    print(f"── 기저 궤적: capture={base.capture:.2f}  fracture_t={base.fracture_t}  "
          f"성공(θ={THETA})={base.capture >= THETA}")
    print(f"   포획 천장 peak_prox={out['baseline']['peak_prox']} (T*=1.0 에 못 미침 = MAGA mass 가 끌어당김)\n")

    # ── 2요인 몬테카를로 ─────────────────────────────────────────────────────
    W = regime("궤적-only", random_traj=True, random_struct=False, seed=1)
    S = regime("구조-only", random_traj=False, random_struct=True, seed=2)
    B = regime("둘다", random_traj=True, random_struct=True, seed=3)

    print(f"{'regime':<10} {'P(성공)':>8} {'var':>7} {'peak_prox':>10} {'win_steps':>10}  capture(med/mean)")
    for r in (W, S, B):
        print(f"{r['regime']:<10} {r['p_success']:>8.1%} {r['var_success']:>7.3f} "
              f"{r['peak_prox_mean']:>10.3f} {r['win_steps_mean']:>10.1f}  "
              f"{r['capture']['median']:.2f}/{r['capture']['mean']:.2f}")

    caps_by = {"traj_only": W.pop("_caps"), "struct_only": S.pop("_caps"), "both": B.pop("_caps")}
    out["regimes"] = [W, S, B]
    out["theta_curve"] = theta_curve(caps_by)

    # ── 분산분해(실패 귀속): 창길이 vs 구조 ─────────────────────────────────
    # 기저는 결정론적으로 성공(capture≥θ)이라, 각 노브 단독은 성공을 '깨뜨리는' 방향으로만 작동.
    # → 실패 귀속률 = 1−P(단독). p(1−p) 분산은 0.5 근접에 편향돼 지표로 부적합.
    fail_W, fail_S = round(1 - W["p_success"], 3), round(1 - S["p_success"], 3)
    dom = "창길이(window)" if fail_W > fail_S else "구조(structure)"
    out["failure_attribution"] = {"window": fail_W, "structure": fail_S, "dominant": dom}
    print(f"\n실패 귀속(기저는 결정론적 성공) → 성공을 더 깨뜨리는 노브: **{dom}**"
          f"  (창길이 {fail_W:.1%} vs 구조 {fail_S:.1%} 실패 유발)")

    print("\nθ-스윕 P(성공):")
    print(f"   {'θ':>5}  {'궤적-only':>9} {'구조-only':>9} {'둘다':>7}")
    for row in out["theta_curve"]:
        print(f"   {row['theta']:>5}  {row['traj_only']:>9.1%} {row['struct_only']:>9.1%} {row['both']:>7.1%}")

    dst = Path(__file__).with_name("success_result.json")
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n→ 저장: {dst.name}")


if __name__ == "__main__":
    main()

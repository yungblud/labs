"""기술공화국 기획 성공 실험 — 위협궤적 + 정책 포획 적분.

선행(model.py, 부정연합 붕괴)의 후속. 선행은 "적이 식으면 반드시 깨진다"까지 답했고,
여기선 그 다음 질문 — **무너지기 전에 기술공화국을 정책으로 포획했나** 를 잰다.
스펙: specs/coalition-success.md

핵심:
    capture = Σ_t  [winning_t] · proximity(center_t, T*) · Δt      # 집권 중 목표 근접의 적분
    성공 = capture ≥ θ

두 확률 노브:
    창길이  — s(t) 확률적 궤적(고점→감쇠+충격). 적이 얼마나 오래 사는가.
    구조    — w·이상점·stake 노이즈(sensitivity.jitter 재사용). 이념거리가 얼마나 험한가.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from model import AXES, THREATS, Faction, coalition_center, resolve_coalition

# ─────────────────────────────────────────────────────────────────────────────
# 기술공화국 목표점 T* — "기획이 성공했다면 정책이 놓일 자리"
# 테크우파의 국가주의 분파(Thiel·Karp) 비전 기준: 세속 + 중앙 기술관료·안보국가 + 시장 탈규제.
# (Musk·Crypto 의 반국가 편향은 같은 연합이지만 '다른 목표'라 T* 아님 — 이 긴장이 핵심 관찰의 씨앗.)
#   religion=0.25(세속) nativism=0.60(자국 일부) market_dereg=0.75(탈규제) state_order=0.90(안보국가)
# ─────────────────────────────────────────────────────────────────────────────
TECH_REPUBLIC = np.array([0.25, 0.60, 0.75, 0.90])
D_MAX = float(np.sqrt(len(AXES)))  # [0,1]^K 대각 길이 = proximity 정규화 상수


def proximity(center: np.ndarray, target: np.ndarray = TECH_REPUBLIC) -> float:
    """목표점 근접도 ∈ [0,1]. 1 = 정확히 목표, 0 = 이념공간 정반대 끝."""
    return 1.0 - float(np.linalg.norm(center - target)) / D_MAX


# ─────────────────────────────────────────────────────────────────────────────
# 위협궤적 — 창길이 노브. 두 위협(china, cultural)이 각각 독립 궤적.
#   s_t(t) = 1.0                      (t < t_peak)         고점 유지(적이 살아있는 창)
#          = exp(−k·(t − t_peak))     (t ≥ t_peak)         이후 감쇠
#          + 충격(선택): shock_t 에서 s 를 일시 재점화
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ThreatPath:
    t_peak: int          # 고점 유지 길이(창 지속)
    k: float             # 감쇠속도(클수록 빨리 식음)
    shock_t: int = -1    # 충격 시점(-1 = 없음)
    shock_mag: float = 0.0


def _path_value(p: ThreatPath, t: int) -> float:
    s = 1.0 if t < p.t_peak else float(np.exp(-p.k * (t - p.t_peak)))
    if p.shock_t >= 0 and t >= p.shock_t:
        s += p.shock_mag * float(np.exp(-p.k * (t - p.shock_t)))
    return float(np.clip(s, 0.0, 1.0))


@dataclass(frozen=True)
class Trajectory:
    china: ThreatPath
    cultural: ThreatPath
    horizon: int = 40

    def s_at(self, t: int) -> np.ndarray:
        v = np.zeros(len(THREATS))
        v[THREATS.index("china")] = _path_value(self.china, t)
        v[THREATS.index("cultural")] = _path_value(self.cultural, t)
        return v


def baseline_trajectory(horizon: int = 40) -> Trajectory:
    """결정론적 기저 궤적 — 두 적이 비대칭 감쇠(문화가 더 오래 산다: MAGA mass 가 크므로)."""
    return Trajectory(
        china=ThreatPath(t_peak=4, k=0.18),      # 중국위협: 짧은 고점 후 감쇠
        cultural=ThreatPath(t_peak=8, k=0.12),   # 문화위협: 더 오래·천천히 식음
        horizon=horizon,
    )


def sample_trajectory(rng: np.random.Generator, horizon: int = 40) -> Trajectory:
    """창길이 노브 — 두 위협의 (t_peak, k, 충격)을 따로 뽑는다(비대칭 유지)."""
    def one() -> ThreatPath:
        t_peak = int(rng.integers(2, 16))         # 창 2~15 스텝
        k = float(rng.uniform(0.05, 0.40))        # 감쇠 느림~빠름
        if rng.random() < 0.25:                   # 25% 확률로 재점화 충격
            return ThreatPath(t_peak, k, int(rng.integers(t_peak + 3, horizon)), float(rng.uniform(0.3, 0.8)))
        return ThreatPath(t_peak, k)
    return Trajectory(china=one(), cultural=one(), horizon=horizon)


# ─────────────────────────────────────────────────────────────────────────────
# 포획 적분 — 궤적을 따라 매 스텝 연합을 재수렴하고 winning 중 목표근접을 누적
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CaptureRun:
    capture: float                  # 누적 포획량
    fracture_t: int                 # 처음 winning 을 잃은 스텝(-1 = 끝까지 winning)
    timeline: list[dict]            # 스텝별 스냅샷(사람이 읽는 타임라인)


def capture_over(traj: Trajectory, factions: list[Faction], w: float = 1.5,
                 target: np.ndarray = TECH_REPUBLIC) -> CaptureRun:
    capture = 0.0
    fracture_t = -1
    timeline: list[dict] = []
    for t in range(traj.horizon):
        s = traj.s_at(t)
        r = resolve_coalition(factions, s, w=w)
        prox = proximity(r.center, target)
        step_cap = prox if r.winning else 0.0   # 집권 못하면 정책 못 박음 → 포획 0
        capture += step_cap
        if not r.winning and fracture_t < 0:
            fracture_t = t
        timeline.append({
            "t": t,
            "s": {THREATS[i]: round(float(s[i]), 3) for i in range(len(THREATS))},
            "winning": r.winning,
            "members": r.members,
            "center": {AXES[i]: round(float(r.center[i]), 3) for i in range(len(AXES))},
            "proximity": round(prox, 3),
            "capture_cum": round(capture, 3),
        })
    return CaptureRun(capture=capture, fracture_t=fracture_t, timeline=timeline)


# ─────────────────────────────────────────────────────────────────────────────
# 데모: 기저 궤적 한 판
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from model import build_factions

    factions = build_factions()
    run = capture_over(baseline_trajectory(), factions)
    print(f"T* = {dict(zip(AXES, TECH_REPUBLIC))}")
    print(f"capture={run.capture:.2f}  fracture_t={run.fracture_t}\n")
    print(f"{'t':>2} {'china':>6} {'cult':>6}  win  prox  cap    members")
    for row in run.timeline:
        print(f"{row['t']:>2} {row['s']['china']:>6} {row['s']['cultural']:>6}  "
              f"{'O' if row['winning'] else 'X'}   {row['proximity']:.2f}  "
              f"{row['capture_cum']:>5.2f}  {row['members']}")

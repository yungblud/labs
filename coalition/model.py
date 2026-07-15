"""부정연합 붕괴 시뮬레이션 — 최소 행위자 모델.

Riker(1962) size principle & negative coalition 재현용 장난감.
현실 예측이 아니라 "공동의 적으로만 묶인 과대연합의 불안정성"이라는 이론을 관찰하기 위한 모델.
스펙: specs/coalition-fracture.md

핵심 수식 (파벌 i 의 연합 잔류 효용):
    U_i(s) = Σ_t  s_t · stake_i[t]        # 위협방어 이득: 적이 셀수록·내가 그 적에 민감할수록 큼
           −  w · dist(ideal_i, center)   # 내부 마찰: 연합 정책중심과 내 이념거리
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# 이념축 (K=4). 각 값은 [0,1]. "번역(모델링) 지점" — 값의 근거는 각 파벌 주석에.
# ─────────────────────────────────────────────────────────────────────────────
AXES = (
    "religion",      # 세속 0 ↔ 종교전통 1
    "nativism",      # 세계주의 0 ↔ 자국우선 1
    "market_dereg",  # 친규제 0 ↔ 자유지상(시장 탈규제) 1
    "state_order",   # 탈중앙/반권위 0 ↔ 중앙집권 기술관료·안보국가 1
)

# 공동위협 두 축 (비대칭이 실험의 핵심). 각 파벌은 두 적에 대한 stake(민감도)를 가진다.
THREATS = (
    "china",     # 중국·기술패권 위협 → 테크우파가 크게 반응
    "cultural",  # 세계주의·문화좌파·딥스테이트 위협 → MAGA·복음주의가 크게 반응
)


@dataclass(frozen=True)
class Faction:
    name: str
    ideal: np.ndarray          # shape (len(AXES),) — 이념 이상점
    stake: np.ndarray          # shape (len(THREATS),) — 위협별 방어이득 민감도
    mass: float                # 정치적 무게(유권자·자금·플랫폼) — 승리크기 판정용
    note: str = ""             # 배치 근거 한 줄


def _f(name, ideal, stake, mass, note) -> Faction:
    return Faction(name, np.array(ideal, dtype=float), np.array(stake, dtype=float), mass, note)


def build_factions() -> list[Faction]:
    """현실 동맹을 6개 파벌로 축소. 이상점·stake·mass 는 손 배치(근거는 note).

    이념축 순서: (religion, nativism, market_dereg, state_order)
    위협 stake 순서: (china, cultural)
    """
    return [
        _f("Evangelical",   [0.95, 0.80, 0.45, 0.55], [0.20, 0.95], 1.5,
           "기독교 백인 보수: 종교전통 최고, 문화전쟁이 유일한 접착제(중국엔 둔감)"),
        _f("MAGA-populist", [0.65, 0.95, 0.40, 0.50], [0.45, 0.90], 2.0,
           "반엘리트·반이민 대중기반: 자국우선 최고, 문화+일자리로 중국도 일부 반응"),
        _f("Thiel/NatCon",  [0.25, 0.70, 0.85, 0.85], [0.95, 0.55], 1.0,
           "기술국가주의: 세속·탈규제·중앙기술관료, 중국을 문명존망으로 봄"),
        _f("Musk/DOGE",     [0.15, 0.50, 0.95, 0.25], [0.70, 0.45], 1.5,
           "관료제 해체·자유지상: 탈규제 최고 & 국가질서 최저(테크우파 내 이단아)"),
        _f("Karp/Palantir", [0.35, 0.55, 0.60, 0.90], [0.98, 0.40], 0.8,
           "국가안보 SaaS: 안보국가 최고, 중국방어가 곧 사업(문화엔 관심 낮음)"),
        _f("Crypto",        [0.10, 0.25, 0.95, 0.10], [0.55, 0.20], 0.8,
           "화폐주권·탈규제 헤지: 탈중앙 최저치, 접착점 거의 없음(가장 약한 고리 후보)"),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# 효용·잔류수렴
# ─────────────────────────────────────────────────────────────────────────────
def coalition_center(members: list[Faction]) -> np.ndarray:
    """잔류 파벌들의 mass-가중 이상점 평균 = 연합의 정책 위치."""
    if not members:
        return np.zeros(len(AXES))
    W = np.array([m.mass for m in members])
    X = np.stack([m.ideal for m in members])
    return (W[:, None] * X).sum(axis=0) / W.sum()


def utility(f: Faction, center: np.ndarray, s: np.ndarray, w: float) -> float:
    """U_i(s) = Σ s_t·stake_i[t]  −  w·||ideal_i − center||₂."""
    threat_gain = float(np.dot(s, f.stake))
    friction = w * float(np.linalg.norm(f.ideal - center))
    return threat_gain - friction


@dataclass
class Resolution:
    """주어진 위협현저성 s 에서 수렴한 연합 상태."""
    members: list[str]              # 잔류 파벌 이름
    exit_order: list[str]           # 이탈한 순서(먼저 나간 순)
    center: np.ndarray              # 잔존 연합의 정책 위치
    utilities: dict[str, float]     # 잔류 파벌의 최종 효용
    total_mass: float               # 잔류 mass 합
    winning: bool                   # 승리크기 임계 이상인가
    s: np.ndarray = field(default_factory=lambda: np.zeros(len(THREATS)))


def resolve_coalition(
    factions: list[Faction],
    s: np.ndarray,
    w: float = 1.5,
    win_threshold: float = 0.5,
) -> Resolution:
    """위협현저성 s 에서 연합을 수렴시킨다.

    결정론적 peeling: 매 반복마다 U<0 중 '가장 손해 큰'(U 최소) 파벌 한 명을 떼고
    center 를 다시 계산해 재평가. 모든 잔류자가 U≥0 이 되면 정지.
    한 명씩 떼므로 이탈 순서가 자연히 기록된다(가장 약한 고리부터).

    win_threshold: 전체 mass 대비 이 비율 이상이면 '승리 크기'로 본다(size principle 판정).
    """
    s = np.asarray(s, dtype=float)
    total = sum(f.mass for f in factions)
    members = list(factions)
    exit_order: list[str] = []

    while members:
        center = coalition_center(members)
        us = {f.name: utility(f, center, s, w) for f in members}
        worst_name = min(us, key=us.get)
        if us[worst_name] >= 0:
            break  # 전원 잔류 이유 있음 → 수렴
        # 가장 손해 큰 파벌 이탈
        members = [f for f in members if f.name != worst_name]
        exit_order.append(worst_name)

    center = coalition_center(members)
    us = {f.name: utility(f, center, s, w) for f in members}
    mass = sum(f.mass for f in members)
    return Resolution(
        members=[f.name for f in members],
        exit_order=exit_order,
        center=center,
        utilities=us,
        total_mass=mass,
        winning=mass >= win_threshold * total,
        s=s,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 데모: 위협 최고(1,1) vs 위협 소멸(0,0)
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    factions = build_factions()
    total = sum(f.mass for f in factions)
    print(f"factions={len(factions)}, total_mass={total:.1f}, axes={AXES}, threats={THREATS}\n")

    for label, s in [("위협 최고  s=(china=1, cultural=1)", [1.0, 1.0]),
                     ("위협 소멸  s=(china=0, cultural=0)", [0.0, 0.0])]:
        r = resolve_coalition(factions, np.array(s))
        print(f"── {label}")
        print(f"   잔류({r.total_mass:.1f}/{total:.1f}, winning={r.winning}): {r.members}")
        print(f"   이탈 순서: {r.exit_order or '(없음)'}")
        print(f"   연합중심: {dict(zip(AXES, [round(float(x), 2) for x in r.center]))}")
        print(f"   효용: { {k: round(v, 2) for k, v in r.utilities.items()} }\n")

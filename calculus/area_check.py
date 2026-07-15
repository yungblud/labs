"""리만 합 수렴 검산 (파트 4 보조) — 순수 표준 라이브러리.

g(x)=2x 의 x=0..3 아래 넓이를, 막대기 n 개로 채워 더한다 (리만 합).
n→∞ 일 때 무엇으로 향하나? 손검산(삼각형 ½·3·6 = 9)과 대조.
왼끝/오른끝/중점 세 방식이 각각 얼마나 빨리 9 에 붙는지도 관찰.
"""


def g(x):
    return 2 * x


def riemann(a, b, n, kind):
    dx = (b - a) / n
    total = 0.0
    for i in range(n):
        if kind == "left":
            x = a + i * dx            # 막대 왼쪽 끝 높이
        elif kind == "right":
            x = a + (i + 1) * dx      # 막대 오른쪽 끝 높이
        else:  # mid
            x = a + (i + 0.5) * dx    # 막대 중점 높이
        total += g(x) * dx            # (높이 × 폭) 을 전부 더함
    return total


if __name__ == "__main__":
    a, b, TRUE = 0.0, 3.0, 9.0
    print("∫₀³ 2x dx  (손검산: 삼각형 ½·3·6 = 9)\n")
    print(f"{'n':>8} | {'left':>12} | {'right':>12} | {'mid':>12}")
    print("-" * 52)
    for n in (10, 100, 1000, 10000):
        L = riemann(a, b, n, "left")
        R = riemann(a, b, n, "right")
        M = riemann(a, b, n, "mid")
        print(f"{n:>8} | {L:>12.6f} | {R:>12.6f} | {M:>12.6f}")
    print(f"\n참값 → {TRUE}")

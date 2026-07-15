"""극한 수렴 검산 (파트 1~2 보조) — 순수 표준 라이브러리.

x=2 에서 f(x)=x² 의 순간 기울기를, 간격 h 를 줄이며 두 점 기울기로 근사한다.
정의: slope(h) = ( f(2+h) - f(2) ) / h  →  h→0 일 때 무엇으로 향하나?
손유도 결과(2x → f'(2)=4)와 대조하고, h 가 너무 작을 때의 부동소수 흔들림도 관찰.
"""


def f(x):
    return x * x


def slope(x, h):
    return (f(x + h) - f(x)) / h


if __name__ == "__main__":
    x = 2.0
    print(f"x={x} 에서 x² 의 순간 기울기 (손유도 정답: 2x = 4)\n")
    print(f"{'h':>12} | {'slope(h)':>18} | {'|오차|':>12}")
    print("-" * 48)
    # h 를 1 부터 0 쪽으로 계속 줄인다
    for k in range(0, 16):
        h = 10.0 ** (-k)
        s = slope(x, h)
        err = abs(s - 4.0)
        print(f"{h:>12.0e} | {s:>18.12f} | {err:>12.2e}")

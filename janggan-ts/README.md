# janggan-ts — 장단 인코딩 실험 (TypeScript 이식)

파이썬 [`../janggan`](../janggan) 실험의 TS 포트. 스펙: [`../specs/janggan-ts-port.md`](../specs/janggan-ts-port.md).
"TS로도 이 파이프라인이 돈다"를 실제 동작으로 증명하는 게 목적.

## 스택

- **Node 22 + tsx** (TS 직접 실행), 패키지매니저 **pnpm**
- **순수 TS**: 인코딩·데이터·지표 (`tokens.ts` `buildDataset.ts` `encode.ts` `dataset.ts` `compare.ts`)
- **`@tensorflow/tfjs-node`**: from-scratch 학습부 (`model.ts` `train.ts`) — CPU

## 실행

```bash
pnpm install          # tfjs-node 네이티브 빌드 포함
pnpm build:data       # 장단 34개 → data/patterns.json
pnpm encode           # 왕복 검증 + 어휘 비교 (파이썬과 동일 출력)
pnpm train            # v1/v2 tfjs 학습 → runs/*.model.json
pnpm compare          # 생성 + 구조 유효성 → runs/compare.json
```

## 파이썬 대비 의도된 차이

| | 파이썬 | TS |
| --- | --- | --- |
| 프레임워크 | PyTorch | tfjs-node |
| 가속 | MPS(Mac GPU) | CPU |
| 배치 | full-batch + 패딩 | batch=1 가변길이(패딩 없음) |
| attention | 4-head | single-head, d=32 |

→ 순수 TS 부분(인코딩·데이터)은 **파이썬과 바이트 단위로 동일**(patterns.json·어휘 8/10·왕복 실패 0).
학습부는 위 차이로 **모델이 더 약함** → loss/지표 수치는 다름.

## 흥미로운 관찰 — 약한 모델에서 오히려 v2 이점이 드러남

파이썬 실험 결론은 "이 장단 레짐에선 인코딩 차이가 거의 없다"였다(강한 모델이 둘 다 암기).
그런데 **더 약한 TS 모델에선 v2 의 구조 이점이 나타났다:**

| temp 1.0 cycle 유효율 | v1 | v2 |
| --- | --- | --- |
| 파이썬 (강한 모델) | 100% | 99.5% |
| **TS (약한 모델)** | **61.5%** | **90%** |

약한 모델은 v1(박 경계 없음)에서 언더핏으로 구조가 무너지지만(cycle 61.5%, novel 58%로 방황),
v2 는 `|N` 칸 경계가 토큰에 박혀 있어 약한 모델에서도 박 구조를 유지한다(90%).

→ **"학습이 어려울수록(모델·데이터가 빈약할수록) 위치 인코딩의 구조 scaffolding 이 더 중요"** —
이는 파이썬 FINDINGS 의 **레짐 의존성** 결론과 정확히 정합하며, 오히려 논문 주장(적은 자원에서
좋은 인코딩이 도움)의 방향을 약한-모델 축에서 재현한 셈이다.

> 두 구현을 나란히 두니, 파이썬만으로는 안 보이던 "인코딩 효과는 학습 난이도에 의존한다"가
> 드러났다. 포트가 단순 재현을 넘어 관찰을 하나 더 준 것.

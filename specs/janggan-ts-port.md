# janggan TS 포트 — 파이썬 실험의 TypeScript 이식

> 원본: [`janggan-encoding-research.md`](./janggan-encoding-research.md) (파이썬, 완결).
> 목적: **"TS로도 된다"를 실제 동작 코드로 증명.** 새 과학이 아니라 파이프라인 이식.
> 결론(레짐 의존성)은 이미 났으므로 TS판은 재현 데모.

## 스택 (확정)

- **런타임:** Node 22 + `tsx`(TS 직접 실행). 패키지매니저 **pnpm**.
- **학습부:** `@tensorflow/tfjs-node` — JS에서 from-scratch 학습되는 유일 성숙 라이브러리.
- **나머지(인코딩·데이터·지표):** 순수 TS (프레임워크 없음).

## 파이썬 → TS 매핑

| 파이썬 | TS | 비고 |
| --- | --- | --- |
| `build_dataset.py` | `src/buildDataset.ts` | shorthand→그리드, 동일 patterns.json 생성 |
| `encode.py` | `src/encode.ts` | v1/v2 인코더·디코더·왕복 (거의 1:1) |
| `dataset.py` | `src/dataset.ts` | vocab·시퀀스화 |
| `model.py` `train.py` | `src/model.ts` `src/train.ts` | tfjs GPT + 학습 루프 |
| `compare.py` | `src/compare.ts` | 생성 + 구조 유효성 |

## 의도된 차이 (파이썬과 100% 동일 아님)

- **단일 배치(batch=1) 가변길이 학습** — 패딩·padding-mask 제거해 TS 이식 단순화. (파이썬은 full-batch 패딩)
- **CPU 학습** — tfjs-node는 Mac MPS 미사용. 모델 작아 문제없음.
- 위 차이로 loss/지표 수치는 파이썬과 정확히 같지 않으나 **파이프라인·결론 방향은 동일** 확인이 목표.

## 체크리스트 ✅ 전부 완료

- [x] 1 · pnpm 스캐폴드 + tfjs-node 네이티브 빌드 + 타입체크 통과
- [x] 2 · `buildDataset.ts` → patterns.json (**파이썬과 바이트 동일** 34개)
- [x] 3 · `encode.ts` + 왕복 검증 **실패 0** (어휘 8/10 파이썬 동일)
- [x] 4 · `model.ts`(tfjs TinyGPT, single-head) + `train.ts` (v1/v2, loss v1=0.40·v2=0.29)
- [x] 5 · `compare.ts` (생성·지표 → `runs/compare.json`)
- [x] 6 · `janggan-ts/README.md` (실행법·파이썬 대비·관찰)

> **관찰(포트가 준 추가 발견):** 약한 TS 모델에선 temp 1.0 cycle 유효율 v1=61.5% ≪ v2=90%.
> 파이썬(강한 모델)에선 안 보이던 v2 구조 이점이 드러남 → "인코딩 효과는 학습 난이도에 의존"
> = 파이썬 FINDINGS 의 레짐 의존성 결론과 정합. 상세 `janggan-ts/README.md`.

## 범위

| 영역 | 상태 |
| --- | --- |
| 순수 TS(인코딩·데이터·지표) | ✅ 목표 |
| tfjs 학습부 | ✅ 목표 |
| 파이썬과 수치 일치 | ❌ 비목표 (파이프라인 이식이 목표) |
| MIDI·시각화 | ❌ 범위 밖 |

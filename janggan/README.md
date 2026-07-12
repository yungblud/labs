# janggan — 장단 인코딩 연구 토이

한국 장단(리듬)을 소재로 **"적은 데이터에서 도메인 인코딩이 결과를 가른다"** 를 재현하는 실험.
설계·체크리스트 정본: [`../specs/janggan-encoding-research.md`](../specs/janggan-encoding-research.md).

## 계획 구조

```
janggan/
├── requirements.txt
├── data/            # 손코딩 장단 시퀀스(JSON) — 2단계
├── encode.py        # v1(flat) / v2(positional) 인코더·디코더 — 3단계
├── model.py         # 초소형 GPT (from scratch) — 4단계
├── train.py         # v1/v2 동일 조건 학습 — 4단계
└── compare.py       # 인코딩별 생성·지표 비교 — 5단계
```

## 셋업

```bash
cd janggan
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 진행 (완료)

- [x] 1단계 — 환경·디렉터리 뼈대
- [x] 2단계 — 장단 데이터 제작 (34 시퀀스)
- [x] 3단계 — 인코딩 2종 (v1/v2 + 왕복 검증)
- [x] 4단계 — 학습 (동일 조건 v1/v2)
- [x] 5단계 — 비교·측정 (생성 구조 유효성)
- [x] 6단계 — 리서치 노트 → [`FINDINGS.md`](./FINDINGS.md)

**결론:** 논문 주장은 이 장단 레짐에서 비재현 — 위치 인코딩 이득은 레짐 의존적이며,
확실한 이점은 성능이 아니라 **구조 checkability**. 자세히는 `FINDINGS.md`.

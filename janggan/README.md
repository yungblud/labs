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

## 현재 진행

- [x] 1단계 — 환경·디렉터리 뼈대
- [ ] 2단계 — 장단 데이터 제작
- [ ] 3단계 — 인코딩 2종
- [ ] 4단계 — 학습
- [ ] 5단계 — 비교·측정
- [ ] 6단계 — 리서치 노트

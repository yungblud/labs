# labs

논문·연구를 **직접 손으로 한 바퀴 돌려보며 체험**하는 실험 레포.
"결과물"이 아니라 **연구 과정 자체를 재현**하는 게 목적.

## 진행 중 실험

### janggan-encoding-research — 적은 데이터 × 도메인 인코딩
- 출발점: [arXiv 2408.01096](https://arxiv.org/abs/2408.01096) *Six Dragons Fly Again* (정간보 위치 인코딩 + 표준 트랜스포머로 15세기 궁중음악 복원)
- 재현할 것: **"인코딩 설계가 적은 데이터에서 결과를 가른다"** 를 한국 장단(리듬)으로 축소해 관찰
- 스펙: [`specs/janggan-encoding-research.md`](./specs/janggan-encoding-research.md)
- 후보 비교·재사용 레시피: [`specs/music-ai-encoding-toys.md`](./specs/music-ai-encoding-toys.md)

## 구조

```
labs/
├── specs/     # 각 실험의 목표·설계·체크리스트
└── <toy>/     # 실험별 코드 (예정)
```

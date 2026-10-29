# 실제 평가 방법과 실행 코드

현재 성능 비교는 **AIHub 한국어 실제 발화**에 대한 Original / KoNet Full epoch9 / KoNet LoRA rank8·alpha16 epoch5의 정합 점수입니다. 애니메이션 예시에는 현재 학습·평가 점수를 부여하지 않았습니다.

## 바로 재계산 가능한 실제 점수 평가

[heldout_scores.csv](../reproducibility-20261029/datasets/heldout_scores.csv)는 4화자·44기록·96개 공통 B-window × 3모델 = 288행의 실제 점수 데이터입니다. [재계산기](../reproducibility-20261029/datasets/recompute_heldout_scores.py)는 여섯 거리의 strict both-win과 평균 최소 마진을 계산합니다.

저장소 루트에서, Python·NumPy가 이미 있는 환경으로 실행합니다.

```powershell
uv run --offline --no-project --python python python reproducibility-20261029/datasets/recompute_heldout_scores.py
uv run --offline --no-project --python python python reproducibility-20261029/experiments-20261029/run_random_offset_probe.py
uv run --offline --no-project --python python python -m unittest discover -s reproducibility-20261029/experiments-20261029 -p test_random_offset.py -v
```

고정 거리 지표의 18개 모델-거리 셀에서 strict both-win·최소 마진 총 36개 값이 보존된 표와 1e-12 이내 일치했습니다. 랜덤 오프셋 평가는 동일 입력, seed1701 공통 선택, rank 계산·계층 평균, 보존 결과 일치를 검사하는 CPU unittest 4개가 통과했습니다. [수치 대조 근거](../reproducibility-20261029/verification/score-table-parity.json), [랜덤 평가 자료](../reproducibility-20261029/experiments-20261029/README.md), [기존 지표 테스트](../reproducibility-20261029/tests/test_ko_offset_metrics.py).

## 평가량: 영상 창·프레임·점수는 다른 단위

| 구분 | 기존 공개 짝비교 | 별도 추가 실행의 완료 부분 |
|---|---:|---:|
| 화자 / 발화 기록 | 4명 / 44기록 | 4명 / 44기록 |
| 완료된 서로 다른 창 | 96개 | 50개 |
| 창당 프레임·시간 | 16프레임 / 0.64초 | 16프레임 / 0.64초 |
| 선택 시각 프레임 합계 | 1,536프레임 | 800프레임 |
| 선택 시각 구간 시간 합계 | 61.44초 | 32초 |
| 실제 비교 모델 | Original / Full / LoRA | Original / LoRA |
| 점수 개수 | 창당 모델별 중심+12오프셋=13개; 모델별1,248개, 총3,744개 | 창당 두 모델×중심/target=4개; 총200개 |

시간은 **선택된 시각 창 길이의 합**입니다. 전체 원본 파일 재생시간, 고유 오디오 길이 또는 서로 겹치지 않는 전체 발화시간을 의미하지 않습니다. 점수 3,744개를 독립 영상 3,744개라고 세지 않습니다.

별도 추가 실행의 terminal 상태는 `deadline_prefix`이며 192창 계획 중 50창만 paired 완료됐습니다. 이는 제한 시간 내 완료된 탐색적 부분 실행으로, 기존 96창 공개 지표에 합치지 않았습니다. Full 모델은 이 추가 실행에 포함되지 않습니다. 위 표는 완료량 설명이며 새로운 성능 우위 주장이 아닙니다.
## 프로토콜

A 창에서 기준 offset을 고정하고 분리된 B 창에서 ±40/80/120/160/240/320ms를 평가합니다. strict both-win은 중심 점수가 양쪽보다 각각 1e-6을 초과해 높을 때만 성공이며 tie는 실패입니다. 최소 마진은 두 방향의 중심-이동 점수 차 중 작은 값입니다. 창→기록→화자 순으로 평균합니다.

랜덤 오프셋 진단은 저장된 12개 signed target의 균등 평균과 seed1701 공통 단일 선택을 계산합니다. rank는 tie에0.5를 주므로 strict both-win과 다른 지표입니다. 물리적 동기화 정답률·이진 분류 정확도·SOTA 증명으로 해석하지 않습니다.

## 원본 AV를 모델에 다시 넣는 평가

[실제 AV 평가 CLI](../reproducibility-20261029/scripts/eval_korean_offsets.py), [보고서 생성기](../reproducibility-20261029/scripts/report_korean_offsets.py), [모델 어댑터](../baseline/README.md)를 제공합니다. 실행에는 별도 [AIHub 원본 취득](../dataset/README.md), cohort 준비, upstream LatentSync, 모델 가중치가 필요합니다. 점수 CSV는 원본 AV 입력이 아니므로 새 checkpoint의 추론을 대신하지 않습니다. 상세 준비 명령은 [전체 재현 안내](../reproducibility-20261029/README.md)를 따릅니다.

저장된 완료 결과와 아직 완료되지 않은 추가 실행을 구분합니다. 여기의 재현 검증은 공개된 완료 점수와 CPU 집계에 한정됩니다.

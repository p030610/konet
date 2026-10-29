# KoNet — 한국어 음성·입모양 정합 판별 개선

박성재 · 202020814 · 아주대학교 소프트웨어학과

사전 학습 StableSyncNet을 **AIHub 한국어 발화 데이터로 적응시켜 한국어 영상·음성 동기화 판단 정확도를 개선**하는 프로젝트입니다. 현재 KoNet Full/LoRA의 학습·평가에 사용한 데이터는 AIHub 538입니다. 애니메이션 영상은 활용을 보여 주는 선택적 예시이며 현재 학습·평가에 사용하지 않았습니다.

## 코드·데이터·평가 바로 찾기

| 항목 | 실제 제공 내용과 위치 | 범위 |
|---|---|---|
| Baseline 코드 | [Baseline 안내](baseline/README.md), [원본 모델 로더](reproducibility-20261029/src/pose_aware_roi_generator/stable_syncnet_loss.py), [평가 CLI](reproducibility-20261029/scripts/eval_korean_offsets.py) | 원본 모델 호출·평가 코드 제공. upstream LatentSync와 가중치는 별도 준비 |
| 실제 사용 데이터 | [AIHub 학습·평가 데이터 설명](dataset/README.md), [익명화 실제 평가 점수 CSV](reproducibility-20261029/datasets/heldout_scores.csv), [점수 데이터 카드](reproducibility-20261029/datasets/heldout_scores_datacard.md) | 학습 코호트·분할·취득 절차와 실제 점수 제공. AIHub 원본 AV·라벨은 재배포하지 않음 |
| Evaluation 코드 | [평가 방법·실행 안내](evaluation/README.md), [점수 재계산기](reproducibility-20261029/datasets/recompute_heldout_scores.py), [랜덤 오프셋 코드·테스트](reproducibility-20261029/experiments-20261029/README.md) | 실제 저장 점수의 CPU 재계산 가능. 새 AV 추론에는 별도 데이터·모델 필요 |

## 실제 한국어 학습 데이터

AIHub 538 **립리딩(입모양) 음성인식 데이터**의 일부를 사용했습니다. Full과 LoRA 설정이 공유하는 현재 고정 반복 학습 `frozen40000_epoch0` 코호트의 실제 인덱스를 집계한 결과는 **40,000쌍 = 정합 20,000 + 비정합 20,000**, **15명 화자**, **42개 발화 기록 그룹**, **210개 시점별 영상 자산**입니다. 40,000개의 서로 다른 영상이나 전체 AIHub 데이터를 학습했다는 뜻이 아닙니다.

화자 분할은 학습 15명·개발 validation 4명·test 4명입니다. 현재 표의 공통지원 평가 44기록·96창·4화자는 학습 40,000쌍과 별도입니다. 정확한 단위, 코호트 고정 방식과 원본 취득 절차는 [dataset/README.md](dataset/README.md)에 있습니다.

## 결과와 재현

- [프로젝트 사이트](https://p030610.github.io/konet/)
- [전체 재현 안내](reproducibility-20261029/README.md)
- [실제 평가 결과 JSON·CSV·보고서](reproducibility-20261029/results/)
- [코드·점수 재현 보조 ZIP](reproducibility-20261029/konet-reproducibility-20261029.zip)

원본 StableSyncNet, KoNet Full epoch9, KoNet LoRA rank8·alpha16 epoch5를 같은 한국어 held-out B-window에서 비교합니다. strict both-win과 랜덤 오프셋 순위 승률은 다른 지표입니다. 실제 점수의 CPU 재계산 및 검증 테스트를 제공하며, 원본 AIHub 영상·음성·전사 라벨·학습 체크포인트는 포함하지 않습니다. 따라서 저장소만으로 모델 재학습 또는 원본 AV 전체 재추론이 즉시 가능하다고 주장하지 않습니다.

별도 TA2.0 애니메이션 4개와 기존 생성 모델 영상은 **선택적 활용 예시·향후 연구 참고자료**입니다. 현재 KoNet 학습 데이터나 성능 평가 입력이 아니며, 한국어 AIHub 결과를 대신하지 않습니다.

외부 코드·모델·데이터·미디어는 각 원저작자의 라이선스와 이용 조건을 따릅니다.

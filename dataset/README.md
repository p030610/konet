# 실제 사용한 학습·평가 데이터

## 데이터 정체와 사용 범위

현재 KoNet Full/LoRA의 한국어 추가 학습과 한국어 평가에는 **AIHub 538 — 립리딩(입모양) 음성인식 데이터**를 사용했습니다. [공식 데이터 소개·신청](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=538&topMenu=100).

이는 한국어 실사 발화의 영상·음성 데이터입니다. 아래 수량은 이 프로젝트가 실제 준비·선택한 부분집합의 수량이며, AIHub 전체 데이터의 시간·화자 규모를 우리 학습량으로 인용하지 않습니다. HDTF·VoxCeleb2 원본은 이 KoNet 한국어 적응 코호트에 넣지 않았습니다. 애니메이션은 현재 학습·평가에 사용하지 않았습니다.

## 실제 고정 학습 코호트

Full과 LoRA 설정은 동일한 **현재 고정 반복 학습 `frozen40000_epoch0` offline 코호트**를 가리킵니다. catalog가 지정한 3개 generation의 63개 index JSONL을 집계했으며, 대용량 tensor나 원본 영상을 읽어 얻은 추정치가 아닙니다.

| 항목 | 확인된 수량·의미 |
|---|---|
| 학습 예제 | 고유 pair_id 40,000개 |
| 정합 / 비정합 | label1 정합 20,000쌍 / label0 비정합 20,000쌍 |
| 실제 포함 화자 | 15명; 모두 train 화자 집합에 포함 |
| 실제 발화 기록 그룹 | 42개; window_id에서 noise·시간 anchor를 제외한 recording group 기준 |
| 실제 영상 자산 | visual_source_id 210개; 같은 발화의 시점별 영상은 별도 자산 |
| 실제 음성 자산 식별자 | audio_source_id 210개; 학습 인덱스 기준, 독립 발화 210개를 뜻하지 않음 |
| 고정 선택 | canonical epoch0, ordinal 6912 이상 46912 미만의 연속 40,000개 |
| 입력 형식 | 25fps의 16프레임(0.64초), 시각 tensor 48×128×256 / mel 1×80×52 |

40,000은 AV **쌍의 수**입니다. 40,000개 고유 영상·발화·화자를 뜻하지 않으며, 여러 학습 epoch에서 같은 고정 코호트를 다시 사용하는 것도 새 고유 데이터로 세지 않습니다. 정합 쌍과 시간 이동 비정합 쌍의 생성 규칙은 로컬 코호트 sampling 구현 및 저장 index의 label로 확인했습니다.

설정에는 고정 코호트 시작 이전 처리 이력 `historical_examples_consumed=6912`, source checkpoint cursor 6912와 global update54도 보존되어 있습니다. 위 40,000쌍은 **현재 고정 반복 코호트**의 수량이지 Full 모델의 생애 전체 고유 학습 예제 수가 아닙니다. 이전 이력과 현재 코호트를 단순 합쳐 46,912개 고유 예제라고 주장하지 않으며, 이 설정 이력을 물려받았다는 이유만으로 LoRA도 이전 6,912개를 직접 학습했다고 단정하지 않습니다.

## 분할과 평가 단위

`speaker_splits.json`의 프로젝트 화자 분할은 **train 15명 / validation 4명 / test 4명**입니다. 공식 AIHub 전체 split의 크기를 의미하지 않습니다. 실제 40,000쌍의 화자를 별도로 집계했으며 train의 15명과 일치했습니다.

준비 단계의 pair summary에는 train anchor 후보 86,460개, validation pair 2,112개, test pair 2,112개가 기록되어 있습니다. 이 후보/쌍 집계와 현재 공개 offset-calibrated 평가 표의 분모는 다릅니다. 현재 비교는 test 화자의 48기록 중 calibration과 공통 지원 조건을 만족한 **44기록·96개 B-window·4화자**입니다. 원본 전체 276기록 결과와 이 44기록 짝비교를 직접 같은 순위로 섞지 않습니다.

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
## 저장소에 들어 있는 실제 평가 자료

- [익명화 평가 점수 CSV](../reproducibility-20261029/datasets/heldout_scores.csv): 실제 모델 출력 288행 = 96창 × Original/Full/LoRA 3개 모델. 중심 점수와 여섯 거리의 양·음 방향 점수를 포함합니다.
- [점수 데이터 카드](../reproducibility-20261029/datasets/heldout_scores_datacard.md): 식별자, 지원 집합, 지표 정의.
- [선택 메타데이터](../reproducibility-20261029/data-access/heldout-selection-hashed.json): 48기록의 해시 식별자와 선택 창·calibration 결과. 영상이나 원본 전사가 아닙니다.
- [CPU 평가 코드](../evaluation/README.md): 위 실제 점수로 공개 지표를 다시 계산합니다.

**원본 AIHub AV 파일·전사 라벨·전처리 tensor/cache는 이 저장소에 포함하지 않습니다.** 점수 CSV는 실제 평가 결과 데이터이며, 새로운 모델에 입력하는 원본 AV 데이터셋이 아닙니다. 합성 fixture는 코드 검사에만 쓰며 실제 성능 근거로 사용하지 않습니다.

## 원본 취득과 재현 제한

1. AIHub의 [공식 538 페이지](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=538&topMenu=100)에서 본인 계정으로 데이터 이용을 신청하고 승인 절차에 따라 원본을 취득합니다.
2. 해당 데이터의 이용 조건과 [AIHub FAQ](https://www.aihub.or.kr/aihubnews/faq/list.do)를 확인합니다. 원본 및 단순 편집본의 외부 재배포 제한 때문에 GitHub에 원본을 복사하지 않았습니다.
3. 모델 재추론에는 적법하게 확보한 원본, 준비한 cohort/전처리, upstream 코드와 고정 checkpoint가 필요합니다. [전체 실행 안내](../reproducibility-20261029/README.md)의 준비 조건을 따릅니다. 현재 공개 자료만으로 동일 학습 전체를 재구성할 수 있다고 주장하지 않습니다.

수업의 평가용 데이터 요구에 대해 실제 점수와 재계산 코드는 공개하지만 원본 AV 제공까지 충족했다고 주장하지 않습니다. TA2.0의 애니메이션 영상은 선택적 예시·향후 연구 참고자료일 뿐, 이 학습 데이터나 현재 평가 데이터를 대신하지 않습니다.

## 원본 사전 학습과 구분

[LatentSync 논문 §3.3](https://arxiv.org/html/2412.09262v1#S3.SS3)은 원본 SyncNet을 VoxCeleb2에서 학습하고 HDTF에서 평가한 실험을 설명합니다. 이는 upstream 사전 학습/검증의 출처이며, 우리가 수행한 **AIHub 한국어 추가 학습**과 별개입니다. 논문의 생성 모델 학습용 혼합 데이터 설명을 KoNet 학습 데이터로 옮겨 적지 않습니다.

확인일: 2026-10-30. 근거는 Full/LoRA 설정, 화자 split·pair summary, offline catalog/READY 및 63개 index의 집계입니다. 공개 문서에는 개인 컴퓨터 절대경로나 원본 데이터 식별자를 추가하지 않았습니다.

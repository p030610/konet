# KoNet

박성재 · 202020814 · 아주대학교 소프트웨어학과

한국어 발화의 영상·음성 시간 정합성을 평가하는 KoNet 프로젝트입니다. 장기 목표는 애니메이션 캐릭터의 한국어 더빙이며, 현재 공개 결과는 StableSyncNet의 한국어 적응과 오프라인 동기화 평가입니다.

- 프로젝트 사이트: https://p030610.github.io/konet/
- 재현 안내: [reproducibility-20261029/README.md](reproducibility-20261029/README.md)
- Baseline·KoNet 평가 CLI: [eval_korean_offsets.py](reproducibility-20261029/scripts/eval_korean_offsets.py)
- 지표·보고서 생성: [report_korean_offsets.py](reproducibility-20261029/scripts/report_korean_offsets.py)
- 저장된 실제 평가 결과: [results/](reproducibility-20261029/results/)
- 실제 점수 수준 평가 데이터: [datasets/heldout_scores.csv](reproducibility-20261029/datasets/heldout_scores.csv) 및 [데이터 카드](reproducibility-20261029/datasets/heldout_scores_datacard.md)
- 공개 한국어 애니메이션 평가 입력: [TA2.0 실제 영상 4개·설명·라이선스](reproducibility-20261029/public_animation_eval/README.md)
- 평가 기록 선택 정보: [data-access/heldout-selection-hashed.json](reproducibility-20261029/data-access/heldout-selection-hashed.json)
- 다운로드: [재현 보조 패키지 ZIP](reproducibility-20261029/konet-reproducibility-20261029.zip)

## 평가 범위

공식 StableSyncNet 원본, KoNet Full epoch 9, KoNet LoRA rank 8·alpha 16 epoch 5의 저장된 평가 결과를 제공합니다. 주 비교는 같은 held-out B-window에서 평가한 공통지원 44개 기록·4명 화자의 strict both-win입니다. 개발 validation의 Signed80과 held-out 지표는 서로 다른 프로토콜입니다.

CPU 지표 테스트와 합성 fixture 보고서 생성은 재현 안내에 기록된 범위에서 검증했습니다. 원본 AIHub 영상·음성, 전사 라벨, 학습 체크포인트는 포함하지 않습니다. 따라서 이 저장소만으로 AIHub에서 실제 AV 모델의 전체 평가를 즉시 재실행할 수는 없습니다. 실제 저장된 점수 수준 데이터와 CPU 재집계 코드를 제공하며, 별도 공개 입력 확장으로 TA2.0의 한국어 애니메이션 AV 4개를 포함합니다. 이 4개에서는 성능을 측정하지 않았으며 기존 AIHub 리더보드의 입력도 아닙니다. 모델과 적법하게 확보한 데이터 준비 방법, 평가 명령과 제한은 재현 안내를 확인하세요.

사이트의 공개 영상은 기존 생성 모델의 정성 데모이며, KoNet 성능을 측정한 실제 held-out 데이터셋과 구분됩니다. TA2.0 공개 입력, 결과 집계, 선택 정보, 합성 지표 fixture는 원본 AIHub 평가 영상·음성을 대체하지 않습니다.

## 수업 제출

수업 저장소에는 `submission/Team_박성재_202020814.html` 파일 하나만 제출합니다. HTML의 스타일과 동작 코드는 내부에 포함하고, 영상과 재현 자료는 위 GitHub Pages 주소에서 읽습니다.

외부 코드·모델·데이터·미디어의 권리는 각 원저작자에게 있습니다. 이 저장소는 제3자 자료에 새로운 라이선스를 부여하지 않습니다.

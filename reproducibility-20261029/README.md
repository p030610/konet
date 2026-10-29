# KoNet 수업 제출용 재현 보조 패키지 (2026-10-29)

박성재 · 202020814 · 소프트웨어학과

이 패키지는 실제 한국어 offset 평가 코드, 2026-10-24 모델 출력에서 추출한 점수 수준 평가 데이터셋(score-level evaluation dataset), 결과 집계를 제공합니다. **원본 AIHub 평가 영상·음성, 학습 체크포인트, 외부 LatentSync 저장소는 포함하지 않습니다.** 이 ZIP만으로 실제 AV 모델 평가 전체를 즉시 재현할 수 있다고 주장하지 않습니다. 공개 저장소: https://github.com/p030610/konet · 프로젝트 페이지: https://p030610.github.io/konet/ .

## 포함 파일

- `scripts/eval_korean_offsets.py`: 실제 baseline 및 KoNet 평가 CLI. 기존 구현을 변경 없이 복사했습니다.
- `scripts/report_korean_offsets.py`: 저장된 점수에서 지표·CSV·보고서·SVG 생성.
- `src/ko_offset_eval/`: calibration, held-out 계획, 원본/LoRA/full 어댑터, strict both-win 집계.
- 나머지 `src/` 모듈: 위 CLI가 직접/간접 import하는 프로젝트 코드. 학습 전체 패키지가 아닙니다.
- `configs/`: 실제 attention 구조 YAML과 full/LoRA 모델 로딩용 경로 수정 예제. 모델 예제의 상대경로는 실행 위치 기준이며 자신의 파일 위치에 맞춰 변경합니다.
- `datasets/heldout_scores.csv`: 실제 모델이 출력한 익명화 점수 수준 평가 데이터셋. 모델 재추론용 영상·음성이 아니라 공개 성능 표의 지표 재계산 입력입니다.
- `datasets/heldout_scores_datacard.md`, `datasets/recompute_heldout_scores.py`: 점수 데이터 구성 설명과 CPU 재계산기.
- `data-access/heldout-selection-hashed.json`: 전체 비교 후보 48기록의 해시 기반 선택 명세. 유효 calibration 44기록과 구분합니다.
- `results/`: 실제 저장된 한국어 offset 결과 JSON/CSV/보고서/SVG. 합성 결과와 분리되어 있습니다. 이번 포장 작업에서 새로 측정한 값이 아닙니다.
- `fixtures/synthetic/`: 직접 만든 1기록·1창·13개 점수의 **합성 지표 검증 fixture**. 실제 영상·음성 데이터셋이나 성능 근거가 아닙니다.
- `verification/synthetic-report/`: 합성 fixture의 보고서 출력. 여기서 보이는 성공률은 알고리즘 동작 확인용입니다.
- `tests/test_ko_offset_metrics.py`: 기존 CPU 지표 테스트 13개.
- `SOURCE_MANIFEST.json`: 각 복사본의 원본 상대경로와 SHA-256. 기존 코드를 재작성하지 않았습니다.

## 바로 실행 가능한 CPU 검증

검증 환경: Windows, Python 3.10.11. 보고서/지표 검증에는 GPU가 필요하지 않습니다.

새 환경에서는 `uv venv --python 3.10 .venv` 후 `uv pip install --python .venv/Scripts/python.exe -r requirements-report.txt`를 실행합니다. 아래 예제에서 `python`은 해당 환경의 실행 파일로 바꿀 수 있습니다. 기존 설치 환경이 있다면 재설치할 필요가 없습니다.

```powershell
$env:PYTHONPATH = (Resolve-Path src).Path
$env:PYTHONDONTWRITEBYTECODE = '1'
uv run --offline --no-project --python python python scripts/report_korean_offsets.py --help
uv run --offline --no-project --python python python -m pytest tests/test_ko_offset_metrics.py -q -p no:cacheprovider
uv run --offline --no-project --python python python scripts/report_korean_offsets.py --raw fixtures/synthetic/raw.jsonl --recordings fixtures/synthetic/recordings.jsonl --offsets fixtures/synthetic/offsets.jsonl --output local-smoke-output --bootstrap-samples 40 --seed 2701
```

실제 확인 결과: 두 CLI의 `--help` 성공, CPU 기존 테스트 **13 passed**, 위 합성 보고서 생성 성공. 실제 점수 CSV의 독립 CPU 재계산에서도 네 화자/44기록/96창/288행을 확인했으며, 3모델×6거리의 strict both-win·최소 마진 36개 스칼라가 기존 짝비교 표와 1e-12 이내 일치했습니다(`verification/score-table-parity.json`). 새 환경 설치 자체는 검증하지 않았습니다. `verification/VERIFIED.md`에 정확한 실행 환경과 범위를 기록했습니다.

## 실제 점수 수준 데이터 재계산

`datasets/heldout_scores.csv`는 같은 held-out B-window 96개 × 모델 3개 = 288행의 실제 추론 점수입니다. 공통 지원은 44기록·4화자이며, 최초 48기록 중 4기록의 A-fold 경계 calibration 제외와 구분합니다. 모든 식별자는 공개 표 내부에서만 사용하는 별칭입니다. 영상·음성·원문 전사·원본 경로는 없습니다.

```powershell
uv run --offline --no-project --python python python datasets/recompute_heldout_scores.py
```

기존 공개 표의 strict both-win과 평균 최소 마진을 여섯 거리에서 재계산합니다. 이는 score-level evaluation이며 새 모델 추론이 아닙니다. strict both-win은 중심 점수가 양쪽 오프셋보다 epsilon 1e-6을 초과해 높을 때만 1이며 tie는 0입니다. 최소 마진은 두 방향의 중심-오프셋 점수 차 중 작은 값입니다. 창→기록→화자 순으로 평균합니다. 방향별 W가 별도로 출력되면 각 방향을 개별 비교하고 tie에 0.5를 주는 다른 지표이므로 strict both-win과 섞지 않습니다. 실제 동기화 정답률이나 SOTA 우위를 의미하지 않습니다.
## 실제 한국어 평가 프로토콜

- 원본은 LatentSync StableSyncNet, 비교 모델은 KoNet full epoch9 및 LoRA rank8/alpha16 epoch5입니다.
- 25 fps의 16프레임(0.64초), 16 kHz 음성 및 공식 80×52 mel 인터페이스를 사용합니다. 기록된 정책은 alignment 이전 1280×720 정규화입니다.
- 분리된 A 창에서 -640~+640 ms를 40 ms 간격으로 탐색해 기준 offset을 고정하고, B 창에서 그 기준의 ±40/80/120/160/240/320 ms를 평가합니다. A/B 확장 오디오 지원 구간의 겹침을 배제합니다.
- strict both-win은 중심 점수가 양쪽 오프셋 점수보다 각각 epsilon(1e-6)을 초과해 높은 경우에만 성공입니다. tie는 실패입니다. 창→기록→화자 순으로 평균하며 화자는 동등 가중합니다.
- 조건부 점수와 전체 end-to-end 점수의 분모가 다릅니다. 원본 전체 276기록과 KoNet 4명/48기록을 직접 순위 비교하지 않습니다. 원본/KoNet 짝비교는 같은 B-window의 공통 지원 44기록에서만 합니다.
- 개발 validation의 기존 Signed80과 이 offset-calibrated held-out both-win은 다른 프로토콜입니다. 같은 지표로 혼합하지 않습니다.
- `results/`는 저장된 실험 결과의 보존본입니다. 원본 runner가 FrozenScorer에 NORMALIZE_1280X720를 명시하고 KoNet adapter도 같은 정책을 사용함을 소스에서 확인했습니다. 이 확인은 기존 한국어 평가의 시각 전처리 정책에 대한 것입니다. CPU 지표 테스트 통과를 별도의 GPU 수치 동등성 시험이나 성능 우위 증명으로 읽지 마세요.

## 실제 AV 재평가 전제와 명령

1. [AIHub 538 립리딩(입모양) 음성인식 데이터](https://aihub.or.kr/aihubdata/data/view.do?aihubDataSe=data&currMenu=115&dataSetSn=538&topMenu=100)를 본인 계정으로 신청·승인받아 취득합니다. 이 패키지에는 원본 영상, 오디오, 전사 라벨, 전처리 tensor/cache를 넣지 않았습니다.
2. [공식 LatentSync 코드](https://github.com/bytedance/LatentSync)를 패키지 루트의 `vendor/LatentSync`에 별도로 준비하고 해당 라이선스·실행 의존성을 따릅니다. baseline loader는 이 상대 위치를 사용합니다. FFmpeg/ffprobe도 PATH에서 실행 가능해야 합니다. 공개 upstream 최신 버전과 과거 실행 checkout의 동일성을 자동 보장하지 않습니다.
3. baseline 가중치를 `checkpoints/stable_syncnet.pt`에 별도 준비합니다. 실제 실험의 고정 SHA-256은 `77678d13861a02d6e83d0b169962175b6266be773f1f6f85fda2b0182e225118`이며 prepare가 이를 검사합니다. 학습된 KoNet checkpoint도 별도로 필요합니다. 이 패키지는 그 공개 다운로드 주소를 제공하지 않습니다.
4. 본인이 적법하게 확보한 데이터로 `koen_syncnet_eval.contracts.CohortWindow` 스키마의 cohort JSONL을 만듭니다. 영상·음성 경로, 화자/기록/창 식별자, 시작 프레임, fps와 오디오 지원 정보를 정확히 지정합니다. 실제 전체 cohort와 전체 276기록의 점수는 포함하지 않았으므로 **동일 276개 기록을 ZIP만으로 재구성할 수 없습니다.** 추가한 `data-access/heldout-selection-hashed.json`은 비교용 48기록의 저자 생성 선택 명세입니다. AIHub 취득 후 원래 recording_id 문자열의 UTF-8 SHA-256을 계산해 대응시킬 수 있으며, 실제 영상·음성이나 원본 주석을 포함하지 않습니다.
5. 기존 preparation CLI는 `--reuse-old-scores`가 없어도 과거 실행 identity 설정을 검사합니다. `old-runtime/config.json`에는 protocol_version=koen-signed80-v3와 visual_policy=normalize-1280x720, `score-config.json`에는 backend=frozen, 같은 visual_policy 및 실제 checkpoint/native_root 절대경로가 필요합니다. 실제 과거 실행 설정을 사용해야 하며 아래 경로는 사용자가 준비할 위치입니다.

아래는 실제 실행 인터페이스 예시이며, 필수 데이터·체크포인트가 준비된 후에만 실행합니다. 이번 포장 작업에서는 GPU 명령을 실행하지 않았습니다. `eval` CLI에는 Windows `msvcrt` 잠금 코드가 포함되어 다른 OS 이식은 별도 작업입니다.

```powershell
uv run --offline --no-project --python python python scripts/eval_korean_offsets.py prepare --source-cohort private-data/cohort.jsonl --old-runtime private-data/old-runtime --checkpoint checkpoints/stable_syncnet.pt --native-root vendor/LatentSync --output runs/original --run-id course-reproduction
uv run --offline --no-project --python python python scripts/eval_korean_offsets.py score --output runs/original --old-runtime private-data/old-runtime --backend frozen --device cuda --shared-gpu-owner runs/gpu.owner
uv run --offline --no-project --python python python scripts/eval_korean_offsets.py prepare-arm --source-root runs/original --output runs/lora --model-id konet_lora_r8a16_epoch5 --checkpoint checkpoints/konet-lora-best.pt --training-config configs/lora-model.example.json --native-root vendor/LatentSync --shared-gpu-owner runs/gpu.owner
uv run --offline --no-project --python python python scripts/eval_korean_offsets.py score-arm --source-root runs/original --output runs/lora --model-id konet_lora_r8a16_epoch5 --checkpoint checkpoints/konet-lora-best.pt --training-config configs/lora-model.example.json --native-root vendor/LatentSync --shared-gpu-owner runs/gpu.owner
```

baseline raw와 해당 arm raw를 한 JSONL로 합친 뒤 report CLI에 실제 recordings.jsonl/offsets.jsonl과 함께 전달하면 공통 지원 비교를 생성합니다. 서로 다른 프로토콜 또는 중복·충돌 점수를 섞지 마세요. full 모델은 full-model.example.json과 별도 checkpoint 및 model-id konet_full_epoch9를 사용합니다.

## 별도 공개 AV 평가 입력 4개

실제 다운로드 가능한 한국어 애니메이션 AV 입력은 별도 [public_animation_eval/README.md](public_animation_eval/README.md)에 설명되어 있습니다. [전용 ZIP](public_animation_eval/public_animation_eval.zip), [manifest](public_animation_eval/manifest.json), 원본 파일 4개와 라이선스·출처 표기를 제공합니다. Neosapience TA2.0 Animation Dataset의 고정 revision에서 한국어 clip_id 순 첫 네 개를 선택했으며, 총 39.84초입니다. 파일 SHA와 전체 AV decode를 확인했습니다.

**이 네 클립은 아직 모델 성능을 측정하지 않은 별도 입력 세트입니다. AIHub 44기록 표의 원본 데이터가 아니며, 해당 점수를 재현하지 않습니다.** 기존 720p 평가 파이프라인에 적용하려면 정규화·얼굴 처리·창 선택 등의 명시적 적응이 필요합니다. upstream train split에서 뽑았으므로 보편적인 held-out test set으로 주장하지 않습니다. 원음의 별도 라이선스와 전체 출처 표기는 해당 데이터 카드에 보존했습니다.

`konet-reproducibility-20261029.zip`은 코드·점수 데이터용 작은 ZIP이며 `public_animation_eval/` 폴더는 중복 용량을 피하기 위해 제외합니다. AV 입력은 위 전용 ZIP으로 별도 받으세요. 공개 저장소에는 코드 패키지와 AV 데이터 폴더가 함께 배치됩니다.
## 평가 데이터의 범위와 공개 제한

[수업 과제](https://github.com/LSH-Ajou/Speech-Recognition-Challenge-2026)는 repository에 평가 데이터셋 포함을 요구합니다. [AIHub 공식 FAQ](https://www.aihub.or.kr/aihubnews/faq/list.do)는 원본과 단순 편집본의 외부 재배포를 제한합니다. 본 패키지는 실제 추론에서 얻은 **점수 수준 평가 데이터셋**과 그 지표 재계산 코드를 포함합니다. AIHub 44기록 표의 원본 AV 데이터셋은 포함하지 않으므로 새로운 모델을 그 영상·음성으로 재추론하려면 별도 AIHub 취득이 필요합니다. 수업에서 점수 수준 데이터를 평가용 데이터셋으로 인정하는지는 교수자의 판단에 따르며, 원본 AV 제공까지 완료했다고 주장하지 않습니다. 별도의 합성 fixture는 코드 smoke 검증용이며 실제 benchmark 데이터가 아닙니다.

외부 코드·모델의 권리는 각 원저작자와 라이선스에 따릅니다. 이 보조 패키지는 제3자 원본 데이터 또는 upstream 코드에 새 라이선스를 부여하지 않습니다. 프로젝트 소스와 이 보조 패키지의 공개는 제출자의 요청에 따라 진행합니다.






## 추가 완료 평가 코드
[2026-10-29 랜덤 오프셋 재현 코드·실제 선택 데이터·4개 CPU 테스트](experiments-20261029/README.md)를 포함합니다. Original/Full/LoRA 세 방법만 사용하며, 저장된 점수에 대한 평가입니다.

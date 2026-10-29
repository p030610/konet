# 2026-10-29 실제 점수 기반 랜덤 오프셋 평가

완료된 Original / KoNet Full epoch9 / KoNet LoRA rank8 alpha16 epoch5 세 방법의 CPU 점수 재계산 코드입니다. 입력은 공개된 `../datasets/heldout_scores.csv`이며 96개 동일 창, 44기록, 4화자입니다. 원본 영상·음성을 새로 추론하거나 학습하지 않았습니다. 추가 GPU 실행은 이 완료 결과에 포함하지 않습니다.

## 실행

Python 3.10+와 NumPy가 설치된 환경에서 실행합니다. 스크립트 위치 기준으로 입력/출력을 찾으므로 현재 작업 디렉터리가 달라도 실행할 수 있습니다. 결과 파일은 이 폴더에 재생성됩니다.

```powershell
uv run --offline --no-project --python python python run_random_offset_probe.py
uv run --offline --no-project --python python python -m unittest discover -s . -p test_random_offset.py -v
```

새 환경 다운로드 없이 기존 Python3.10.11/NumPy1.26.4 환경에서 실제 실행했습니다. 평가 스크립트는 다른 작업 디렉터리에서 실행해도 완료했고, 의미 있는 작은 unittest 4개가 통과했습니다. 검증 범위는 세 모델의 동일 창/화자 집합, seed1701의 공통 오프셋 선택, rank 계산과 창→기록→화자 계층 평균, 보존된 실제 결과 JSON과의 1e-12 일치입니다.

## 자료

- `run_random_offset_probe.py`: 상대경로 입력을 쓰는 CPU 평가 코드.
- `random_offset_prespecification.md`: 결과 계산 전에 고정한 12개 signed target과 seed/집계 규칙.
- `random_offset_selection.csv`: 실제 공통 96창의 seed1701 오프셋 선택, 익명화 별칭.
- `random_offset_metrics.json`: 세 방법의 exact uniform expectation 및 fixed draw 결과.
- `random_offset_manifest.json`: 입력 SHA256, 96/44/4 지원, target별 선택 수.
- `random_offset_report.md`: 재생성되는 전체 숫자 표.
- `reference/random_offset_metrics.json`: 완료된 기존 결과에서 세 방법만 보존한 재현 검증 기준.
- `test_random_offset.py`: 4개 독립 검증 테스트.
- `FILES_MANIFEST.json`: 공개 파일 SHA256 및 검증 결과.

원천 점수 CSV는 `../datasets/heldout_scores.csv`에 이미 포함되어 있으므로 복제하지 않습니다. AIHub 원본 AV·전사·경로·학습 데이터는 포함하지 않습니다.

## 해석

주결과는 ±40/80/120/160/240/320ms의 12개 저장된 점수를 동일 가중한 정확한 평균입니다. 가까운 6개/먼 6개도 별도로 보고합니다. 부결과는 창마다 seed1701로 하나를 고른 공통 랜덤 선택입니다. rank는 중심 점수가 높으면1, 낮으면0, 1e-6 이내 tie면0.5입니다. strict는 tie를0으로 둡니다. 각 창을 기록 단위로 평균한 뒤 화자를 동등 가중합니다.

이는 저장된 모델 점수의 순서 진단이며 이진 동기화 정답률이 아닙니다. 40~640ms 연속 구간의 균일 샘플도 아닙니다. 미세조정 방법을 고르기 위한 개발셋이 아니므로 held-out 결과에서 우승 모델을 선택하거나 SOTA를 주장하지 않습니다.

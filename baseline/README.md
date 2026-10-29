# Baseline 및 KoNet 구현 위치

빈 코드 디렉터리를 제공하는 대신 실제 구현으로 연결합니다. 원본은 LatentSync의 pretrained StableSyncNet이며, 이 프로젝트의 목표는 AIHub 한국어 적응을 통한 음성·입모양 정합 판별 개선입니다.

| 구현 | 실제 파일 |
|---|---|
| 원본 StableSyncNet 로딩 wrapper | [stable_syncnet_loss.py](../reproducibility-20261029/src/pose_aware_roi_generator/stable_syncnet_loss.py) |
| 원본 encoder 평가 연결 | [syncnet_utility_runtime_v1.py](../reproducibility-20261029/src/lower_face_generator_v2/syncnet_utility_runtime_v1.py) |
| 원본 scoring 경로 | [koen_syncnet_eval/scoring.py](../reproducibility-20261029/src/koen_syncnet_eval/scoring.py) |
| Full/LoRA checkpoint 어댑터 | [ko_offset_eval/adapters.py](../reproducibility-20261029/src/ko_offset_eval/adapters.py) |
| LoRA 적용 구현 | [konet_training/lora.py](../reproducibility-20261029/src/konet_training/lora.py) |
| 모델 구조 설정 | [syncnet_16_pixel_attn.yaml](../reproducibility-20261029/configs/syncnet_16_pixel_attn.yaml) |
| 평가 진입점 | [eval_korean_offsets.py](../reproducibility-20261029/scripts/eval_korean_offsets.py) |

Full은 원본 모델의 전체 가중치를 한국어 데이터로 추가 학습한 모델이며, LoRA는 양쪽 encoder attention projection의 rank8·alpha16 어댑터를 학습한 모델입니다. 두 한국어 적응 실행은 같은 [AIHub 고정 40,000쌍](../dataset/README.md)을 사용했습니다.

upstream 전체 모델 정의와 실행 의존성은 [공식 LatentSync](https://github.com/bytedance/LatentSync)를 별도로 준비해야 합니다. 원본 가중치와 학습된 KoNet checkpoint는 이 저장소에 포함하지 않았습니다. 모델 로딩 경로·checkpoint SHA256·환경 조건은 [재현 안내](../reproducibility-20261029/README.md)에 있습니다. 공개된 일부 training helper를 전체 학습 파이프라인이나 재학습 완료 패키지로 주장하지 않습니다.

원본 SyncNet의 VoxCeleb2 학습/HDTF 검증은 [논문 §3.3](https://arxiv.org/html/2412.09262v1#S3.SS3)의 upstream 실험입니다. 우리 KoNet 적응에 그 원본 데이터나 애니메이션을 사용했다는 뜻이 아닙니다.

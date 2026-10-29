---
pretty_name: TA2.0 Animation Dataset
license: apache-2.0
language:
  - en
  - ko
  - de
  - fr
  - nl
  - it
  - es
  - pt
  - pl
size_categories:
  - 10K<n<100K
tags:
  - video
  - talking-avatar
  - animation
  - audio-driven-animation
  - avatar-animation
  - video-generation
  - anime
  - synthetic
  - lip-sync
---

# TA2.0 Animation Dataset for Talking Avatars

**11,255 AI-generated anime-style talking avatar video clips** (≈30.3 hours) with per-clip quality metadata.

Each clip shows a single animated character speaking, rendered at **1280×1280 @ 25 fps** (H.264 video + AAC audio, ~10 s per clip). Clips were generated with [**LongCat-Video-Avatar**](https://github.com/meituan-longcat/LongCat-Video) (Meituan LongCat Team), an audio-driven avatar video generation model, conditioned on reference character images and driven by speech audio in 9 languages — then face-tracked, cropped, quality-scored, and human-reviewed.

[TalkingAvatar Model](https://huggingface.co/neosapience/TA2.0) |
[Inference Code](https://github.com/neosapience/TA2.0)

## Dataset contents

```
metadata.parquet       # canonical per-clip metadata
metadata_export.csv    # CSV export of the same metadata
videos/000/clip_00000.mp4
...
videos/011/clip_11254.mp4
```

- 11,255 clips, 2,723,833 frames total (≈30.26 h)
- 6,487 unique reference characters (`ref_id`)
- Single `train` split

## Metadata fields

| column | type | description |
|---|---|---|
| `file_name` | str | relative path of the video file |
| `clip_id` | str | unique clip id (`clip_00000` …) |
| `ref_id` | str | reference character id — clips sharing a `ref_id` were generated from the same reference character image (useful for identity-disjoint splits) |
| `motion_mode` | str | generation motion setting: `active` (5,641) or `static` (5,614) |
| `language` | str | language of the driving speech: en 4,974 / ko 1,510 / de 1,439 / fr 1,139 / nl 762 / it 425 / es 423 / pt 374 / pl 209 |
| `frame_num` | int | number of frames (90–249) |
| `duration_sec` | float | `frame_num / 25` |
| `vqa_score` | float | DOVER video quality score over the full frame (0–1, fused scale) |
| `c50` | float | speech clarity (C50, dB) estimated from the audio track |
| `snr` | float | speech signal-to-noise ratio (dB) estimated from the audio track |
| `pad_top` / `pad_bottom` / `pad_left` / `pad_right` | float | black padding margins (px) added when the source frame was fitted onto the square 1280×1280 canvas. In this release `pad_top`/`pad_bottom` are always 0; left/right padding exists in ~96% of clips (median ≈253/265 px). Crop columns `[pad_left : 1280 - pad_right]` to remove the bars. |

`vqa_score`, `c50` and `snr` are model-estimated signals intended for filtering/weighting, not ground-truth labels.

## Generation & curation

Videos were generated with [LongCat-Video-Avatar](https://huggingface.co/meituan-longcat/LongCat-Video-Avatar) (model weights released under the MIT License). After generation, clips went through automated and manual quality control:

- automated filtering of motion artifacts (optical-flow anomalies) and on-screen text/subtitles (OCR),
- human review of every remaining clip; clips whose character deviated from the reference identity were removed,
- one clip per source video: when multiple faces were detected, only the face best matching the audio (highest AV-sync score) was kept.

## Audio sources

The driving speech audio derives from nine openly licensed public speech corpora (CC BY 4.0 / CC BY 3.0 / CC0 1.0 — all permissive, commercial use allowed): Multilingual LibriSpeech, The People's Speech, VoxPopuli, LibriSpeech, Zeroth-Korean, FLEURS, YODAS, AMI Meeting Corpus, and MInDS-14. The audio was segmented and re-encoded when muxed into the released clips.

Per-corpus credits, clip counts, license details, and removal requests: **[ATTRIBUTION.md](ATTRIBUTION.md)**.

## Usage

```python
from datasets import load_dataset  # datasets >= 3.0 for video support

ds = load_dataset("neosapience/TA2.0_animation_dataset")
```

Or download the files directly and use `metadata.parquet` or
`metadata_export.csv` to index clips by `file_name`.

## Limitations

- All video content is synthetic (AI-generated animation); character appearance may drift slightly within a clip.
- Quality scores are automatic estimates on animated content and should be treated as relative signals.
- No per-clip lip-sync score is provided: automated AV-sync estimates rely on face detection, which is not accurate enough on stylized anime faces to yield a trustworthy per-clip value.
- Left/right black padding is part of the pixel content; use the `pad_*` columns to crop it out if undesired.

## License

This dataset is released under the **Apache License 2.0**.

The video content is synthetic, generated with [LongCat-Video-Avatar](https://github.com/meituan-longcat/LongCat-Video); the model weights are distributed under the MIT License, which places no restrictions on generated outputs.

The embedded speech audio **remains subject to its original licenses** (CC BY 4.0 / CC BY 3.0 / CC0 1.0 — all permissive); per-corpus credits and details are in [ATTRIBUTION.md](ATTRIBUTION.md).

## TA2.0 — model weights & inference code

**TA2.0** is Neosapience's audio-driven talking-avatar video generation model — given a reference character image, speech audio, and a text prompt, it generates a full-body talking video (480×832 @ 25 fps). **TA2.0 was trained in part on this dataset** (the clips themselves were generated with LongCat-Video-Avatar — see *Generation & curation*).

- **Model weights**: [huggingface.co/neosapience/TA2.0](https://huggingface.co/neosapience/TA2.0) — inference weights (safetensors, ~14.5 GB) with config, model card, and integrity checksums
- **Inference code**: [github.com/neosapience/TA2.0](https://github.com/neosapience/TA2.0) — self-contained PyTorch CLI, Dockerfile, and usage docs (Apache License 2.0)

## Acknowledgments

- This dataset was produced as a deliverable of the **첨단 GPU 활용 지원 사업** (project period 2026-04-01 – 2026-07-15), supported by **정보통신산업진흥원** and **한국정보통신진흥협회**.
- Video generation model: [LongCat-Video-Avatar](https://github.com/meituan-longcat/LongCat-Video) by the Meituan LongCat Team.

## Changelog

- **2026-08-04** — added the TA2.0 model weights & inference code section. `metadata.csv` was renamed to `metadata_export.csv` (identical content) to fix `load_dataset` (VideoFolder) metadata discovery.
- **2026-07-28** — removed the `sync_score` metadata column: it depended on face-detection boxes that are not accurate enough on stylized anime faces, so the per-clip values were not a trustworthy signal. All other metadata and all videos are unchanged.
- **2026-07-23** — initial public release (11,255 clips, metadata, audio attribution).

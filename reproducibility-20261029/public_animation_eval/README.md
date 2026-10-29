# Public Korean animation mini-evaluation inputs

This package contains **four actual downloadable audiovisual input clips**, copied unchanged from Neosapience's [TA2.0 Animation Dataset](https://huggingface.co/datasets/neosapience/TA2.0_animation_dataset). It is a small, separate public input set for future animation-domain evaluation. **No KoNet or baseline performance has been measured on this subset. It is not the source of the AIHub 44-record leaderboard and does not reproduce those results.**

## Source and reproducible selection

- Provider: Neosapience; source videos generated with LongCat-Video-Avatar by the Meituan LongCat Team.
- Pinned dataset revision: `d1e554cede78d0830349e1fbfb19f7e2a011460b`.
- Source split: `train`. This is a selected evaluation-input package, not an upstream held-out test split. TA2.0 was trained in part on this dataset; do not claim it is unseen data for that model.
- Selection: filter the pinned `metadata_export.csv` to `language=ko`, sort by `clip_id`, and take the first four records. Quality scores, model outputs, and visual preferences were not used. All four selected files were already locally available; no fallback selection was necessary.
- Original filenames are preserved. Files are byte-identical to the pinned Hugging Face LFS SHA-256 values and the local originals. No cropping, normalization, re-encoding, audio shifts, or other media modifications were applied.

| Clip | Character ID | Duration | Size (bytes) |
|---|---|---:|---:|
| [clip_00001.mp4](videos/clip_00001.mp4) | ref_07118 | 9.96 s | 3,194,617 |
| [clip_00003.mp4](videos/clip_00003.mp4) | ref_04227 | 9.96 s | 5,623,635 |
| [clip_00017.mp4](videos/clip_00017.mp4) | ref_04438 | 9.96 s | 2,825,093 |
| [clip_00025.mp4](videos/clip_00025.mp4) | ref_04412 | 9.96 s | 9,091,997 |

Total: 39.84 seconds, 20,735,342 bytes of MP4 media. Every clip is 1280 × 1280, 25 fps, 249 frames, H.264 video and AAC mono 16 kHz audio. The character IDs identify reference images, not verified human speaker identities. Language is the upstream metadata label, not independently transcribed or manually verified here.

## Downloads and metadata

- [Complete input package ZIP](public_animation_eval.zip)
- [Machine-readable manifest JSON](manifest.json) / [CSV](manifest.csv)
- [Selected original metadata rows](upstream_selected_metadata.csv)
- [Verification results](verification/checks.json)
- [Preserved upstream dataset card](UPSTREAM_README.md), [audio attribution](ATTRIBUTION.md), [Apache 2.0 license text](LICENSE)

The manifest records source download URLs pinned to the revision, SHA-256, dimensions, frame rate, duration, codecs, audio sample rate, and the evaluation status. The selected upstream metadata includes estimated quality signals for provenance; these are not ground-truth synchronization labels.

## Label concept and protocol limits

`nominal_added_audio_offset_ms=0` only means the released audio/video tracks are unchanged. `human_verified_sync_offset_ms=null` means the absolute synchronization offset is unknown. A future relative-offset experiment can use the unmodified pair as a nominal reference and record deliberately added shifts, but must not describe those pairs as human-verified synchronization ground truth. The upstream card explicitly warns that anime face detection does not yield trustworthy per-clip lip-sync scores.

These native square inputs differ from the existing AIHub evaluation's 1280 × 720 preprocessing and recording/window assumptions. That protocol is **not directly applicable without a declared and validated adaptation** (face handling, resize/crop/padding, audio alignment, window selection, and failure policy). The bundled `eval_korean_offsets.py prepare` path also requires a 276-recording source cohort, so this four-file manifest is not advertised as a drop-in CLI adapter. No new model code, CUDA run, training, or model inference was performed for this package. No baseline-versus-KoNet comparison, metric value, or generalization claim is supported by it yet.

Four clips are too small for a representative benchmark. All video is AI-generated animation and may contain source synchronization or identity artifacts. Original padding is retained. Models trained on this source may have data overlap; document training-data provenance before interpreting future results.

## Rights and attribution

The [pinned provider card](https://huggingface.co/datasets/neosapience/TA2.0_animation_dataset/blob/d1e554cede78d0830349e1fbfb19f7e2a011460b/README.md) explicitly releases the dataset under **Apache License 2.0**, provided in [LICENSE](LICENSE). This does not replace the original embedded-speech licenses.

The provider states that speech remains subject to its original licenses. [ATTRIBUTION.md](ATTRIBUTION.md) preserves the full upstream credits and source links: Multilingual LibriSpeech, LibriSpeech, Zeroth-Korean, FLEURS, AMI, and MInDS-14 (CC BY 4.0); YODAS (CC BY 3.0); VoxPopuli (CC0 1.0); and The People's Speech (CC BY 2.0–4.0, item-dependent). The upstream metadata does not map each selected clip to a source corpus, so we retain all upstream corpus credits rather than inventing per-clip audio provenance. Relevant license links are included in the upstream attribution, with [CC BY 2.0](https://creativecommons.org/licenses/by/2.0/) supplied here for completeness.

Original audio modifications by the provider: segmentation, AAC re-encoding, and muxing with synthetic video. This subset package makes no further media changes. Preserve the upstream card, attribution, and license when redistributing. This package contains no AIHub media.

## Verified media checks

On 2026-10-29, all four copied files matched both their local-original SHA-256 and the pinned upstream LFS SHA-256. `ffprobe` found the expected audio and video streams; FFmpeg fully decoded both streams with exit code 0 for every file. Every media file is below 100 MB. These are integrity and decode checks, not model-performance tests.

To repeat a CPU-only decode check after extracting this ZIP, run in this directory with FFmpeg installed:

```powershell
Get-ChildItem -LiteralPath videos -Filter *.mp4 | ForEach-Object {
    ffmpeg -v error -i $_.FullName -map 0:v:0 -map 0:a:0 -f null -
    if ($LASTEXITCODE -ne 0) { throw "Decode failed: $($_.Name)" }
}
```

The four original files remain preserved in their original local source location. Only the new subset directory was created for this packaging task.

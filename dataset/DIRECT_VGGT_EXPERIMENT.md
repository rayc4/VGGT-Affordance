# Direct-anchor VGGT experiment

This is the simplified replacement experiment for preprocessing steps 1--6.
It keeps the original scripts and all existing outputs intact.

The new path retrieves a small, diverse set of full frames from the complete
action description, runs one structured Qwen grounding call per selected
frame, and prompts SAM directly in full-frame coordinates. Those sparse masks
then enter VGGT tracking, 3D anchor-consensus selection, and a semantic-seed
crop. The final writer still publishes the exact six-file `processed_sam2`
leaf consumed by the unchanged training dataset.

The recommended wrapper auto-detects all visible GPUs, starts one worker per
GPU, and runs validation before train. It is safe to re-run after interruption:
cache items and direct anchors resume directly, while VGGT recomputes its
prediction-only evidence before validating and skipping complete legacy leaves.

```bash
bash dataset/run_direct_vggt_experiment.sh
```

The wrapper uses these isolated roots by default:

```text
scenefun3d/preprocessing_experiments/direct_vggt_v1/anchors/{val,train}
scenefun3d/preprocessing_experiments/direct_vggt_v1/clip_cache/{val,train}
scenefun3d/processed_sam2_vggt_direct_v1/{val,train}
```

It never reads the legacy step-3 point tree: `NO_POINT_ROOT=1` is translated
to the preprocessor's explicit `--no-point-root`. This prevents old ranks or
points from silently influencing the direct-anchor experiment.

The VGGT split guard fingerprints the completed direct-anchor configuration
and artifact inventory. A retryable Qwen/SAM error stops its shard before VGGT
starts, while a deliberate later regeneration is rejected from the existing
processed root instead of mixing old and new samples. Local-only runs also set
the Hugging Face/Transformers offline environment, avoiding background HEAD
requests for optional processor files.

The fully expanded experiment is below. Every stage is joined with `&&`, so
train preprocessing cannot start until validation reaches the original
usable-description coverage (`323/445`, gated at `0.7258`). Downstream
training cannot start until train reaches its original coverage (`1506/2598`,
gated at `0.57967`) and both audits confirm nonempty legacy-format masks.

```bash
TASA_ALL_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)" && \
test -n "$TASA_ALL_GPUS" && \
TASA_DIRECT_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v1/anchors && \
TASA_CLIP_CACHE=scenefun3d/preprocessing_experiments/direct_vggt_v1/clip_cache && \
TASA_PROCESSED_ROOT=scenefun3d/processed_sam2_vggt_direct_v1 && \
ENV_NAME=tasa SPLIT=val GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d OUTPUT_ROOT="$TASA_DIRECT_ROOT" CACHE_ROOT="$TASA_CLIP_CACHE" EXTRA_ARGS="--local-files-only" bash dataset/run_direct_anchors_parallel.sh && \
ENV_NAME=tasa SPLIT=val GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d ANCHOR_ROOT="$TASA_DIRECT_ROOT/val" NO_POINT_ROOT=1 SAVE_DIR="$TASA_PROCESSED_ROOT" EXTRA_ARGS="--local-files-only --sam-local-files-only --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --max-published-views 6 --min-published-view-component-f1 0.10" bash dataset/run_preprocess_vggt_legacy_parallel.sh && \
conda run --no-capture-output -n tasa python -m pipeline.step4_vggt_legacy.audit --data-root scenefun3d --root "$TASA_PROCESSED_ROOT" --split val --min-description-coverage 0.7258 --require-hardlink-dedup && \
ENV_NAME=tasa SPLIT=train GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d OUTPUT_ROOT="$TASA_DIRECT_ROOT" CACHE_ROOT="$TASA_CLIP_CACHE" EXTRA_ARGS="--local-files-only" bash dataset/run_direct_anchors_parallel.sh && \
ENV_NAME=tasa SPLIT=train GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d ANCHOR_ROOT="$TASA_DIRECT_ROOT/train" NO_POINT_ROOT=1 SAVE_DIR="$TASA_PROCESSED_ROOT" EXTRA_ARGS="--local-files-only --sam-local-files-only --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --max-published-views 6 --min-published-view-component-f1 0.10" bash dataset/run_preprocess_vggt_legacy_parallel.sh && \
conda run --no-capture-output -n tasa python -m pipeline.step4_vggt_legacy.audit --data-root scenefun3d --root "$TASA_PROCESSED_ROOT" --split train --min-description-coverage 0.57967 --require-hardlink-dedup && \
GPUS="$TASA_ALL_GPUS" GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 bash scripts/train.sh direct_vggt_legacy_v1 "$TASA_PROCESSED_ROOT"
```

Use a different `EXPERIMENT_ROOT`, `DIRECT_ANCHOR_ROOT`, and `PROCESSED_ROOT`
for an ablation. Configuration guards reject attempts to mix result-affecting
settings in an existing root. Use `--preprocess-only` on the wrapper to stop
after the train audit, or `--dry-run` to print its resolved commands.

Each accepted sample directory contains exactly:

```text
filtered_point_cloud.ply
gt_mask_global.npy
gt_mask_local.npy
mask_result.json
pred_mask_global.npy
pred_mask_local.npy
```

The audit fails on malformed leaves, empty local GT, empty local prediction,
loader incompatibility, missing hardlink deduplication, or inadequate
description coverage. Filtering protects training from invalid samples;
coverage gates ensure that filtering alone cannot be mistaken for improved
localization.

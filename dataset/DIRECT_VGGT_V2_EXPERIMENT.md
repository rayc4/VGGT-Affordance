# Direct-anchor VGGT v2 experiment

This version fixes Qwen2.5-VL coordinate interpretation while preserving the
original six-file `processed_sam2` leaf contract. It uses fresh v2 anchor and
processed roots, leaves v1 results intact, and reuses the compatible v1 CLIP
image-feature cache.

For every description, v2 evaluates the top four diverse CLIP frames. If all
four abstain or return invalid grounding, it queries ranks 5–8 sequentially
and stops after the first visible result. Qwen's exact factor-28 smart-resized
image dimensions are supplied in the prompt and used to normalize returned
pixels. An invalid optional box is discarded while a valid model point still
prompts SAM.

The direct-anchor audit runs before VGGT and requires 78% validation coverage
and 65% train coverage. VGGT then performs the existing anchor-consensus and
semantic crop path. Its audit requires the original usable-description
coverage and validates nonempty, loader-compatible legacy leaves.

The short all-GPU entry point is:

```bash
bash dataset/run_direct_vggt_v2_experiment.sh
```

The fully expanded preprocessing, audits, and unchanged training sequence is:

```bash
TASA_ALL_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)" && \
test -n "$TASA_ALL_GPUS" && \
TASA_DIRECT_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v2/anchors && \
TASA_CLIP_CACHE=scenefun3d/preprocessing_experiments/direct_vggt_v1/clip_cache && \
TASA_PROCESSED_ROOT=scenefun3d/processed_sam2_vggt_direct_v2 && \
TASA_DIRECT_ARGS="--local-files-only --grounding-coordinate-mode qwen-resized-pixels --max-candidates 8 --max-per-video 8 --primary-candidates 4 --rescue-on-primary-abstention --rescue-success-target 1" && \
TASA_VGGT_ARGS="--local-files-only --sam-local-files-only --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --max-published-views 6 --min-published-view-component-f1 0.10" && \
ENV_NAME=tasa SPLIT=val GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d OUTPUT_ROOT="$TASA_DIRECT_ROOT" CACHE_ROOT="$TASA_CLIP_CACHE" EXTRA_ARGS="$TASA_DIRECT_ARGS" bash dataset/run_direct_anchors_parallel.sh && \
conda run --no-capture-output -n tasa python -m pipeline.step1_direct_anchors.audit --data-root scenefun3d --root "$TASA_DIRECT_ROOT" --split val --min-description-coverage 0.78 && \
ENV_NAME=tasa SPLIT=val GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d ANCHOR_ROOT="$TASA_DIRECT_ROOT/val" NO_POINT_ROOT=1 SAVE_DIR="$TASA_PROCESSED_ROOT" EXTRA_ARGS="$TASA_VGGT_ARGS" bash dataset/run_preprocess_vggt_legacy_parallel.sh && \
conda run --no-capture-output -n tasa python -m pipeline.step4_vggt_legacy.audit --data-root scenefun3d --root "$TASA_PROCESSED_ROOT" --split val --min-description-coverage 0.7258 --require-hardlink-dedup && \
ENV_NAME=tasa SPLIT=train GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d OUTPUT_ROOT="$TASA_DIRECT_ROOT" CACHE_ROOT="$TASA_CLIP_CACHE" EXTRA_ARGS="$TASA_DIRECT_ARGS" bash dataset/run_direct_anchors_parallel.sh && \
conda run --no-capture-output -n tasa python -m pipeline.step1_direct_anchors.audit --data-root scenefun3d --root "$TASA_DIRECT_ROOT" --split train --min-description-coverage 0.65 && \
ENV_NAME=tasa SPLIT=train GPUS="$TASA_ALL_GPUS" DATA_ROOT=scenefun3d ANCHOR_ROOT="$TASA_DIRECT_ROOT/train" NO_POINT_ROOT=1 SAVE_DIR="$TASA_PROCESSED_ROOT" EXTRA_ARGS="$TASA_VGGT_ARGS" bash dataset/run_preprocess_vggt_legacy_parallel.sh && \
conda run --no-capture-output -n tasa python -m pipeline.step4_vggt_legacy.audit --data-root scenefun3d --root "$TASA_PROCESSED_ROOT" --split train --min-description-coverage 0.57967 --require-hardlink-dedup && \
GPUS="$TASA_ALL_GPUS" GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 bash scripts/train.sh direct_vggt_legacy_v2 "$TASA_PROCESSED_ROOT"
```

Because every stage is joined by `&&`, insufficient direct coverage stops the
run before VGGT, and inadequate final coverage stops it before training.

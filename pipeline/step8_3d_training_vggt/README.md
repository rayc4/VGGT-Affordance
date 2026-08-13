# VGGT-conditioned 3D Affordance Refinement (parallel path)

Alternative to `pipeline/step8_3d_training/` that conditions the mask
refinement network on **per-point VGGT features** in addition to the initial
lifted mask. No existing file is modified; this path adds:

| File | Role |
|---|---|
| `pipeline/step7b_vggt_feats/extract_vggt_features.py` | Offline extraction: samples VGGT's dense point-head latent while lifting pixels onto cropped laser-scan points and writes feature/reliability arrays per sample dir |
| `dataset/AffordanceDatasetVGGT.py` | `AffordanceDataset` subclass that adds `c_vggt_feat`, `c_vggt_conf`, and `c_vggt_view_count` |
| `models/cdm_vggt.py` | VGGT fusion models, including the MLP control and the scalar-gated additive stem design |
| `pipeline/step8_3d_training_vggt/{train,eval}_vggt.py` + `configs/` | Hydra entries mirroring `train_no_diff.py` / `eval_no_diff.py` |

## 1. Extract features (once per split)

Run in the `tasa` env (torch 2.4.1+cu124). Downloads `facebook/VGGT-1B` from
the HF hub on first use (or pass `--ckpt /path/to/model.pt`).

```bash
# training samples (per-frame processed_sam2 layout)
python pipeline/step7b_vggt_feats/extract_vggt_features.py \
    --data_root scenefun3d --split train \
    --processed_dir scenefun3d/processed_sam2/train \
    --require_nonempty_gt

# eval samples (processed_sam2 layout)
python pipeline/step7b_vggt_feats/extract_vggt_features.py \
    --data_root scenefun3d --split val \
    --processed_dir scenefun3d/processed_sam2/val
```

Source-frame extraction is the default. These commands leave the source
directories unchanged and write three aligned arrays plus a manifest in each
directory of the mirrored feature trees:

```text
scenefun3d/processed_sam2/vggt_features/<split>/<visit>/<video>/<sample>/vggt_dpt_source_feat.npy
scenefun3d/processed_sam2/vggt_features/<split>/<visit>/<video>/<sample>/vggt_source_conf.npy
scenefun3d/processed_sam2/vggt_features/<split>/<visit>/<video>/<sample>/vggt_source_view_count.npy
scenefun3d/processed_sam2/vggt_features/<split>/<visit>/<video>/<sample>/vggt_source_meta.json
```

For multi-GPU extraction, use the shard runner. It starts one persistent VGGT
worker per GPU and divides visits among them, avoiding a model reload for every
visit:

```bash
GPUS="0,1,2,3" \
EXTRA_ARGS="--require_nonempty_gt" \
  pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh train

GPUS="0,1,2,3" \
  pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh val
```

Use `--dry-run` to inspect the resolved commands. Paths and extraction options
can be overridden through `DATA_ROOT`, `PROCESSED_DIR`, `CACHE_DIR`,
`FEAT_OUT_ROOT`, and `EXTRA_ARGS`.

Notes:
- The lifted mask's exact source RGB frame is always VGGT frame zero. Up to
  `--chunk_size - 1` temporally nearby frames provide attention context, but
  only frame zero's descriptor is sampled and saved. `--context_stride`
  controls spacing between context frames.
- Samples sharing the same source frame reuse one VGGT forward. The source 3D
  points are projected back into that frame with SceneFun3D pose, intrinsics,
  and depth, then the dense latent is bilinearly sampled at the resulting
  continuous pixel coordinate.
- The default `--feature_source dpt` samples the pretrained 128-D dense tensor
  immediately before the VGGT point head's final XYZ/confidence predictor; no
  random projection is applied.
- Source-mode view count is binary: one when the point is visible in the source
  frame and zero otherwise. Confidence is sampled from that same source pixel.
- Each manifest records source/context filenames and every feature-affecting option.
  A mismatched manifest forces recomputation, preventing old visit-averaged
  arrays or different context settings from being silently reused.
- Use `--frame_mode visit_average` for the legacy behavior. It computes one
  feature field per visit, uses `--frames_per_video`, averages visible-view
  descriptors, and writes the old `vggt_dpt_feat.npy`, `vggt_conf.npy`, and
  `vggt_view_count.npy` names. `--view_aggregation uniform` selects the legacy
  uniform average and `vggt_dpt_feat_uniform.npy`.
- `--feature_source patch` retains the legacy patch-token baseline and writes
  `vggt_source_feat.npy` in source mode. `--out_dim 256` applies the fixed
  seeded projection of 2048-D tokens, while `--out_dim 0` keeps raw tokens.
- The default `--confidence_head point` samples VGGT `world_points_conf` at
  each projected scan point. Use `--confidence_head depth` to use
  `depth_conf` instead.
- Projection uses the SceneFun3D hires poses/intrinsics/depth (same convention
  as step7), **not** VGGT's own predicted cameras. Points unseen in their
  sample's source frame get zero features, confidence, and view count.
- Tune `--chunk_size` for GPU memory. In legacy visit-average mode, also tune
  `--frames_per_video`.

## 2. Train

The launcher keeps a global batch size of 64 by default and divides it evenly
across the selected GPUs. It also assigns every rank the same Hydra output
directory:

```bash
GPUS="0,1,2,3" \
  pipeline/step8_3d_training_vggt/run_train_vggt_parallel.sh \
    vggt-gated scenefun3d/processed_sam2 \
    task.dataset.vggt_feat_name=vggt_dpt_source_feat.npy \
    task.dataset.vggt_conf_name=vggt_source_conf.npy \
    task.dataset.vggt_view_count_name=vggt_source_view_count.npy

# Inspect the resolved torchrun command without starting training.
GPUS="0,1,2,3" \
  pipeline/step8_3d_training_vggt/run_train_vggt_parallel.sh --dry-run \
    vggt-gated scenefun3d/processed_sam2 \
    task.dataset.vggt_feat_name=vggt_dpt_source_feat.npy \
    task.dataset.vggt_conf_name=vggt_source_conf.npy \
    task.dataset.vggt_view_count_name=vggt_source_view_count.npy
```

Set `GLOBAL_BATCH_SIZE`, `GLOBAL_VAL_BATCH_SIZE`, `EXP_NAME`,
`PROCESSED_SAM2_DIR`, or `EXP_DIR` in the environment as needed. Additional
arguments are passed through as Hydra overrides, for example
`task.train.max_steps=25000`.

Each sample is one frame crop. VGGT features are extracted offline with the
sample's source frame plus context; PointTransformer then supervises and
evaluates every sample independently. The path uses the same
`SimpleMaskRefinementTrainLoop` (losses, checkpoint naming) as the original
model. Validation runs once per epoch and writes a numbered checkpoint each
time, together with evaluator-compatible per-frame outputs under
`eval/mask_refinement_model<step>/`. Training stops after eight
non-improving epochs following a five-epoch warmup. The best model is
`mask_refinement_model_best.pt` under the run's `ckpt/` directory.

### Recommended source-pixel dense point-head adapter

After extracting both train and validation features, train the 128-D dense DPT
variant from the mask-only base checkpoint:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_dpt_source_adapter_from_base.sh \
  --dry-run

GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_dpt_source_adapter_from_base.sh
```

This selects `model=cdm_vggt_adapter_dpt_frozen`, loads
`vggt_dpt_source_feat.npy` together with source-pixel confidence and visibility,
and trains only the zero-initialized residual adapter. The exact lifted-mask
source frame is VGGT frame zero; the point-head latent is sampled only from
that frame at projected scan pixels. VGGT's predicted XYZ is not used to align
the laser scan.

The corresponding source-pixel additive-stem experiment is:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_dpt_source_additive_stem_from_base.sh
```

The older `run_vggt_dpt_adapter_from_base.sh` and
`run_vggt_dpt_uniform_*` launchers remain available for visit-averaged
ablation bundles.

For standalone source-feature evaluation, pass the matching bundle names:

```bash
python pipeline/step8_3d_training_vggt/eval_vggt.py \
  exp_dir=outputs/<source-feature-run> \
  model=cdm_vggt_adapter_dpt_frozen \
  task.dataset.vggt_feat_name=vggt_dpt_source_feat.npy \
  task.dataset.vggt_conf_name=vggt_source_conf.npy \
  task.dataset.vggt_view_count_name=vggt_source_view_count.npy \
  gpu=0
```

#### Uniform VGGT + confidence + frozen MLP

To reproduce the `uniform-vggt-conf-frozen` setup with dense DPT features,
extract a separate uniformly averaged feature bundle:

```bash
for split in train val; do
  GPUS=0,1,2,3 \
  EXTRA_ARGS="--frame_mode visit_average --view_aggregation uniform" \
    pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh "$split"
done
```

Then train the frozen-base confidence-conditioned MLP:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_dpt_uniform_adapter_from_base.sh
```

This uses `vggt_dpt_feat_uniform.npy` with the same relative-confidence branch
as the original setup. The trainable adapter is
`LayerNorm(128) -> Linear(128,128) -> SiLU`, plus a bias-free confidence
projection, followed by a zero-initialized `Linear(128,32)` residual. The base
PointTransformer and contact head remain frozen.

#### Additive stem fusion (recommended)

The simpler spatial design replaces the late MLP with

```text
z = Linear(128, 32, bias=False)(LayerNorm(dpt_feature))
g = seen * sigmoid(a * normalized_log_confidence + b)
point_stem = mask_stem + g * z
```

The projection is zero-initialized, while `a` and `b` are learned scalars. The
mask stem and every base checkpoint key remain unchanged. Fusion happens before
the first PointTransformer neighborhood-attention block, so the frozen spatial
backbone propagates VGGT information through all encoder and decoder stages.
Only 4,354 parameters train, with no hidden fusion MLP.

Use the same uniformly averaged DPT feature bundle, then run:

```bash
SCRIPT=pipeline/step8_3d_training_vggt/run_vggt_dpt_uniform_additive_stem_from_base.sh

GPUS=0,1,2,3 "$SCRIPT" --dry-run
GPUS=0,1,2,3 "$SCRIPT"
```

This selects `model=cdm_vggt_additive_stem_dpt_frozen` and evaluates the exact
mask-only initialization at epoch zero before optimizing the projection and
confidence scalars.

### Controlled residual-adapter ablations

The residual design preserves the mask-only PointTransformer path. It maps a
256-D VGGT point feature through
`LayerNorm -> Linear(256,128) -> SiLU -> Linear(128,32)` and adds the result to
the PointTransformer's 32-D output immediately before the existing contact
head. The final adapter layer is initialized to zero, so all variants start
with logits exactly equal to the mask-only checkpoint.

Materialize uniformly averaged features from the legacy visit caches once:

```bash
for split in train val; do
  conda run -n pt python \
    pipeline/step7b_vggt_feats/materialize_uniform_vggt_features.py \
    --data_root scenefun3d \
    --split "$split" \
    --processed_dir "scenefun3d/processed_sam2/$split" \
    --feat_root scenefun3d/processed_sam2/vggt_features
done
```

Then run the four experiments sequentially on the same GPUs:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_residual_ablation_matrix.sh \
    scenefun3d/processed_sam2
```

The matrix holds the initialization checkpoint, seed, dataset, global batches
(64 train / 32 validation), optimizer, 5,800-step budget, threshold, and
validation schedule fixed. It rejects incomplete caches and reused output
directories. Its variants are:

| Tag | VGGT adapter input | View aggregation |
|---|---|---|
| `a0_mask_only` | none | n/a |
| `a1_residual_uniform` | VGGT feature | uniform |
| `a2_confidence_uniform` | VGGT feature + normalized log-confidence | uniform |
| `a3_confidence_weighted` | VGGT feature + normalized log-confidence | confidence-weighted |

The last three use the same adapter parameters and initialization. In the
feature-only run, a zero scalar replaces confidence at a bias-free projection;
this makes the a1-to-a2 comparison an input ablation rather than a model-size
change. Each completed run prints and records its best validation mAP, mIoU,
precision, recall, epoch, and step.

### Frozen adapter on the July 29 base

Use the dedicated launcher for a true adapter-only continuation of
`outputs/2026-07-29_20-29-16_base/ckpt/mask_refinement_model_best.pt`:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_adapter_from_base.sh --dry-run

GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_adapter_from_base.sh
```

This preset uses uniformly aggregated VGGT features plus normalized
confidence, validates and checkpoints the zero-residual initialization at
step 0, and then optimizes only `vggt_adapter`. The PointTransformer, contact
head, and PointTransformer BatchNorm buffers remain fixed. The run uses the
same 5,800-step budget and global batches (64 train / 32 validation) as the
controlled matrix, and checks for exactly 9,257 train and 5,619 validation
frames. Override `BASE_CKPT`, `EXP_DIR`, or other documented launcher
environment variables when needed.

### Enriched reliability adapter

This experiment preserves the same frozen base and zero-initialized residual,
but gives the reliability projection four inputs per point:

1. within-sample normalized log confidence;
2. globally normalized absolute `log1p(confidence)`;
3. globally normalized `log1p(view_count)`;
4. an observed/unobserved flag.

Run the uniform-feature experiment with:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_enriched_reliability_from_base.sh \
    --dry-run

GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_enriched_reliability_from_base.sh
```

The fixed normalization moments were computed over all observed points in the
9,257 non-empty training samples and are recorded in
`outputs/vggt_reliability_stats/train_nonempty.json`. Recompute them after a
dataset or extraction change with:

```bash
conda run -n pt python scripts/compute_vggt_reliability_stats.py \
  --processed-sam2-dir scenefun3d/processed_sam2 \
  --vggt-feature-root scenefun3d/processed_sam2/vggt_features \
  --output outputs/vggt_reliability_stats/train_nonempty.json
```

Set `VGGT_FEATURE_NAME=vggt_feat.npy` to combine the enriched inputs with
confidence-weighted view aggregation. The default remains
`vggt_feat_uniform.npy` so the experiment isolates the reliability change.

For standalone evaluation, select the matching model configuration:

```bash
python pipeline/step8_3d_training_vggt/eval_vggt.py \
  exp_dir=outputs/<enriched-reliability-run> \
  model=cdm_vggt_adapter_enriched_reliability_frozen \
  task.dataset.vggt_feat_name=vggt_feat_uniform.npy \
  gpu=0
```

### Base-preserving early fusion

The controlled early-fusion launcher reproduces the important architectural
property of the July 27 `CDMVGGT` run: a projected VGGT feature is concatenated
with the mask before PointTransformer, so every encoder/decoder stage processes
the fused representation. Unlike the historical run, it starts exactly from a
trained mask-only checkpoint:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_early_fusion_from_base.sh --dry-run

GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_early_fusion_from_base.sh
```

The PointTransformer input stem is represented as separate mask and VGGT
linear blocks. The mask block and every downstream model weight are copied
strictly from the July 29 base; the VGGT block starts at zero. Consequently,
the epoch-0 logits are exactly the base logits even though the model has a
33-channel `[mask, VGGT]` PointTransformer input. Epoch 0 is evaluated and
checkpointed before optimization.

By default, confidence-weighted `vggt_feat.npy` is combined with the explicit
confidence/view-count gate. For the first 725 steps (five epochs), only
parameters whose names contain `vggt` train, at `1e-4`. PointTransformer and
the contact head then unfreeze at `1e-5` while the VGGT path remains at
`1e-4`. BatchNorm running statistics stay fixed throughout. Override
`VGGT_FEAT_NAME=vggt_feat_uniform.npy` to run the otherwise identical
uniform-view experiment.

Evaluate a resulting run with the matching architecture and feature file:

```bash
python pipeline/step8_3d_training_vggt/eval_vggt.py \
  exp_dir=outputs/<early-fusion-run> \
  model=cdm_vggt_early_fusion_controlled \
  task.dataset.vggt_feat_name=vggt_feat.npy \
  gpu=0
```

### Post-stem fusion

The post-stem variant first applies the checkpointed `1 -> 32` mask stem
without VGGT. It then concatenates that completed mask embedding with the
gated 32-D VGGT projection and adds a learned `64 -> 32` residual before the
first PointTransformer block. The residual weights and bias start at zero, so
the initialization is exactly the mask-only model while the mask embedding has
an explicit identity path around fusion.

Run the otherwise identical controlled experiment with:

```bash
GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_post_stem_fusion_from_base.sh \
    --dry-run

GPUS=0,1,2,3 \
  pipeline/step8_3d_training_vggt/run_vggt_post_stem_fusion_from_base.sh
```

This uses the same base checkpoint, confidence-weighted VGGT features,
5,800-step budget, 725-step VGGT-path warmup, learning rates, global batches,
validation schedule, and fixed BatchNorm statistics as controlled early
fusion.

## 3. Evaluate

```bash
python pipeline/step8_3d_training_vggt/eval_vggt.py \
    exp_dir=outputs/<your-exp-dir> gpu=0
```

## Config knobs

- `model.vggt_feat_dim` — use 128 for the default DPT latent. For the legacy
  patch source, it must equal `--out_dim` (256 by default).
- `model.vggt_proj_dim` — learned projection and point-wise gate width. Gate
  logits combine projected feature content with normalized log confidence,
  `log1p(view_count)`, and an observed/unobserved flag. The gated result is
  concatenated with the mask channel (backbone input dim =
  `1 + vggt_proj_dim`).
- `model.adapter_hidden_dim` — hidden width of the residual VGGT adapter (128
  by default in `cdm_vggt_adapter.yaml`).
- `task.dataset.vggt_feat_root` — common feature root containing the mirrored
  `train/` and `val/` trees (default:
  `scenefun3d/processed_sam2/vggt_features`).

# VGGT-conditioned 3D Affordance Refinement (parallel path)

Alternative to `pipeline/step8_3d_training/` that conditions the mask
refinement network on **per-point VGGT features** in addition to the initial
lifted mask. No existing file is modified; this path adds:

| File | Role |
|---|---|
| `pipeline/step7b_vggt_feats/extract_vggt_features.py` | Offline extraction: lifts VGGT aggregator tokens onto the cropped laser-scan points and writes `vggt_feat.npy` per sample dir |
| `dataset/AffordanceDatasetVGGT.py` | `AffordanceDataset` subclass that adds `c_vggt_feat` (and aliases `pred_mask_local`) |
| `models/cdm_vggt.py` | `CDMVGGT`: LayerNorm+Linear projection of VGGT features, concatenated with the mask channel into `PointTransformerSeg` |
| `pipeline/step8_3d_training_vggt/{train,eval}_vggt.py` + `configs/` | Hydra entries mirroring `train_no_diff.py` / `eval_no_diff.py` |

## 1. Extract features (once per split)

Run in the `tasa` env (torch 2.4.1+cu124). Downloads `facebook/VGGT-1B` from
the HF hub on first use (or pass `--ckpt /path/to/model.pt`).

```bash
# training samples (division_8192 layout)
python pipeline/step7b_vggt_feats/extract_vggt_features.py \
    --data_root scenefun3d --split train \
    --processed_dir data/processed_data/division_8192/train

# eval samples (processed_sam2 layout)
python pipeline/step7b_vggt_feats/extract_vggt_features.py \
    --data_root scenefun3d --split val \
    --processed_dir data/processed_sam2/val
```

Notes:
- Features are computed **once per visit** on the cropped laser scan and cached
  in `outputs/vggt_feat_cache/`, then gathered per sample via exact KD-tree
  matching; re-runs are cheap.
- `--out_dim 256` (default) applies a fixed seeded orthonormal projection of
  the 2048-dim tokens; the matrix is saved in the cache dir. Use `--out_dim 0`
  for raw 2048-dim tokens (larger files; set `model.vggt_feat_dim=2048`).
- Projection uses the SceneFun3D hires poses/intrinsics/depth (same convention
  as step7), **not** VGGT's own predicted cameras. Points unseen by every
  sampled frame get zero features.
- Parallelize over scenes with `--visit_id`; tune `--frames_per_video` /
  `--chunk_size` for GPU memory.

## 2. Train

```bash
CUDA_VISIBLE_DEVICES=0 python pipeline/step8_3d_training_vggt/train_vggt.py \
    exp_name=vggt_refine \
    task.train.batch_size=64 \
    task.train.max_steps=200000
```

Uses the same `SimpleMaskRefinementTrainLoop` (losses, checkpoint naming) as
the original path; checkpoints land in `outputs/<date>_<exp_name>/ckpt/`.

## 3. Evaluate

```bash
python pipeline/step8_3d_training_vggt/eval_vggt.py \
    exp_dir=outputs/<your-exp-dir> gpu=0
```

## Config knobs

- `model.vggt_feat_dim` — must equal the extractor's `--out_dim` (256 default).
- `model.vggt_proj_dim` — learned projection width concatenated with the mask
  channel (backbone input dim = `1 + vggt_proj_dim`).
- `task.dataset.vggt_feat_root` — set if extraction used `--feat_out_root` to
  write features into a mirror tree instead of the sample dirs.

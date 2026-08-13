## [AAAI 2026]Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement

## Environment Setup

Python 3.10+ and a virtual environment are recommended.

```bash
git clone https://github.com/LianHe00/TASA-main.git
cd affseg
conda create -n affseg python=3.10
conda activate affseg
pip install -r requirements.txt
```

Notes:
- `pointops_cuda` (used by PointTransformer) may need to be built separately for your CUDA / PyTorch version (see PointTransformer official docs or project-specific notes).
- Weights for large models (Qwen, Molmo, SAM2, etc.) are not shipped with the repo; they are downloaded via `transformers` or official repositories when needed.

---

## Data Preparation (Brief)

We use data from the [Scenefun3D](https://scenefun3d.github.io) dataset. The downloaded data includes the following assets:

- **laser_scan_5mm** — dense 3D point clouds (5 mm resolution)
- **crop_mask** — segmentation masks used to crop the scene
- **annotations** — affordance / contact annotations
- **descriptions** — natural language descriptions per visit
- **hires_wide** — high-resolution wide-angle RGB frames
- **hires_wide_intrinsics** — camera intrinsics for the wide-angle frames
- **hires_depth** — depth maps aligned to the RGB frames
- **hires_poses** — camera poses (extrinsics) for 2D→3D projection

---

## Typical Usage

### 1. Text instructions → objects & affordances

```bash
python pipeline/step1_affordance/qwen.py \
  --input_instructions path/to/instructions.json \
  --output path/to/affordance.json
```

Output includes target object categories and affordance labels (e.g. door handle, drawer handle).

### 2. Frame selection & operation point prediction

- Use CLIP to select the top-K frames most relevant to the affordance:

```bash
python pipeline/step2_clipwithaffordance/clip_affordance.py \
  --frames_dir path/to/frames \
  --affordance_json path/to/affordance.json \
  --output path/to/topk_frames.json
```

- Filter relational instructions so every retained frame visibly contains the
  target, its reference objects, and the stated spatial relation:

```bash
pipeline/step2b_relational_filter/run_step2b_parallel.sh val
```

- Use Qwen-VL to predict operation points on those frames:

```bash
pipeline/step3_point_prediction/run_step3_parallel.sh val \
  --input-root path/to/step2b_results \
  --output-root path/to/points
```

The input root must contain `val/<visit_id>/*_result.json`; the output root
receives `val/<visit_id>/*_point.json`. The same paths can be provided through
the `INPUT_ROOT` and `OUTPUT_ROOT` environment variables. For a single process,
pass `--input_root` and `--output_root` to `qwen_point.py` together with
`--split`.

- Crop around the predicted operation points for segmentation:

```bash
python pipeline/step4_crop_images/qwen_seg_image.py \
  --split val \
  --input_root path/to/step3_points \
  --output_root path/to/cropped_images
```

Step 4 appends `--split` to both roots, reading
`<input_root>/<split>/<visit_id>/*_point.json` and writing beneath
`<output_root>/<split>/`. The legacy `--data_root` input option remains
available with the same split-appending behavior.

### 3. 2D segmentation & 2D mask merging

```bash
# Segment with Molmo + SAM / SAM2
pipeline/step5_molmo_sam/run_step5_parallel.sh val \
  --input-root path/to/step4_crops \
  --output-root path/to/step5_masks

# Merge local masks back to full image coordinates
python pipeline/step6_molmo_merge/molmo_merge.py \
  --split val \
  --cropinfo_root path/to/step4_crops \
  --molmo_root path/to/step5_masks \
  --merge_root path/to/step6_masks

# Expand the Step 6 frame set with VGGT mask-pixel tracks (optional)
pipeline/step6b_vggt_track/run_step6b_parallel.sh val \
  --input-root path/to/step6_masks \
  --output-root path/to/step6b_masks
```

Steps 5, 6, and 6b append the selected split to every input and output root.
For example, `--output-root path/to/output` with `val` writes beneath
`path/to/output/val`.

### 4. 2D→3D lifting & point cloud extraction

```bash
python pipeline/step7_lift_3d/molmo_lift_2d_to_3d.py \
  --data_root scenefun3d \
  --split val \
  --molmo_root pipeline/step6b_vggt_track/vggt_track_output/val \
  --output_dir pipeline/step7_lift_3d/lift_output_vggt_track/val
```

Step 6b preserves all Step 6 masks and adds nearby frames whose mask pixels can
be tracked confidently by VGGT. Step 7 uses SceneFun3D geometry utilities to
project either the Step 6 or Step 6b masks into 3D.

### 5. 3D training & evaluation (no diffusion)

- **Training (main entry)**

Run `train_no_diff.py` with Hydra; see `scripts/train.sh` for an example:

```bash
bash scripts/train.sh perceiver_division_8192_pointtransformer_with_new_loss \
  scenefun3d/processed_sam2
```

Or run directly:

```bash
TASA_PROCESSED_SAM2_DIR=scenefun3d/processed_sam2 CUDA_VISIBLE_DEVICES=0 \
python pipeline/step8_3d_training/train_no_diff.py \
  exp_name=perceiver_division_8192_pointtransformer_with_new_loss \
  output_dir=outputs \
  platform=TensorBoard \
  diffusion.steps=500 \
  task=contact_gen \
  task.train.batch_size=64 \
  task.train.max_steps=50000 \
  model=cdm
```

Training uses `AffordanceDataset` to load one 3D frame crop and its local mask
per sample. PointTransformer predicts and is supervised on each frame crop
independently. The **diffusion module is not used**.
The mask objective emphasizes Dice/IoU overlap (70% combined) and uses
positive-weighted focal loss to counter point-level class imbalance.
Validation runs once per epoch on individual frames. Training writes a
numbered checkpoint after every validation and updates
`mask_refinement_model_best.pt` whenever the configured monitored metric
improves. Set `task.train.best_checkpoint_ap50_floor=<value>` to restrict best
checkpoint eligibility to validations meeting that AP50 floor; the selected
checkpoint is then the highest-scoring monitored metric among eligible epochs.
Early-stopping patience remains inactive until the first eligible checkpoint.
Validation also reports per-frame precision, recall, and mIoU at the configured
mask threshold. mAP is the macro mean of per-frame pointwise average precision
computed from continuous probabilities; AP25/AP50 and mAR/AR25/AR50 retain the
legacy hard-mask reporting fields.
Each validation checkpoint also gets evaluator-compatible `results.json`,
`metadata.json`, `test.log`, and `viz/` outputs under
`eval/mask_refinement_model<step>/test-<time>/`, so new runs do not
need a separate all-checkpoints evaluation pass.

- **Evaluation**

```bash
python pipeline/step8_3d_training/eval_no_diff.py \
  exp_dir=outputs/<your-exp-dir> \
  gpu=0
```

The script uses `mask_refinement_model_best.pt` when available (otherwise the
latest numbered checkpoint), runs per-frame 3D segmentation evaluation on the
validation set, and writes `results.json`.

To evaluate every checkpoint in a run and then collect the training and
evaluation curves:

```bash
python3 scripts/eval_all_checkpoints.py outputs/<your-exp-dir> \
  --cuda-visible-devices 0

python3 scripts/plot_metrics.py outputs/<your-exp-dir>
```

Batch evaluation remains useful for older or missing results and writes one
resumable result tree per checkpoint directly below `eval/`. Pass additional
Hydra overrides after `--`, for example
`-- task.evaluator.eval_nbatch=32`. For the VGGT model, select its evaluator
with `--eval-script pipeline/step8_3d_training_vggt/eval_vggt.py`.

The metrics script writes `training_metrics.csv`, `evaluation_metrics.csv`,
`training_metrics.png`, `loss_metrics.png`, and `evaluation_metrics.png` below
`metrics/`. It can also compare multiple experiments by passing multiple
experiment directories.

---

## Configuration and Experiment Management

The project uses **Hydra + OmegaConf** for configuration:

- Default configs: `pipeline/step8_3d_training/configs/` (add or edit configs here for local experiments).
- Each run creates an experiment directory under `outputs/<date>_<time>_<exp_name>` with:
  - `log/runtime.log`: full config and training log
  - `ckpt/`: model checkpoints
  - `eval/`: evaluation outputs

You can override options (e.g. use color, load text, point count, batch size) via the command line or config files.

---

## Contributing and Extension

- To adapt to other scene datasets or robot platforms you can:
  - Implement a new `DataParser` or extend `AffordanceDataset`;
  - Add new backbones or attention modules under `models/`;
  - Add new `pipeline/stepX_*` steps for different multimodal models (e.g. replace Qwen, Molmo, SAM2).

- For pull requests, please include:
  - A short description of the change;
  - Example commands or a minimal reproduction script where relevant.

---

## Acknowledgement

This project is built on top of prior efforts in 3D functionality understanding. In particular, we took inspiration from **Fun3DU** ([GitHub repository](https://github.com/tev-fbk/fun3du)), and we use the **SceneFun3D** dataset ([project page](https://scenefun3d.github.io/)) as our main data source. We sincerely thank the authors of Fun3DU and SceneFun3D for releasing their code and dataset to the community.

## Citation

If you find this work helpful, please cite:

```text
@article{he2025task,
  title={Task-Aware 3D Affordance Segmentation via 2D Guidance and Geometric Refinement},
  author={He, Lian and Liu, Meng and Ye, Qilang and Zhou, Yu and Deng, Xiang and Ding, Gangyi},
  journal={arXiv preprint arXiv:2511.11702},
  year={2025}
}
```

---

## License

This project is released under the **Apache License 2.0**. See `LICENSE` for the full license text.

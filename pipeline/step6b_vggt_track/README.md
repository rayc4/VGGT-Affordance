# Step 6b: VGGT mask-track frame expansion

Step 6b expands the frame set produced by Step 6 while retaining its exact
consumer-facing format. For each `*_mask_data.npz` source frame, it:

1. selects nearby, not-yet-published RGB frames from the same SceneFun3D video;
2. places the source at VGGT sequence index zero;
3. samples foreground pixels from every source mask and tracks them with
   VGGT's tracking head;
4. keeps target frames with enough visible, confident tracks; and
5. warps every foreground source pixel with its nearest track displacement.

The output is a strict superset of the Step 6 frame inventory. Input and output
roots both contain a split directory. Original NPZs
and red-overlay JPEGs are copied unchanged. Each accepted new frame is written
as:

```text
<output>/<split>/<visit_id>/<video_id>/<desc_id>/
  <video_id>_<timestamp>_mask_data.npz  # key: masks, shape [M,H,W]
  <video_id>_<timestamp>_mask_000.jpg
```

When multiple source masks reach the same new frame, compatible proposals are
deduplicated by mask IoU and the remaining masks are stacked in the NPZ.
Frames that already exist in Step 6 always retain their original masks.

## Run

The VGGT checkpoint is loaded once by each worker. One worker per GPU is
recommended because VGGT-1B is large.

```bash
pipeline/step6b_vggt_track/run_step6b_parallel.sh val
pipeline/step6b_vggt_track/run_step6b_parallel.sh train
```

Useful overrides:

```bash
GPUS="0 1" EXTRA_ARGS="--local_files_only --context_frames 12" \
  pipeline/step6b_vggt_track/run_step6b_parallel.sh val \
  --input-root path/to/step6 --output-root path/to/step6b

CKPT=/path/to/model.pt GPUS=0 \
  pipeline/step6b_vggt_track/run_step6b_parallel.sh val
```

The direct single-worker command is:

```bash
conda run -n affordvggt python pipeline/step6b_vggt_track/vggt_track.py \
  --split val \
  --local_files_only
```

Completed description directories are skipped on rerun. Pass `--overwrite`
to recompute them. The most relevant quality controls are
`--visibility_threshold`, `--confidence_threshold`, `--min_track_fraction`,
and `--context_window_seconds`.

## Step 7 handoff

Point Step 7 at the Step 6b superset:

```bash
MOLMO_ROOT=pipeline/step6b_vggt_track/vggt_track_output/val \
OUTPUT_DIR=pipeline/step7_lift_3d/lift_output_vggt_track/val \
SPLIT=val pipeline/step7_lift_3d/run_step7_parallel.sh
```

Step 7 requires no format changes because it already consumes all
`*_mask_data.npz` files below each description directory.

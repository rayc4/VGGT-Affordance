# Qwen2.5 relational VGGT repair experiment

This is an isolated repair-only experiment. It does not modify or consume the
previous direct-anchor proposal roots, and it never replaces a valid leaf from
`scenefun3d/processed_sam2`.

## Process

1. Inventory the repository-original six-file samples and create an
   `uncovered` repair plan. Only descriptions with invalid original leaves and
   no valid original leaf are included.
2. Retrieve fresh full-frame candidates using the complete action description,
   the target object, and its explicit component.
3. Qwen2.5-VL produces a strict target signature containing the target,
   constraints, and required reference objects.
4. A frame is grounded only when the target is visible and unambiguous, every
   required reference is visible, and every constraint is satisfied. Missing
   reference objects are terminal abstentions for that frame.
5. Qwen2.5-VL independently verifies the point on a full image marked with a
   magenta ring. Relocation is disabled. Rejected frames cause lazy retrieval
   of another candidate rather than a guessed point.
6. SAM converts each verified point into a nonempty semantic anchor mask.
7. VGGT tracks positive and negative anchor pixels through eight nearby views;
   SAM refines tracked prompts, and prediction-only 3D component agreement
   filters propagated masks. Up to six compatible views are published.
8. The original-first union hardlinks every valid original leaf unchanged,
   excludes invalid originals, and adds only valid planned repairs.

Ground truth cannot select a target, frame, track, component, crop, or published
view. The semantic-alignment report is evaluation-only and is not consumed by
any later stage.

## Output contract

The final root is:

`scenefun3d/processed_sam2_qwen25_relational_vggt_repair_v1`

Each sample uses the unchanged original six-file contract, so training uses the
existing dataset loader and command. The final audit requires original hardlink
identity, collision safety, zero empty local masks, valid loader smoke tests,
and validation coverage of at least 326/445 descriptions. The stronger useful
target, reported but not used for selection, is 333/445.

## Run

```bash
GPUS=0,1,2,3,4,5,6,7 N_WORKERS=8 bash dataset/run_qwen25_relational_vggt_repair_experiment.sh --with-train && \
GPUS=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 bash scripts/train.sh qwen25_relational_vggt_repair_v1 scenefun3d/processed_sam2_qwen25_relational_vggt_repair_v1
```

Use `--dry-run --with-train` to print the complete val/train command expansion
without loading models or creating artifacts.

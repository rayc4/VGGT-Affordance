# Original preprocessing + VGGT-track hybrid

This experiment preserves the repository's semantic preprocessing through
step 6. It reads merged masks from
`pipeline/step6_molmo_merge/molmo_merge_output/<split>` and point predictions
from `pipeline/step3_point_prediction/point_clipwithaffordance_output`, then
uses VGGT/SAM multi-view tracking, anchor-consensus 3D fusion, and a
prediction-only semantic crop.

Those two input roots are pinned by the runner: setting `STEP6_ROOT` or
`POINT_ROOT` to direct-anchor, v3, or other experiment artifacts is rejected.
This runner is specifically the enhancement of repository-original steps
1–6, not a generic source-ablation wrapper. Ambient `NO_POINT_ROOT` and
`ALLOW_COPY_FALLBACK` values are ignored: step-3 evidence and fail-closed
hardlink storage are mandatory. `VGGT_EXTRA_ARGS` may add model/runtime
options, but the hybrid selection, crop, and publication preset is appended
by the runner and cannot be removed through that variable.

Consensus uses `--independent-anchor-video-votes`: every original step-6
anchor and every physical video contributes at most one component vote.
Propagated frames still improve the mask and geometry, but repeatedly tracking
one mistaken anchor can no longer manufacture extra semantic support. Distinct
video agreement chooses the component first; step-3 semantic validity breaks
ties, so one confident but wrong video cannot overrule two independent views.
The split guard fingerprints the step-6 NPZ inventory and hashes the step-3
JSON inputs, preventing an in-place source change from mixing stale and new
hybrid leaves during resume.

It never writes into either source tree or `scenefun3d/processed_sam2`.
Outputs use the unchanged six-file `processed_sam2` leaf contract under
`scenefun3d/processed_sam2_original_vggt_hybrid_v1`, so the existing dataset
loader and training entry point need no changes. Empty prediction-local and
GT-local masks are rejected before publication, and immutable repeated files
must use the content-addressed hardlink store.

Run validation on all GPUs first:

```bash
bash dataset/run_original_vggt_hybrid_experiment.sh
```

Validation must represent at least 323 of 445 descriptions (exact coverage
floor `0.7258426966292135`),
contain no malformed or empty-mask leaves, pass the ordinary loader smoke
test, respect the configured six-view cap even after a resumed run, and satisfy
required hardlink deduplication. `VAL_MIN_COVERAGE` may raise this floor but
cannot lower it. To print all commands and paths without running models, add
`--dry-run`.

After validation succeeds, process train with the same configuration and
audit its file/loader/empty-mask/hardlink contract:

```bash
bash dataset/run_original_vggt_hybrid_experiment.sh --with-train
```

The runner always repeats the validation audit before it permits train to
start. A failed validation therefore cannot silently produce or train on a
hybrid train tree. Runs resume valid completed leaves in the fresh output
root; no original artifact is deleted or rewritten.

Train with the repository's ordinary command:

```bash
GPUS=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 \
  bash scripts/train.sh original_vggt_hybrid_v1 \
  scenefun3d/processed_sam2_original_vggt_hybrid_v1
```

Logs and JSON audits are isolated under
`scenefun3d/preprocessing_experiments/original_vggt_hybrid_v1`. Use new roots
for ablations instead of mixing configurations in this experiment.

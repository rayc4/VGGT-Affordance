# Original + VGGT Repair-Only Experiment

This experiment keeps every valid sample from the repository-original
`processed_sam2` preprocessing and uses VGGT only for descriptions that have
at least one invalid original leaf. It is additive: a VGGT result cannot
replace or modify a valid original sample.

## Run

The validation split always runs first. The train split does not start unless
the final validation union preserves the original baseline and passes the
legacy loader/contract checks.

```bash
bash dataset/run_original_vggt_repair_experiment.sh --with-train && \
GPUS=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 \
bash scripts/train.sh original_vggt_repair_v1 \
scenefun3d/processed_sam2_original_vggt_repair_v1
```

To inspect every command and the GPU-to-shard assignment without writing data
or loading VGGT/SAM:

```bash
bash dataset/run_original_vggt_repair_experiment.sh --dry-run --with-train
```

## Flow and guarantees

For each split, the wrapper performs these operations in order:

1. Validate the pinned repository-original `processed_sam2` leaves and write a
   deterministic `any-invalid` description allowlist.
2. Run only that allowlist through the original step-6 masks, original step-3
   points, VGGT tracks, SAM propagation, independent-video consensus, and
   prediction-only adaptive crops on GPUs 0-7.
3. Audit the repair candidates for the unchanged six-file contract, nonempty
   local masks, and required content-addressed hardlinks.
4. Materialize a fresh hardlink union. Every valid original leaf wins an exact
   path collision; invalid originals are omitted; only valid in-plan repairs
   at unoccupied paths are added.
5. Audit source inode identity, baseline leaf/description preservation, zero
   invalid or empty leaves, benchmark membership, coverage, and the unchanged
   downstream loader.

The validation floor is exactly the repository-original usable coverage:
2,232 valid leaves representing 323 of 445 descriptions. A successful final
audit therefore cannot have lower baseline coverage. VGGT can increase
coverage when it repairs one of the original descriptions that had no valid
leaf.

Ground truth is used only to reject a fixed publication crop. It does not
select the component, crop, retry, source view, or published prediction.

## Isolated outputs

- Repair plans, logs, and diagnostics:
  `scenefun3d/preprocessing_experiments/original_vggt_repair_v1`
- VGGT repair candidates:
  `scenefun3d/processed_sam2_original_vggt_repair_candidates_v1`
- Training-ready original-first union:
  `scenefun3d/processed_sam2_original_vggt_repair_v1`

The original `scenefun3d/processed_sam2`, step-3 points, and step-6 masks are
read-only pinned inputs. Output roots must use their dedicated fresh
namespaces. Copy fallback and point-root disabling are forced off, including
when hostile ambient environment variables request them.

## Safety test

```bash
bash dataset/test_original_vggt_repair_runner.sh
```

This checks shell syntax, all eight worker GPUs, source pinning, repair-plan
propagation, hardlink enforcement, validation-before-train gate order, output
isolation, and hostile-environment rejection.

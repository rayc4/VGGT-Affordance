# Qwen2.5 relational VGGT standalone experiment

This v2 experiment measures the coupled relational method independently of the
repository-original preprocessing. Every benchmark description is eligible.
It does not inventory, copy, union, replace, or otherwise consume
`scenefun3d/processed_sam2`.

For each description, Qwen2.5-VL emits target-local qualifiers and coupled
relation/reference objects. Descriptions with no spatial constraint use empty
lists and remain valid. One deterministic repair call is allowed after a
schema or literal-support error. Full-frame grounding checks identity and
relations; mandatory verification sees both the marked full frame and a marked
point zoom, with relocation disabled.

SAM creates candidate semantic anchors. Before any VGGT inference, released
camera/depth geometry must lift each semantic point onto a nearby visible scan
point. The selector prefers a 0.20 m cluster supported by two independent
videos and otherwise permits only a confidence-at-least-0.85 single-anchor
fallback. VGGT then tracks positive and negative prompts through nearby frames,
SAM propagates masks, and prediction-only anchor/video consensus selects a 3D
component, an 8,192-point crop, and up to six compatible views. Strict
abstention is allowed at every semantic stage.

The final standalone root is:

`scenefun3d/processed_sam2_qwen25_relational_vggt_standalone_v2`

GT is loaded only after localization, component selection, cropping, and view
ranking are frozen. It can reject an empty local target but cannot steer a
published sample. The validation audit reports coverage against all 445
descriptions and requires a nonempty set of valid six-file leaves, zero empty
local masks, loader compatibility, and content-addressed hardlink integrity.
No replacement-quality coverage floor is assumed before the first full pilot;
raise `VAL_MIN_COVERAGE` when a justified promotion threshold is known.

Validation-only processing:

```bash
GPUS=0,1,2,3,4,5,6,7 N_WORKERS=8 \
  bash dataset/run_qwen25_relational_vggt_standalone_experiment.sh
```

Use `--dry-run` to print the complete command expansion without loading models
or writing artifacts. Use `--with-train` only after reviewing validation
coverage and the evaluation-only semantic-alignment report.

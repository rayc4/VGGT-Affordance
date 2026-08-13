# Step 2b: direct relational frame filtering

This stage sits between CLIP frame retrieval (Step 2) and Qwen point
prediction (Step 3). It is intentionally simple.

For every candidate frame belonging to a description with an external spatial
relation, Qwen receives the image and the complete description and answers one
token:

```text
YES
```

The frame is retained only when Qwen returns `YES`. A recognizable partial
object is enough; the complete object does not have to fit inside the frame.
The person, action, result, carried device, and an object controlled by a
visible switch do not need to be visible.

There is no relation graph, coordinate grounding, geometric test, crop pass,
or second model call. Descriptions without an external relation pass through
unchanged. Invalid model responses are rejected.

The main output preserves Step 2's JSON schema and only filters each
`image_name` list. A sibling `*_relational_filter.json` audit contains the
direct decision and raw `YES`/`NO` response. Schema version 4 causes older Step
2b outputs to be regenerated automatically.

Run one split on multiple GPUs:

```bash
pipeline/step2b_relational_filter/run_step2b_parallel.sh val
```

Or run one process:

```bash
python -m pipeline.step2b_relational_filter.relational_filter \
  --data_root scenefun3d \
  --split val
```

Test one image interactively with the exact same Qwen evaluator and prompt:

```bash
python -m pipeline.step2b_relational_filter.relational_filter_tester \
  scenefun3d/train_val_set/421254/42444754/hires_wide/42444754_81016.576.jpg \
  --description "Open the bottom drawer of the cabinet located to the left of the TV" \
  --local-files-only \
  --show-prompt
```

The tester prints Qwen's raw `YES`/`NO` response and the parsed decision. Add
`--output /tmp/step2b_test.json` to save the image path, prompt, and result.

The default output is
`pipeline/step2b_relational_filter/clipwithaffordance_output`. Step 3 uses this
root by default. Both stages are resumable. Pass `--overwrite` to regenerate
current-schema files as well.

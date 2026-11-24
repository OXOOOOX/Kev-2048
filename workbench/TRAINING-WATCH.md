# Local training progress

The training commands `train`, `experiment`, `improve`, `correction`, `round6-order` and `round7-scale` publish optimizer events to the run's `progress.jsonl`. Standalone `python -m kev2048_release.<module>` drivers that accept `--out` and call the shared `head_fit.fit` publish events during that fit. This covers the round 8, 9, 10 and 12 training drivers. The dashboard discovers logs under `runs/<run>/` while it is running.

### Custom entry points

Use `kev2048_release.training_progress.TrainingProgress` for a custom training or data preparation loop. Give it a new `runs/<run>` directory, or the verified directory of a resumed run. The context writes `starting` on entry and `completed`, `stopped`, or `error` on exit. Keep required output writes inside the context so `completed` reflects a successful run.

```python
from kev2048_release.training_progress import TrainingProgress

with TrainingProgress(run_dir) as progress:
    progress.emit_cache(split="train", done=cached_rows, total=train_size)
    progress.emit_optimizer(trial=trial_name, epoch=epoch, optimizer_step=step, steps=steps,
                            mean_training_loss=epoch_loss, samples_seen=epoch_boards_seen,
                            batch_rows=rows, batch_indices=indices, pre_update_logits=logits)
```

Call `emit_optimizer` after each successful optimizer update. `indices` are zero-based dataset indices in batch order. Pass one CPU Python score vector per row in that row's `valid_moves` order, from the same pre-update forward pass used for its loss. The publisher takes up to 64 samples from the batch and packs their legal-action probabilities. For a long stage, publish periodic progress from the work being performed. Use `emit_data_collection`, `emit_teacher_labeling`, `emit_cache`, `emit_evaluation`, or `emit(phase, status, ...)` for the appropriate phase. The experiment defines its trial label.

When the entry point manages its own lifecycle, use the same publisher directly with `emit` and `emit_terminal`. Keep one writer per journal. Keep detailed experiment logs alongside `progress.jsonl` using the [logging template](../experiments/plans/training-logging.template.md).

Before a long run, use the normal entry point for a short trial with a new run directory. At the workbench URL, check `/channels` for automatic discovery, changing progress and the correct end status. On the console page, open the run and inspect a recorded training batch for its board, teacher moves, and candidate probabilities. For data-only runs, inspect the stages the command actually performs. Record the commands and verification with the experiment.

### Optimizer events

Each optimizer update publishes an event with `schema: "kev2048.training.v1"`, `kind: "training"`, `phase: "optimizer"`, `status`, UTC `timestamp`, `elapsed_seconds`, `trial`, `epoch`, `optimizer_step`, `steps`, and `mean_training_loss`. Step counters are per epoch. Loss is the sample-weighted mean objective through that step in the epoch. Every event includes `batch_size` and up to 64 `samples` from the actual batch. A sample contains its one-based `row` in the training input, its 4×4 `board`, `best_actions`, and any available `stage` and `source_policy`.

Each newly recorded sample includes its pre-update `candidate.choice` and `candidate.probabilities`. The four direction keys are `up`, `down`, `left` and `right`. Legal probabilities are fractions of one and unavailable moves are `null`. The values come from the forward pass used for that sample's loss, including each microbatch in an accumulated update.

`samples_seen` counts all boards processed in the current epoch. After two updates of the same optimizer, trial and epoch, `boards_per_second` measures the change in that count divided by the elapsed time. In the training detail view, **按训练速度** uses that rate, capped at 12 boards per second. For older journals the dashboard estimates throughput from the time between logged steps and shows the interval used. Playback visits the boards in order and advances to the next available batch.

### Optional native cache progress

Use the existing wrapper to monitor a native run during the cache phase before the first optimizer update.

```sh
python workbench/watch_train.py --events runs/local-a.progress.jsonl -- --preset smoke --base assets/qwen-base --checkpoint assets/official-kev --out runs/local-a
```

Use a new output and event filename for each run. Keep the wrapper feed beside the training output directory, as in the example. The dashboard automatically discovers `runs/<name>.progress.jsonl` and switches to `runs/<name>/progress.jsonl` once its optimizer events are available. For a different location, pass the journal via `--watch local-a:training:path/to/progress.jsonl`.

The wrapper reads `train: cached N/T` and `dev: cached N/T` markers from stdout. Every eight seconds it repeats the latest recorded marker as a heartbeat while the process is alive. An expired heartbeat is shown as stale.

Wrapper events contain UTC `timestamp`, `elapsed_seconds`, `status`, and `phase`. Cache markers include `step` and `total`. A nonzero exit becomes `error`, and a caught Ctrl+C becomes `stopped`. `completed` requires a successful process exit and the expected result file.

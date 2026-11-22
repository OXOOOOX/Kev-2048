# Optional local training feed

The dashboard reads only a local progress file. It never launches, resumes, or stops training. Start the following explicitly in a separate terminal after installing the native CLI and preparing its model inputs. No cloud or public service is implied.

```sh
python workbench/watch_train.py --events runs/local-a.progress.jsonl -- --preset smoke --base assets/qwen-base --checkpoint assets/official-kev --out runs/local-a
```

Use a new output and event filename for each run. Point the dashboard at the event file, e.g. via `--watch local-a:training:runs/local-a.progress.jsonl` when starting the local dashboard. Do not put the event file inside the training output directory: the native command requires that its output directory not already exist.

The existing CLI emits real `train: cached N/T` and `dev: cached N/T` markers on stdout. The watcher exposes those per-phase counts only. It does not fabricate epoch progress, percent complete, training losses, or loader estimates. The native CLI currently has no per-epoch stdout progress. A heartbeat every eight seconds preserves the same phase/counters while the process is alive; it means process liveness, not an advancing step. During unannounced intermediate work the last real marker remains on display. When the watcher is abruptly killed the heartbeat stops and the dashboard can mark it stale, not complete. Worker stderr remains in the operator terminal and is not forwarded to the dashboard. The normal training result file is not copied to the feed.

Normalized JSONL rows contain UTC `timestamp`, measured `elapsed_seconds`, `status`, and a fixed `phase`; cache markers include `step` and `total`. No raw CLI text, commands, paths, checkpoint values, or model outputs are attached to a row. A nonzero exit becomes `error`; a caught Ctrl+C becomes `stopped`. `completed` requires a successful process exit and its expected final result artifact.

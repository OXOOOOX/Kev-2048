"""Optional local operator command; emits safe training NDJSON, never used by HTTP."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from queue import Empty, SimpleQueue
import re
import subprocess
import sys
import threading
import time

MARKER = re.compile(rb"(train|dev): cached ([0-9]{1,8})/([0-9]{1,8})\s*")
HEARTBEAT_SECONDS = 8.0


def run_worker(argv, output, event_path, *, heartbeat=HEARTBEAT_SECONDS):
    """Run a command already built by main; tests inject a known fake worker here."""
    output = Path(output).resolve()
    event_path = Path(event_path).resolve()
    if len(argv) == 0 or heartbeat <= 0:
        raise ValueError("missing worker or nonpositive heartbeat")
    if output.exists():
        raise FileExistsError("training output must be a new directory")
    if event_path == output or output in event_path.parents:
        raise ValueError("progress must be a sibling of or outside the training output")
    event_path.parent.mkdir(parents=True, exist_ok=True)
    now = time.monotonic
    with event_path.open("x", encoding="utf-8", newline="\n") as stream:
        started = now()
        state = {"status": "loading", "phase": "starting"}
        def emit(new_state=None):
            if new_state is not None:
                state.clear()
                state.update(new_state)
            row = dict(timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                       elapsed_seconds=round(now()-started, 3), **state)
            stream.write(json.dumps(row, separators=(",", ":"), allow_nan=False) + "\n")
            stream.flush()

        emit()
        try:
            worker = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                      stderr=None, shell=False)
        except OSError:
            emit({"status": "error", "phase": "launch_failed"})
            return 1

        queue = SimpleQueue()
        def drain():
            # Only complete, narrowly matched stdout lines reach the progress stream.
            # Do not parse stderr: warnings and model loader counters are not CLI markers.
            assert worker.stdout is not None
            try:
                while line := worker.stdout.readline(8192):
                    if not line.endswith(b"\n"):
                        while line and not line.endswith(b"\n"):
                            line = worker.stdout.readline(8192)
                        continue
                    m = MARKER.fullmatch(line)
                    if m:
                        part, a, b = m.groups()
                        queue.put((part.decode("ascii"), int(a), int(b)))
            finally:
                worker.stdout.close()
                queue.put(None)

        listener = threading.Thread(target=drain, name="train-progress-reader", daemon=True)
        listener.start()
        markers = {}
        deadline = now() + heartbeat
        eof = False
        exited_at = None
        try:
            while worker.poll() is None or not eof:
                try:
                    x = queue.get(timeout=min(0.2, max(deadline-now(), 0.001)))
                except Empty:
                    pass
                else:
                    if x is None:
                        eof = True
                    else:
                        part, done, total = x
                        old = markers.get(part)
                        if (0 < done <= total
                            and (old is None or (done > old[0] and total == old[1]))
                            and not (part == "train" and "dev" in markers)
                            and (part != "dev" or (markers.get("train", (0, 1))[0]
                                                    == markers.get("train", (0, 1))[1]))):
                            markers[part] = (done, total)
                            emit({"status": "running", "phase": part+"_cache", "step": done, "total": total})
                            deadline = now() + heartbeat
                if now() >= deadline:
                    if worker.poll() is None:
                        emit()  # actual process liveness, exact existing counters or no counters
                    deadline = now() + heartbeat
                if worker.poll() is not None:
                    exited_at = now() if exited_at is None else exited_at
                    # A descendant could inherit the stdout handle. No heartbeat after
                    # the direct worker exits; avoid hanging on that unrelated handle.
                    if (not listener.is_alive() and queue.empty()) or now()-exited_at >= 2:
                        break
            rc = worker.wait()
        except KeyboardInterrupt:
            if worker.poll() is None:
                worker.terminate()
                try:
                    worker.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    worker.kill()
                    worker.wait()
            emit({"status": "stopped", "phase": "interrupted"})
            return 130
        if rc != 0:
            emit({"status": "error", "phase": "worker_exit_error"})
            return rc if rc > 0 else 1
        # Return code alone is insufficient if the expected CLI artifact is missing.
        if not (output/"result.json").is_file():
            emit({"status": "error", "phase": "missing_result"})
            return 1
        emit({"status": "completed", "phase": "result_saved"})
        return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    p = argparse.ArgumentParser(
        description="Run the existing local training CLI and write a safe, read-only progress feed",
        epilog="Arguments after -- are passed to the installed kev2048 train command; --out must be new.",
        allow_abbrev=False)
    p.add_argument("--events", required=True, type=Path)
    if "--" not in argv:
        p.parse_args(argv)
        p.error("separate native arguments with --")
    n = argv.index("--")
    a = p.parse_args(argv[:n])
    train = argv[n+1:]
    # Mirror the installed command parser, obtaining its exact --out without loading a model.
    from kev2048_release.cli import cli_parser
    native = cli_parser().parse_args(["train", *train])
    if native.command != "train":
        p.error("only the native train command is supported")
    cmd = [sys.executable, "-u", "-m", "kev2048_release.cli", "train", *train]
    try:
        return run_worker(cmd, native.out, a.events)
    except (FileExistsError, ValueError, OSError) as exc:
        # These errors occur in the explicitly invoked operator CLI, not in the feed.
        p.error(str(exc))

if __name__ == "__main__":
    raise SystemExit(main())

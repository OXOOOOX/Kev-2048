"""Publish workbench progress directly or observe the existing training entry points."""

from collections.abc import Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import logging
import math
from pathlib import Path
import re
import sys
from threading import get_ident
import time
from weakref import WeakKeyDictionary

from .game import DIRS, flatten, nested


SCHEMA = "kev2048.training.v1"
MAX_PREVIEW_SAMPLES = 64
_TERMINAL_STATUSES = frozenset(("completed", "stopped", "error"))

_NATIVE = ("kev2048_release.native", "train")
_EXPERIMENT = ("kev2048_release.head_fit", "fit")
_IMPROVE = ("kev2048_release.improve_fit", "fit")
_LORA = ("kev2048_release.improve_lora", "_fit_steps")
_ROUND6 = ("kev2048_release.round6_fit", "fit")
_TRAINERS = {"train": {_NATIVE}, "experiment": {_EXPERIMENT}, "improve": {_IMPROVE, _LORA},
             "correction": {_EXPERIMENT}, "round6-order": {_ROUND6}, "round7-scale": {_EXPERIMENT},
             "round8-retest": {_EXPERIMENT}, "round9-kl": {_EXPERIMENT}, "standalone-head": {_EXPERIMENT}}
_ACTIVE_JOURNAL = ContextVar("kev2048_training_journal", default=None)
_CAPTURE_ERRORS = (AttributeError, IndexError, KeyError, OverflowError, RuntimeError, TypeError,
                   ValueError, ZeroDivisionError)


def _function(frame):
    module = getattr(frame.f_globals.get("__spec__"), "name", None) or frame.f_globals.get("__name__")
    return module, frame.f_code.co_name


def _trial_name(frame):
    owner = frame.f_back
    while owner and _function(owner) in (("kev2048_release.head_fit", "fit_head"),
                                          ("kev2048_release.training_entry", "recorded_fit")):
        owner = owner.f_back
    if owner:
        match _function(owner):
            case ("kev2048_release.improve", "fit_trial"):
                return owner.f_locals["name"]
            case ("kev2048_release.experiment", "execute"):
                return owner.f_locals["trial_config"]["id"]
            case ("kev2048_release.correction_fit", "run"):
                return Path(owner.f_locals["out"]).name
            case ("kev2048_release.round7_fit", "fit_arm"):
                return owner.f_locals["arm"]
        if _function(frame) == _EXPERIMENT and owner.f_globals.get("__package__") == __package__:
            resolved = owner.f_locals.get("resolved")
            arm = resolved.get("arm") if isinstance(resolved, Mapping) else None
            arm = arm or owner.f_globals.get("ARM")
            if isinstance(arm, str) and arm:
                return arm
    if _function(frame) == _ROUND6:
        return "alternating"
    return "head"


def _sample(row, index):
    preview = dict(row=index + 1, board=nested(flatten(row["board"])), best_actions=list(row["best_actions"]))
    for key in ("stage", "source_policy"):
        if isinstance(row.get(key), str):
            preview[key] = row[key]
    return preview


def _candidate(moves, probabilities, choice_index):
    legal = dict(zip(moves, probabilities, strict=True))
    if (not 1 <= len(moves) <= len(DIRS) or len(legal) != len(moves)
            or not all(move in DIRS and 0 <= p <= 1 for move, p in legal.items())
            or not math.isclose(sum(legal.values()), 1., abs_tol=1e-5)):
        raise ValueError("Invalid candidate probability distribution")
    return dict(choice=moves[choice_index], probabilities={direction: legal.get(direction) for direction in DIRS})


def _preview(row, index, logits):
    moves = row["valid_moves"]
    if len(logits) != len(moves) or not logits or not all(math.isfinite(value) for value in logits):
        raise ValueError("Training logits must be finite and match the legal moves")
    choice = max(range(len(logits)), key=logits.__getitem__)
    weights = [math.exp(value - logits[choice]) for value in logits]
    total = math.fsum(weights)
    return {**_sample(row, index), "candidate": _candidate(moves, [weight / total for weight in weights], choice)}


class TrainingProgress:
    """Publish from a custom loop using ordinary Python values and a single writer per run."""

    def __init__(self, run_root, *, started_monotonic=None, native_writer=True):
        self.run_root = Path(run_root).resolve()
        self.path = self.run_root / "progress.jsonl"
        self.started = time.monotonic() if started_monotonic is None else started_monotonic
        self.native_writer = native_writer
        self._last_optimizer = {}

    def __enter__(self):
        self.emit("starting", "loading")
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        status = "completed"
        if exc_type is not None:
            status = "stopped" if issubclass(exc_type, KeyboardInterrupt) else "error"
        try:
            self.emit_terminal(status)
        except (OSError, ValueError):
            if exc_type is None:
                raise
            logging.getLogger(__name__).warning("Could not record the terminal training status", exc_info=True)

    def emit(self, phase, status, **fields):
        payload = {key: value for key, value in fields.items() if value is not None}
        payload.update(schema=SCHEMA, kind="training", phase=phase, status=status,
                       timestamp=datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                       elapsed_seconds=round(max(0.0, time.monotonic() - self.started), 3))
        if self.native_writer:
            payload["native_writer"] = True
        line = json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        self.run_root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(line)
        return payload

    def emit_data_collection(self, *, step, total, stage, complete_games=None, requested_games=None):
        return self.emit("data_collection", "running", step=step, total=total, stage=stage,
                         complete_games=complete_games, requested_games=requested_games)

    def emit_teacher_labeling(self, *, step, total, stage):
        return self.emit("teacher_labeling", "running", step=step, total=total, stage=stage)

    def emit_cache(self, *, split, done, total):
        phase = "dev_cache" if "dev" in split else "train_cache"
        return self.emit(phase, "running", step=done, total=total, stage=split)

    def emit_optimizer(self, *, trial, epoch, optimizer_step, steps, mean_training_loss, samples_seen,
                       batch_rows, batch_indices, pre_update_logits):
        """Record an update using zero-based dataset indices and the logits used for its loss."""
        size = len(batch_indices)
        if not size or size != len(batch_rows) or size != len(pre_update_logits):
            raise ValueError("Each batch index and row must have one vector of pre-update logits")
        if not isinstance(trial, str) or not re.fullmatch(r"[\w. -]{1,100}", trial):
            raise ValueError("Invalid training trial label")
        if not 1 <= optimizer_step <= steps or epoch < 1 or samples_seen < size:
            raise ValueError("Invalid epoch or optimizer step counters")
        cap = MAX_PREVIEW_SAMPLES
        samples = [_preview(row, index, logits) for row, index, logits in
                   zip(batch_rows[:cap], batch_indices[:cap], pre_update_logits[:cap], strict=True)]

        now = time.perf_counter()
        key = (trial, epoch)
        previous = self._last_optimizer.get(key)
        speed = None
        if previous is not None:
            old_step, old_seen, old_time = previous
            if old_step < optimizer_step and old_seen < samples_seen and old_time < now:
                speed = (samples_seen - old_seen) / (now - old_time)
        payload = self.emit("optimizer", "running", trial=trial, epoch=epoch, optimizer_step=optimizer_step,
                            steps=steps, mean_training_loss=float(mean_training_loss), batch_size=size,
                            samples_seen=samples_seen, samples=samples, boards_per_second=speed)
        self._last_optimizer[key] = (optimizer_step, samples_seen, now)
        return payload

    def emit_evaluation(self, *, trial, epoch, dev_loss, dev_best_rate, stage=None):
        return self.emit("evaluate", "running", trial=trial, epoch=epoch,
                         dev_loss=float(dev_loss), dev_best_rate=float(dev_best_rate), stage=stage)

    def emit_terminal(self, status, **fields):
        if status not in _TERMINAL_STATUSES:
            raise ValueError("Expected completed, stopped or error")
        return self.emit(status, status, **fields)


def _candidates(rows, scores, *, log_probabilities):
    """Pack the legal scores from this forward and transfer once per reported batch."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    counts = [len(row["valid_moves"]) for row in rows]
    if isinstance(scores, torch.Tensor):  # improve_fit pads logits to the training set's maximum width
        vectors = [row[:n] for row, n in zip(scores[:len(rows)].detach(), counts, strict=True)]
    else:
        vectors = [value.detach() for value in scores[:len(rows)]]
    if (len(vectors) != len(rows) or any(value.shape != (n,) or not 1 <= n <= len(DIRS)
                                         for value, n in zip(vectors, counts, strict=True))):
        raise ValueError("Training scores and legal moves disagree")
    packed = pad_sequence(vectors, batch_first=True, padding_value=-math.inf)
    if packed.dtype in (torch.float16, torch.bfloat16):
        packed = packed.float()
    probabilities = packed.exp() if log_probabilities else packed.softmax(-1)
    # Keep the original option-order tie break even if two rounded probabilities compare equal.
    choices = packed.argmax(-1, keepdim=True).to(probabilities.dtype)
    values = torch.cat((probabilities, choices), dim=1).cpu().tolist()
    return [_candidate(row["valid_moves"], numbers[:len(row["valid_moves"])], int(numbers[-1]))
            for row, numbers in zip(rows, values, strict=True)]


class _Journal:
    def __init__(self, out, command, *, started=None):
        self.out = Path(out).absolute()
        self.command = command
        self.progress = TrainingProgress(self.out, started_monotonic=started, native_writer=False)
        self.thread = get_ident()
        self.warned = False
        self.native_started = False
        self.head_losses = WeakKeyDictionary()
        self.micro_scores = WeakKeyDictionary()
        self.last_sample = WeakKeyDictionary()

    def _warn(self):
        if not self.warned:
            self.warned = True
            try:
                print("Training progress could not be recorded. Training will continue.", file=sys.stderr, flush=True)
            except (OSError, ValueError):
                pass

    def write(self, phase, status, **fields):
        try:
            self.progress.emit(phase, status, **fields)
        except (OSError, ValueError):
            self._warn()

    def _publish_batch(self, optimizer, batch_rows, indices, scores, *, trial, epoch, step, steps, loss, seen,
                       log_probabilities=False):
        now = time.perf_counter()
        previous = self.last_sample.get(optimizer)
        self.last_sample[optimizer] = (trial, epoch, step, seen, now)
        speed = {}
        if previous:
            old_trial, old_epoch, old_step, old_seen, old_time = previous
            if (old_trial, old_epoch) == (trial, epoch) and old_step < step and old_seen < seen and old_time < now:
                speed["boards_per_second"] = (seen - old_seen) / (now - old_time)
        rows = batch_rows[:MAX_PREVIEW_SAMPLES]
        samples = [_sample(row, i) for row, i in zip(rows, indices[:MAX_PREVIEW_SAMPLES], strict=True)]
        if scores is not None:
            try:
                decisions = _candidates(rows, scores, log_probabilities=log_probabilities)
                for sample, decision in zip(samples, decisions, strict=True):
                    sample["candidate"] = decision
            except _CAPTURE_ERRORS:
                self._warn()
        self.write("optimizer", "running", trial=trial, epoch=epoch, optimizer_step=step, steps=steps,
                   mean_training_loss=loss, batch_size=len(indices), samples_seen=seen, samples=samples, **speed)

    def capture_logp(self, _code, _offset):
        """Retain the actual loss input for sampled rows consumed one forward at a time."""
        if get_ident() != self.thread:
            return
        call = sys._getframe(1)
        trainer = call.f_back
        if trainer is None:
            return
        target = _function(trainer)
        if target not in (_NATIVE, _LORA) or target not in _TRAINERS[self.command]:
            return
        try:
            local = trainer.f_locals
            if target == _NATIVE:
                optimizer, index, first = local["opt"], local["j"], not local["z"]
            else:
                optimizer, index = local["optimizer"], local["index"]
                first = index == local["group"][0]
            if first:
                self.micro_scores[optimizer] = []
            captured = self.micro_scores[optimizer]
            if len(captured) < MAX_PREVIEW_SAMPLES:
                captured.append((index, call.f_locals["input"].detach()))
        except _CAPTURE_ERRORS:
            self._warn()

    def _micro_batch(self, optimizer, rows, indices, **fields):
        captured = self.micro_scores.pop(optimizer, ())
        if [i for i, _ in captured] == list(indices[:MAX_PREVIEW_SAMPLES]):
            scores = [value for _, value in captured]
        else:
            scores = None
            self._warn()
        self._publish_batch(optimizer, [rows[i] for i in indices[:MAX_PREVIEW_SAMPLES]], indices, scores,
                            log_probabilities=True, **fields)

    def _native(self, local):
        order, offset, previous = local["order"], local["i"], local["ll"]
        indices = order[offset:offset + len(local["z"])]
        size = offset // len(previous) if previous else len(indices)
        step, steps = len(previous) + 1, math.ceil(len(order) / size)
        # The native function has acquired its new output directory before any optimizer step.
        self.native_started = True
        total = sum(previous) * size + float(local["loss"].detach()) * len(indices)
        seen = offset + len(indices)
        self._micro_batch(local["opt"], local["tr"], indices, trial=local["args"].preset, epoch=local["ep"],
                          step=step, steps=steps, loss=total / seen, seen=seen)

    def _lora(self, local):
        step, steps = local["step"], local["steps"]
        group = local["group"]
        seen = local["offset"] + len(group)
        self._micro_batch(local["optimizer"], local["rows"], group, trial=local["label"], epoch=local["epoch"],
                          step=step, steps=steps, loss=float(local["running_loss"]) / seen, seen=seen)

    def _head(self, frame, local, optimizer, target):
        epoch, order = local["epoch"], local["order"]
        if target == _ROUND6:
            indices, offset, size = local["indices"], local["begin"], frame.f_globals["BATCH_SIZE"]
        elif target == _IMPROVE:
            indices, offset, size = local["batch_indices"], local["begin"], local["config"].batch_size
        else:
            indices, offset, size = local["indices"], local["start"], local["config"].batch_size
        step, steps = offset // size + 1, math.ceil(len(order) / size)
        # Keep loss detached from the training graph and weight a partial batch by its real size.
        total = local["loss"].detach() * len(indices)
        if offset:
            previous_epoch, previous_step, previous_total = self.head_losses[optimizer]
            if previous_epoch != epoch or previous_step != step - 1:
                raise ValueError("Missing previous optimizer loss")
            total = previous_total + total
        self.head_losses[optimizer] = (epoch, step, total)
        rows = local["batch_rows"] if target == _ROUND6 else [
            local["train_rows"][i] for i in indices[:MAX_PREVIEW_SAMPLES]]
        seen = offset + len(indices)
        self._publish_batch(optimizer, rows, indices, local["logits"], trial=_trial_name(frame), epoch=epoch,
                            step=step, steps=steps, loss=float(total) / seen, seen=seen)

    def observe(self, optimizer, _args, _kwargs):
        if get_ident() != self.thread:
            return
        # Older experiments verify their training-source file hashes during and after a run.
        # Read the executed optimizer inputs at the step boundary without replacing those files.
        frame = sys._getframe(1)
        while frame and _function(frame) not in _TRAINERS[self.command]:
            frame = frame.f_back
        if frame is None:
            return
        target = _function(frame)
        local = frame.f_locals
        optimizer_name = "opt" if target == _NATIVE else "optimizer"
        if local.get(optimizer_name) is not optimizer:
            return
        try:
            if target == _NATIVE:
                self._native(local)
            elif target == _LORA:
                self._lora(local)
            else:
                self._head(frame, local, optimizer, target)
        except _CAPTURE_ERRORS:
            self._warn()


@contextmanager
def _observe_sequential_loss(journal):
    if not _TRAINERS[journal.command].intersection((_NATIVE, _LORA)):
        yield
        return
    from torch.nn.functional import kl_div

    # Both frozen loops pass each row's computed log probabilities to this Python function.
    # A code-local PEP 669 event leaves the model, loss, optimizer and other code unmodified.
    monitor = sys.monitoring
    for tool in (3, 4, 1, 2, 0, 5):
        try:
            monitor.use_tool_id(tool, "kev2048 training samples")
            break
        except ValueError:  # owned by another debugger, profiler, or training thread
            continue
    else:
        journal._warn()
        yield
        return
    try:
        monitor.register_callback(tool, monitor.events.PY_START, journal.capture_logp)
        monitor.set_local_events(tool, kl_div.__code__, monitor.events.PY_START)
        yield
    finally:
        monitor.set_local_events(tool, kl_div.__code__, 0)
        monitor.register_callback(tool, monitor.events.PY_START, None)
        monitor.free_tool_id(tool)


@contextmanager
def watch_training(out, command, *, started=None):
    """Add an observer to the CLI process for the duration of its training command."""
    from torch.optim.optimizer import register_optimizer_step_post_hook

    current = _ACTIVE_JOURNAL.get()
    if (command == "standalone-head" and current and current.thread == get_ident()
            and current.out == Path(out).absolute() and _EXPERIMENT in _TRAINERS[current.command]):
        # A driver can also opt in explicitly. Its observer already owns this fit.
        yield
        return

    journal = _Journal(out, command, started=started)
    hook = register_optimizer_step_post_hook(journal.observe)
    token = _ACTIVE_JOURNAL.set(journal)
    try:
        with _observe_sequential_loss(journal):
            yield
    except KeyboardInterrupt:
        if journal.native_started:
            journal.write("stopped", "stopped")
        raise
    except BaseException:
        if journal.native_started:
            journal.write("error", "error")
        raise
    else:
        if journal.native_started:
            journal.write("completed", "completed")
    finally:
        _ACTIVE_JOURNAL.reset(token)
        hook.remove()

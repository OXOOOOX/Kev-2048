"""Index registered training journals by optimizer batch; load boards only on demand."""

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from threading import Lock

from kev2048_release.game import flatten, legal
from training_boards import MAX_PREVIEW_SAMPLES, SOURCES, STAGES, TRAINING_SCHEMA, _recorded_candidate, recorded_batch
from training_discovery import OPEN_FLAGS


MAX_EVENT_BYTES = 512 * 1024
MAX_CACHED_JOURNALS = 16
LABEL = re.compile(r"[\w. -]{1,100}")


def _int(value, minimum=0, maximum=100_000_000):
    return type(value) is int and minimum <= value <= maximum


def _number(value):
    return type(value) in (int, float) and -1e15 < value < 1e15 and math.isfinite(value)


def _json(raw):
    try:
        value = json.loads(raw)
    except (ValueError, OverflowError, RecursionError):
        return {}
    return value if isinstance(value, dict) else {}


def _trial(record):
    value = record.get("trial", record.get("arm"))
    return value if isinstance(value, str) and LABEL.fullmatch(value) else ""


def _timestamp(record):
    value = record.get("timestamp")
    if isinstance(value, str) and len(value) <= 40:
        try:
            if datetime.fromisoformat(value).tzinfo is not None:
                return value
        except ValueError:
            pass
    return None


def _key(trial, epoch, step):
    ident = json.dumps([trial, epoch, step], ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(ident).hexdigest()[:24]


def _epoch_metrics(record):
    epoch = record.get("epoch")
    if (not _int(epoch) or "optimizer_step" in record
            or record.get("phase") in ("completed", "trial_completed", "training_completed")):
        return None
    metrics = {}
    for output, keys in (
        ("train_loss", ("train_loss", "train_teacher_loss")),
        ("dev_loss", ("dev_loss", "dev_teacher_loss")),
        ("dev_objective", ("dev_objective",)),
        ("loss", ("loss",)),
        ("train_best_rate", ("train_best_rate", "train_hit")),
        ("dev_best_rate", ("dev_best_rate",)),
    ):
        for key in keys:
            value = record.get(key)
            if _number(value) and (not output.endswith("_rate") or 0 <= value <= 1):
                metrics[output] = value
                break
    if not metrics:
        return None
    if record.get("phase") == "head_fit" and "train_loss" in record and "loss" in metrics:
        metrics.setdefault("dev_loss", metrics["loss"])
    if _int(record.get("epochs"), 1):
        metrics["epochs"] = record["epochs"]
    if _int(record.get("best_epoch")):
        metrics["best_epoch"] = record["best_epoch"]
    return dict(trial=_trial(record), epoch=epoch, timestamp=_timestamp(record), **metrics)


def _step(record):
    phase = record.get("phase")
    if phase == "optimizer":
        if record.get("schema") != TRAINING_SCHEMA or record.get("kind") != "training":
            return None
    elif phase != "lora_fit":
        return None
    epoch, step, steps = (record.get(k) for k in ("epoch", "optimizer_step", "steps"))
    if not (_int(epoch, 1) and _int(steps, 1) and _int(step, 1, steps)):
        return None
    return dict(phase=phase, trial=_trial(record), epoch=epoch, optimizer_step=step,
                steps=steps, timestamp=_timestamp(record))


def _unlabeled_sample(sample):
    board, row = sample.get("board"), sample.get("row")
    if not _int(row, 1) or not isinstance(board, list) or len(board) != 4:
        return None
    if not all(isinstance(line, list) and len(line) == 4 and all(
            type(v) is int and 0 <= v <= 1_048_576 and (v == 0 or v >= 2 and not v & (v-1))
            for v in line) for line in board):
        return None
    out = dict(row=row, board=board, best_actions=[])
    for key, allowed in (("source_policy", SOURCES), ("stage", STAGES)):
        if sample.get(key) in allowed:
            out[key] = sample[key]
    candidate = _recorded_candidate(sample.get("candidate"), legal(flatten(board)))
    if candidate is not None:
        out["candidate"] = candidate
    return out


def _recorded_samples(record, step):
    complete = recorded_batch(record, step)
    if complete is not None:
        return complete["samples"]
    # Some journals capture fewer examples than their batch size, or omit the teacher.
    # Keep a real board without assigning it a made-up label.
    samples, size = record.get("samples"), record.get("batch_size")
    if not (_int(size, 1) and isinstance(samples, list) and 0 < len(samples) <= min(size, MAX_PREVIEW_SAMPLES)):
        return []
    rows, checked = set(), []
    for sample in samples:
        if not isinstance(sample, dict):
            return []
        if sample.get("best_actions") in (None, []):
            public = _unlabeled_sample(sample)
        else:
            one = recorded_batch({"samples": [sample], "batch_size": 1}, step)
            public = one["samples"][0] if one else None
        if public is None or public["row"] in rows:
            return []
        rows.add(public["row"])
        checked.append(public)
    return checked


@contextmanager
def _open_journal(watch):
    path = Path(watch.path)
    guard = getattr(watch, "path_guard", None)
    if guard is not None and not guard(path):
        raise ValueError("Training journal is no longer registered at this path")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("Expected a regular training journal")
    with os.fdopen(os.open(path, OPEN_FLAGS), "rb") as stream:
        if not os.path.samestat(info, os.fstat(stream.fileno())):
            raise ValueError("Training journal was replaced while opening")
        yield stream, info
        after = path.lstat()
        if not os.path.samestat(info, after) or (guard is not None and not guard(path)):
            raise ValueError("Training journal changed path while reading")
        if after.st_size < info.st_size or (after.st_size == info.st_size and after.st_mtime_ns != info.st_mtime_ns):
            raise ValueError("Training journal changed while reading")


@dataclass
class _Entry:
    offset: int
    length: int
    phase: str
    meta: dict


@dataclass
class _Index:
    stamp: tuple = ()
    offset: int = 0
    skipping: bool = False
    anchor: bytes = b""
    entries: dict = field(default_factory=dict)
    epochs: dict = field(default_factory=dict)


class TrainingArchive:
    """Keep compact, append-aware indexes. A batch request rereads one journal line."""

    def __init__(self, boards=None):
        self.boards = boards
        self.cache = OrderedDict()
        self.lock = Lock()

    def _samples(self, path, record, step):
        if step["phase"] == "lora_fit" and "samples" not in record:
            resolved = self.boards.batch(path.parent, step) if self.boards is not None else None
            return (resolved["samples"], "reconstructed") if resolved else ([], None)
        samples = _recorded_samples(record, step)
        for sample in samples:
            if "candidate" in sample:
                sample["candidate_source"] = "recorded"
        return samples, "recorded" if samples else None

    def _add(self, index, path, line, offset):
        if b'"epoch"' not in line:
            return
        record = _json(line)
        step = _step(record)
        if step:
            trial, epoch, number = step["trial"], step["epoch"], step["optimizer_step"]
            key = _key(trial, epoch, number)
            previous = index.entries.get(key)
            if previous and previous.phase == "optimizer" and step["phase"] == "lora_fit":
                return  # The same update may also emit the legacy progress-only line.
            samples, source = self._samples(path, record, step)
            size, loss = record.get("batch_size"), record.get("mean_training_loss")
            meta = dict(key=key, trial=trial, epoch=epoch, step=number, steps=step["steps"],
                        size=size if _int(size, 1) else len(samples) or None, sample_count=len(samples),
                        loss=loss if _number(loss) else None, timestamp=step["timestamp"], board_source=source,
                        revision=hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()[:24])
            rate = record.get("boards_per_second")
            if _number(rate) and rate > 0:
                meta["boards_per_second"] = rate
            index.entries[key] = _Entry(offset, len(line), step["phase"], meta)
        else:
            epoch = _epoch_metrics(record)
            if epoch is not None:
                key = epoch["trial"], epoch["epoch"]
                index.epochs.setdefault(key, {}).update(epoch)

    def _read_index(self, watch, stream, info):
        path = Path(watch.path)
        stamp = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        index = self.cache.get(path)
        if index is not None:
            self.cache.move_to_end(path)
            if index.stamp == stamp:
                return index
            if index.stamp[:2] != stamp[:2] or info.st_size <= index.stamp[2]:
                index = None
            elif index.anchor:
                stream.seek(index.offset-len(index.anchor))
                if stream.read(len(index.anchor)) != index.anchor:
                    index = None
        if index is None:
            index = _Index()
        stream.seek(index.offset)
        while index.offset < info.st_size:
            start = index.offset
            line = stream.readline(min(MAX_EVENT_BYTES+1, info.st_size-start))
            if not line:
                raise ValueError("Training journal was truncated")
            complete = line.endswith(b"\n")
            if not complete and not index.skipping and len(line) <= MAX_EVENT_BYTES:
                break
            if not index.skipping and complete and len(line) <= MAX_EVENT_BYTES:
                self._add(index, path, line, start)
            index.skipping = not complete
            index.offset += len(line)
        index.stamp = stamp
        stream.seek(max(0, index.offset-256))
        index.anchor = stream.read(min(index.offset, 256))
        self.cache[path] = index
        if len(self.cache) > MAX_CACHED_JOURNALS:
            self.cache.popitem(last=False)
        return index

    def index(self, watch):
        if Path(watch.path).suffix.lower() not in (".jsonl", ".ndjson"):
            return dict(batches=[], epochs=[], revision="")
        with self.lock, _open_journal(watch) as (stream, info):
            idx = self._read_index(watch, stream, info)
            epochs = {key: value.copy() for key, value in idx.epochs.items()}
            batches = {key: entry.meta.copy() for key, entry in idx.entries.items()}
            return dict(batches=list(batches.values()), epochs=[epochs[k] for k in sorted(epochs)],
                        revision="-".join(map(str, idx.stamp)))

    def batch(self, watch, key):
        if not isinstance(key, str) or not re.fullmatch(r"[0-9a-f]{24}", key):
            return None
        if Path(watch.path).suffix.lower() not in (".jsonl", ".ndjson"):
            return None
        with self.lock, _open_journal(watch) as (stream, info):
            idx = self._read_index(watch, stream, info)
            entry = idx.entries.get(key)
            if entry is None:
                return None
            stream.seek(entry.offset)
            line = stream.read(entry.length)
            if hashlib.sha256(line.rstrip(b"\r\n")).hexdigest()[:24] != entry.meta["revision"]:
                self.cache.pop(Path(watch.path), None)
                raise ValueError("Selected training record changed while reading")
            record = _json(line)
            samples, source = self._samples(Path(watch.path), record, _step(record))
            return dict(entry.meta, sample_count=len(samples), samples=samples, board_source=source)

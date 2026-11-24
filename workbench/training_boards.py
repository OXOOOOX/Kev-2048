"""Read recorded training batches and resolve boards from older LoRA journals."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import random
from threading import Lock

from kev2048_release.game import DIRS, flatten, legal, nested
from kev2048_release.artifacts import identity
from kev2048_release.training_progress import MAX_PREVIEW_SAMPLES, SCHEMA as TRAINING_SCHEMA

from training_discovery import MAX_CONFIG_BYTES, _plain_directory, _plain_file, _prefix


# This implementation records per-epoch optimizer steps against a fixed shuffle.
LORA_IMPLEMENTATION = "dd9925c80d9d75b401d1032bfb7f2a70bc0c82a7d5630d4529f0d8ae584bfde1"
SEED = 96_729_100
ROWS = 2_000
ACCUMULATION = 16
PILOT_STEPS = 32
MAX_DATA_BYTES = 8 * 1024 * 1024
MAX_CACHED_RUNS = 8
SUBSET = "train-2000.jsonl"
TRIAL_STEPS = {"lora": ROWS // ACCUMULATION, "lora-head-control": ROWS // ACCUMULATION,
               "lora-pilot": PILOT_STEPS, "lora-control-timing": 4}
STAGES = ("early", "middle", "late")
SOURCES = ("teacher", "official", "stock", "perturbed")


def _recorded_candidate(value, moves):
    if not isinstance(value, dict) or value.get("choice") not in moves:
        return None
    probabilities = value.get("probabilities")
    if not isinstance(probabilities, dict) or set(probabilities) != set(DIRS):
        return None
    for move, probability in probabilities.items():
        if move in moves:
            if type(probability) not in (int, float) or not 0 <= probability <= 1:
                return None
        elif probability is not None:
            return None
    legal_values = [probabilities[move] for move in moves]
    if not math.isclose(sum(legal_values), 1, abs_tol=1e-5) or probabilities[value["choice"]] != max(legal_values):
        return None
    return dict(choice=value["choice"], probabilities={move: probabilities[move] for move in DIRS})


def recorded_batch(record, step):
    """Validate the samples an optimizer event includes before exposing them to the page."""
    size, samples = record.get("batch_size"), record.get("samples")
    epoch = step.get("epoch")
    if (type(epoch) is not int or epoch < 1 or type(size) is not int or not 1 <= size <= 100_000_000
            or not isinstance(samples, list) or len(samples) != min(size, MAX_PREVIEW_SAMPLES)):
        return None
    public_samples = []
    seen = set()
    for sample in samples:
        if not isinstance(sample, dict):
            return None
        index, board, best = sample.get("row"), sample.get("board"), sample.get("best_actions")
        if (type(index) is not int or not 1 <= index <= 100_000_000 or index in seen
                or not isinstance(board, list) or len(board) != 4
                or not all(isinstance(line, list) and len(line) == 4 for line in board)
                or not all(type(value) is int and 0 <= value <= 1_048_576
                           and (value == 0 or value >= 2 and not value & (value - 1))
                           for line in board for value in line)):
            return None
        moves = legal(flatten(board))
        if (not isinstance(best, list) or not 1 <= len(best) <= 4
                or not all(isinstance(action, str) and action in moves for action in best)
                or len(set(best)) != len(best)):
            return None
        seen.add(index)
        fields = dict(row=index, board=board, best_actions=best)
        candidate = _recorded_candidate(sample.get("candidate"), moves)
        if candidate is not None:
            fields["candidate"] = candidate
        for key, allowed in (("stage", STAGES), ("source_policy", SOURCES)):
            if sample.get(key) in allowed:
                fields[key] = sample[key]
        public_samples.append(fields)
    batch = dict(phase="optimizer", trial=step.get("trial"), epoch=epoch, optimizer_step=step["optimizer_step"],
                 steps=step["steps"], batch_size=size, samples=public_samples)
    if step.get("timestamp"):
        batch["timestamp"] = step["timestamp"]
    seen_count, rate = record.get("samples_seen"), record.get("boards_per_second")
    if type(seen_count) is int and seen_count >= size:
        batch["samples_seen"] = seen_count
    if type(rate) in (int, float) and 0 < rate <= 1e12:
        batch["boards_per_second"] = rate
    return batch


def _stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _read(path, info, limit):
    if info.st_size > limit:
        raise ValueError("Training input exceeds read limit")
    raw = _prefix(path, info, limit + 1)
    if len(raw) != info.st_size or _stamp(_plain_file(path)) != _stamp(info):
        raise ValueError("Training input changed while reading")
    return raw


def _resolved(path, info):
    report = json.loads(_read(path, info, MAX_CONFIG_BYTES))
    if not isinstance(report, dict):
        raise ValueError("Expected training run metadata")
    claimed = report.pop("identity", None)
    if identity(report) != claimed:
        raise ValueError("Unverified training run identity")
    return report, claimed


@dataclass(frozen=True)
class _Sample:
    board: tuple
    stage: str
    source_policy: str
    best_actions: tuple

    @classmethod
    def from_row(cls, row):
        board = flatten(row["board"])
        if any(type(value) is not int or value < 0 or value > 1_048_576
               or value and (value < 2 or value & (value - 1)) for value in board):
            raise ValueError("Invalid training board")
        moves = legal(board)
        best = row["best_actions"]
        if (row.get("split") != "train" or row["stage"] not in STAGES or row["source_policy"] not in SOURCES
                or row["valid_moves"] != moves or len(moves) < 2 or not isinstance(best, list)
                or not best or not all(action in moves for action in best) or len(set(best)) != len(best)):
            raise ValueError("Invalid training label")
        return cls(board, row["stage"], row["source_policy"], tuple(best))

    def payload(self, index):
        return dict(row=index + 1, board=nested(self.board), stage=self.stage,
                    source_policy=self.source_policy, best_actions=list(self.best_actions))


@dataclass(frozen=True)
class _Dataset:
    metadata_stamp: tuple
    path: Path
    data_stamp: tuple
    samples: tuple


@lru_cache(maxsize=2)
def _epoch_order(epoch):
    order = list(range(ROWS))
    random.Random(SEED + epoch).shuffle(order)
    return tuple(order)


class TrainingBoards:
    """Reconstruct an emitted batch from frozen files within the runs directory."""

    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        self._lock = Lock()
        self._cache = OrderedDict()

    def _input_info(self, path):
        if not path.is_relative_to(self.root) or ".." in path.parts:
            raise ValueError("Training input lies outside the runs directory")
        for parent in path.parents:
            if not _plain_directory(parent):
                raise ValueError("Training input must use ordinary directories")
            if parent == self.root:
                return _plain_file(path)
        raise ValueError("Expected an input within the runs directory")

    def _samples(self, run):
        parent_path, lora_path = run / "resolved-config.json", run / "lora" / "resolved-config.json"
        parent_info, lora_info = self._input_info(parent_path), self._input_info(lora_path)
        metadata_stamp = _stamp(parent_info), _stamp(lora_info)
        cached = self._cache.get(run)
        if cached and cached.metadata_stamp == metadata_stamp:
            if _stamp(self._input_info(cached.path)) == cached.data_stamp:
                self._cache.move_to_end(run)
                return cached.samples

        parent, parent_id = _resolved(parent_path, parent_info)
        branch, _ = _resolved(lora_path, lora_info)
        if (branch.get("parent_identity") != parent_id or branch.get("implementation_sha256") != LORA_IMPLEMENTATION
                or any(type(branch.get(key)) is not int or branch[key] != value for key, value in
                       (("seed", SEED), ("accumulation", ACCUMULATION), ("microbatch", 1), ("pilot_steps", PILOT_STEPS)))):
            raise ValueError("Unrecognized LoRA batch schedule")

        path = Path(parent["config"]["data"]) / SUBSET
        info = self._input_info(path)
        raw = _read(path, info, MAX_DATA_BYTES)
        if hashlib.sha256(raw).hexdigest() != parent["data_files"][SUBSET]:
            raise ValueError("Training subset file checksum mismatch")
        rows = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(rows) != ROWS or identity(rows) != branch.get("train_sha256"):
            raise ValueError("Training subset order or contents changed")
        samples = tuple(_Sample.from_row(row) for row in rows)
        self._cache[run] = _Dataset(metadata_stamp, path, _stamp(info), samples)
        self._cache.move_to_end(run)
        if len(self._cache) > MAX_CACHED_RUNS:
            self._cache.popitem(last=False)
        return samples

    def batch(self, run_directory, event):
        """Return the batch at the logged optimizer step, or None without verified input."""
        if not isinstance(event, dict) or event.get("phase") != "lora_fit":
            return None
        trial = event.get("trial")
        if not isinstance(trial, str) or trial not in TRIAL_STEPS:
            return None
        epoch, step, steps = (event.get(key) for key in ("epoch", "optimizer_step", "steps"))
        if (any(type(value) is not int for value in (epoch, step, steps)) or steps != TRIAL_STEPS[trial]
                or not 1 <= step <= steps or epoch not in (1, 2)
                or trial in ("lora-pilot", "lora-control-timing") and epoch != 1):
            return None
        try:
            run = Path(os.path.abspath(run_directory))
            if run.parent != self.root:
                return None
            with self._lock:
                samples = self._samples(run)
        except (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError):
            return None
        start = (step - 1) * ACCUMULATION
        indices = _epoch_order(epoch)[start:start + ACCUMULATION]
        return dict(trial=trial, epoch=epoch, optimizer_step=step, steps=steps,
                    samples=[samples[index].payload(index) for index in indices])

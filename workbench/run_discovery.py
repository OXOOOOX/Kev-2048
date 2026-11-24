"""Find the journals and aggregate results that drive the local workbench."""

from datetime import datetime
from functools import lru_cache
import hashlib
import json
import os
import re
import stat

from training_discovery import (
    PROGRESS_NAME, REPARSE_POINT, SCAN_INTERVAL, SIGNATURE_BYTES, TRAINING_PHASES, WRAPPER_SUFFIX,
    TrainingDiscovery, _config_kind, _display_name, _plain_directory, _plain_file, _prefix,
)


RESULT_NAMES = ("benchmark.json", "comparison.json", "preparation-report.json", "result.json", "status.json")
PRIMARY_FILES = (PROGRESS_NAME, *RESULT_NAMES)
MAX_RESULT_BYTES = 4 * 1024 * 1024
KINDS = {"training", "evaluation", "benchmark", "inference", "gameplay"}
STAGE_NAMES = {"training": "训练", "fit": "训练", "gameplay": "整局测评", "evaluation": "测评",
               "benchmark": "测速", "inference": "推理", "data": "数据采集", "training-data": "数据采集"}
TRIAL_GROUPS = {"arms", "trials", "stages", "lora"}
# Do not enumerate model artifacts, generated datasets, or individual seed directories.
DATA_DIRECTORIES = {"episodes", "seeds", "labels", "traces", "checkpoints", "checkpoint", "features",
                    "cache", "dataset", "datasets", "artifacts", "implementation", "instant-play"}
INVALID_INPUT = (OSError, ValueError, OverflowError, RuntimeError)


def _stage_directory(name):
    return (not name.startswith(".") and not name.isdecimal() and name.casefold() not in DATA_DIRECTORIES
            and not name.casefold().startswith("dataset-"))


def _run_directory(name):
    return not name.startswith(".") and name.casefold() != "instant-play"


def _date_group(name):
    return re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", name) is not None


def _stage_paths(directory, accept=_stage_directory):
    with os.scandir(directory) as entries:
        for entry in entries:
            if accept(entry.name) and entry.is_dir(follow_symlinks=False):
                yield directory / entry.name


@lru_cache(maxsize=2048)
def _candidate_files(directory, device, inode, modified):
    # Probe only known names even if a model directory holds thousands of individual games.
    # Directory write time changes when a new output is created or a file is replaced.
    names = []
    for name in (*PRIMARY_FILES, "resolved-config.json"):
        try:
            if stat.S_ISREG((directory/name).lstat().st_mode):
                names.append(name)
        except OSError:
            pass
    return tuple(names)


def _journal_kind(event, *, wrapper=False):
    """Recognize the declared activity, or the unit of work in legacy journals."""
    declared = event.get("kind")
    if isinstance(declared, str):
        return declared if declared in KINDS else None
    phase = event.get("phase")
    if not isinstance(phase, str):
        return None
    if wrapper and phase == "starting" and event.get("status") == "loading":
        return "training"
    if phase in ("train_cache", "dev_cache") and type(event.get("step")) is int and type(event.get("total")) is int:
        return "training"
    words = set(phase.split("_"))
    if ("collection" in words or "complete_games" in event and "until" in event
            or phase in ("training_episode_completed", "training_data_completed", "training_loading_native")):
        return "inference"
    if "benchmark" in words or "timing" in words and "repeats" in event:
        return "benchmark"
    if ("gameplay" in words or "screening" in words or "evaluation" in words
            or phase == "episode_completed" and "seed" in event):
        return "evaluation"
    if "inference" in words and phase != "inference_gate":
        return "inference"
    if (phase in TRAINING_PHASES or "fit" in words or "optimizer" in words
            or "epoch" in words and ("loss" in event or "train_loss" in event)
            or type(event.get("train_rows")) is int and type(event.get("dev_rows")) is int):
        return "training"
    return None


def _result_kind(item, name):
    # An individual game can have an explicit kind and completion status too.
    if isinstance(item.get("seed"), int) and ("score" in item or "max_tile" in item):
        return None
    if isinstance(item.get("kind"), str) and item["kind"] in KINDS:
        return item["kind"]
    if name == "status.json":
        return _journal_kind(item)
    if (isinstance(item.get("runs"), list) and ("median_wall_seconds" in item or
            {"preset", "repeats", "completed"} <= item.keys()) or isinstance(item.get("runs"), dict)
            and {"process_speedup", "checkpoints_identical"} <= item.keys()):
        return "benchmark"
    if {"optimizer", "control_timing", "wall_seconds", "probe_indices"} <= item.keys():
        return "benchmark"
    if "collection_identity" in item or {"total_collected_episodes", "teacher_labels_rechecked"} <= item.keys():
        return "inference"
    if (type(item.get("train_rows")) is int and type(item.get("dev_rows")) is int
            and ("completed_epochs" in item or "trials" in item)):
        return "training"
    if (type(item.get("completed_epochs")) is int and type(item.get("best_epoch")) is int
            and ("optimizer_steps" in item or "checkpoint_files" in item and "trial" in item)):
        return "training"
    if ({"candidate_id", "reference_id", "total_games"} <= item.keys()
            or {"results", "episode_sha256"} <= item.keys() or "candidate_episode_sha256" in item
            or isinstance(item.get("games"), list) and "checkpoint_head_sha256" in item and "cap" in item):
        return "evaluation"
    return None


@lru_cache(maxsize=2048)
def _classified(path, device, inode, size, modified):
    """Read a new version of a file once. Its inode and write time are part of the cache key."""
    info = _plain_file(path)
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (device, inode, size, modified):
        return None
    if path.name == PROGRESS_NAME or path.name.endswith(WRAPPER_SUFFIX):
        lines = _prefix(path, info, SIGNATURE_BYTES).splitlines(keepends=True)
        starting = None
        for line in lines:
            if not line.endswith(b"\n"):
                break
            try:
                event = json.loads(line.decode("utf-8-sig"))
                if not isinstance(event, dict):
                    continue
                stamp = event.get("timestamp")
                if not isinstance(stamp, str) or len(stamp) > 40 or datetime.fromisoformat(stamp).tzinfo is None:
                    continue
                kind = _journal_kind(event, wrapper=path.name.endswith(WRAPPER_SUFFIX))
                if kind:
                    return kind
                if (event.get("kind") is None and event.get("phase") == "starting" and event.get("status") == "loading"
                        and type(event.get("elapsed_seconds")) in (int, float)):
                    starting = "training"
            except (ValueError, OverflowError, RecursionError):
                continue
        return starting
    config = path.name == "resolved-config.json"
    limit = MAX_RESULT_BYTES
    if size > limit:
        return None
    item = json.loads(_prefix(path, info, limit + 1).decode("utf-8-sig"))
    if not isinstance(item, dict):
        return None
    if config:
        if (item.get("schema_version") == 1 and isinstance(item.get("seeds"), list)
                and isinstance(item.get("checkpoints"), dict) and isinstance(item.get("config"), dict)):
            return "gameplay"
        return _config_kind(path)
    if (path.name == "result.json" and path.parent.name.casefold() == "artifacts"
            and isinstance(item.get("candidate_head_sha256"), str) and re.fullmatch(r"[0-9a-f]{64}", item["candidate_head_sha256"])
            and all(type(item.get(key)) is int and item[key] > 0 for key in ("train", "dev"))
            and isinstance(item.get("metrics"), dict) and isinstance(item.get("timing"), dict)):
        return "training"
    return _result_kind(item, path.name)


def _kind(path, info):
    try:
        return _classified(path, info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)
    except INVALID_INPUT:
        return None


class RunDiscovery(TrainingDiscovery):
    """Return (stable key, name, kind, path) for both existing and newly created activities."""

    def __init__(self, root, excluded=(), *, interval=SCAN_INTERVAL):
        super().__init__(root, excluded, interval=interval)
        self._excluded_stages = {p.parent for p in self.excluded if p.name in PRIMARY_FILES}
        self._excluded_stages.update(p.parent.parent for p in self.excluded
                                     if p.name == "result.json" and p.parent.name.casefold() == "artifacts")

    def _progress_stat(self, path):
        relative = path.relative_to(self.root)
        if path in self.excluded or not _plain_directory(self.root) or self.root.resolve(strict=True) != self.root:
            raise ValueError("Not a discoverable run")
        if len(relative.parts) == 1:
            if path.name.endswith(WRAPPER_SUFFIX):
                return _plain_file(path)
            raise ValueError("Not a run log")
        parts = relative.parts
        relative_run = parts[1:] if len(parts) >= 3 and _date_group(parts[0]) else parts
        artifact = len(relative_run) == 3 and relative_run[1].casefold() == "artifacts" and path.name == "result.json"
        if (not _run_directory(relative_run[0]) or path.name not in PRIMARY_FILES or not artifact and
                (not all(_stage_directory(c) for c in relative_run[1:-1]) or len(relative_run) > 4
                 or len(relative_run) == 4 and relative_run[1].casefold() not in TRIAL_GROUPS)):
            raise ValueError("Not a run log or aggregate result")
        parent = self.root
        for component in parts[:-1]:
            parent /= component
            if not _plain_directory(parent):
                raise ValueError("Not a run directory")
        return _plain_file(path)

    def _activity(self, directory):
        # A configured watch for any primary output of this stage already owns it.
        if directory in self.excluded or directory in self._excluded_stages:
            return None
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or getattr(info, "st_file_attributes", 0) & REPARSE_POINT:
            return None
        names = _candidate_files(directory, info.st_dev, info.st_ino, info.st_mtime_ns)
        config_kind = None
        if "resolved-config.json" in names:
            config = directory / "resolved-config.json"
            try:
                config_kind = _kind(config, _plain_file(config))
            except INVALID_INPUT:
                pass
        for name in names:
            if name not in PRIMARY_FILES:
                continue
            path = directory / name
            try:
                info = _plain_file(path)
            except INVALID_INPUT:
                continue
            kind = _kind(path, info)
            if name == PROGRESS_NAME:
                if config_kind == "gameplay" and kind in (None, "gameplay", "evaluation"):
                    kind = "gameplay"
                elif not kind and config_kind in ("experiment", "improve"):
                    kind = "training"
            if kind:
                if kind == "gameplay" and (name != PROGRESS_NAME or config_kind != "gameplay"):
                    kind = "evaluation"
                return (path, info.st_mtime_ns, kind, config_kind)
        return None

    def _run_activities(self, path):
        own = self._activity(path)
        if own:
            yield own
        # The CLI training wrapper stores its final result in a sibling output directory.
        # Read just this known file; checkpoints and other artifacts are never enumerated.
        excluded = path in self._excluded_stages or path in self.excluded
        if not excluded and (not own or own[2] != "training"):
            result = path / "artifacts" / "result.json"
            try:
                info = self._progress_stat(result)
                if _kind(result, info) == "training":
                    yield result, info.st_mtime_ns, "training", None
            except INVALID_INPUT:
                pass
        for stage in _stage_paths(path):
            activity = self._activity(stage)
            if activity:
                # A single training/result.json can mirror the top-level fit journal.
                if (own and own[2] == "training" and activity[2] == "training"
                        and activity[0].name == "result.json" and stage.name == "training"):
                    continue
                yield activity
            if stage.name.casefold() in TRIAL_GROUPS and _plain_directory(stage):
                for trial in _stage_paths(stage):
                    activity = self._activity(trial)
                    if activity:
                        yield activity

    def _scan(self):
        if not _plain_directory(self.root) or self.root.resolve(strict=True) != self.root:
            return ()
        primary = set()
        records = []
        with os.scandir(self.root) as stream:
            for entry in stream:
                path = self.root / entry.name
                try:
                    if entry.is_dir(follow_symlinks=False) and _run_directory(entry.name) and _plain_directory(path):
                        if _date_group(entry.name):
                            for run in _stage_paths(path, _run_directory):
                                if _plain_directory(run):
                                    records.extend(self._run_activities(run))
                        else:
                            for activity in self._run_activities(path):
                                records.append(activity)
                                if activity[0].parent == path or activity[0] == path/"artifacts"/"result.json":
                                    primary.add((entry.name, activity[2]))
                            if path in self._excluded_stages or path in self.excluded:
                                primary.add((entry.name, None))
                    elif entry.name.endswith(WRAPPER_SUFFIX):
                        info = self._progress_stat(path)
                        kind = _kind(path, info)
                        if kind:
                            records.append((path, info.st_mtime_ns, kind, None))
                except INVALID_INPUT:
                    continue
        sources = []
        for path, _, kind, config_kind in sorted(records, key=lambda row: (row[1], row[0]), reverse=True):
            relative = path.relative_to(self.root)
            if path.name.endswith(WRAPPER_SUFFIX):
                name = path.name[:-len(WRAPPER_SUFFIX)]
                if (name, kind) in primary or (name, None) in primary:
                    continue
                identity = relative
            else:
                owning = relative.parent
                if path.name == "result.json" and owning.name.casefold() == "artifacts":
                    owning = owning.parent
                name = owning.parts[0]
                for component in owning.parts[1:]:
                    name += " " + STAGE_NAMES.get(component, component)
                identity = owning / PROGRESS_NAME
            name = _display_name(name, config_kind)
            id_kind = "evaluation" if kind == "gameplay" else kind
            key = os.path.normcase(str(identity))
            source_id = id_kind + "-" + hashlib.sha256(os.fsencode(key)).hexdigest()[:16]
            sources.append((source_id, name, kind, path.parent if kind == "gameplay" else path))
        return tuple(sources)

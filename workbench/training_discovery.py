"""Discover local training journals in the standard run directory."""

from datetime import datetime
import hashlib
from itertools import islice
import json
import os
from pathlib import Path
import re
import stat
from threading import Lock
import time


SCAN_INTERVAL = 2.0
MAX_RUN_ENTRIES = 512
MAX_TRAINING_SOURCES = 10
MAX_CONFIG_BYTES = 64 * 1024
# A first optimizer event can contain 64 full boards and their four-way predictions.
SIGNATURE_BYTES = 64 * 1024
PROGRESS_NAME = "progress.jsonl"
WRAPPER_SUFFIX = ".progress.jsonl"
TRAINING_PHASES = frozenset(("verify_data", "train_encoding", "head_fit", "fit_epoch", "lora_fit",
                             "prepare_order_features", "round6_epoch", "scale_inputs", "round7_fit_epoch",
                             "retest_head_fit", "retest_epoch", "head_fit_epoch"))
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


def _plain_directory(path):
    info = path.lstat()
    return stat.S_ISDIR(info.st_mode) and not getattr(info, "st_file_attributes", 0) & REPARSE_POINT


def _plain_file(path):
    info = path.lstat()
    if (not stat.S_ISREG(info.st_mode) or getattr(info, "st_file_attributes", 0) & REPARSE_POINT
            or info.st_nlink != 1 or path.resolve(strict=True) != path):
        raise ValueError("Training progress must use a regular file within the runs directory")
    return info


def _prefix(path, info, limit):
    with os.fdopen(os.open(path, OPEN_FLAGS), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or not os.path.samestat(info, opened)
                or path.resolve(strict=True) != path):
            raise ValueError("Training journal changed while opening")
        return stream.read(limit)


def _config_kind(path):
    try:
        info = _plain_file(path)
        if info.st_size > MAX_CONFIG_BYTES:
            return None
        raw = _prefix(path, info, MAX_CONFIG_BYTES + 1)
        if len(raw) > MAX_CONFIG_BYTES:
            return None
        resolved = json.loads(raw.decode("utf-8-sig"))
    except (OSError, ValueError, RecursionError, RuntimeError):
        return None
    if not isinstance(resolved, dict) or not isinstance(config := resolved.get("config"), dict):
        return None
    if (resolved.get("training_scope") == "pointer_head_only" and isinstance(config.get("train_file"), str)
            and isinstance(config.get("trials"), list) and config["trials"]):
        return "experiment"
    if (all(isinstance(config.get(key), str) for key in ("head_reference", "cache_run"))
            and type(config.get("lora_planning_seconds")) is int
            and isinstance(diagnostic := resolved.get("diagnostic_indices_sha256"), str)
            and re.fullmatch(r"[0-9a-f]{64}", diagnostic)):
        return "improve"
    return None


def _journal_kind(path, info, *, wrapper):
    try:
        lines = _prefix(path, info, SIGNATURE_BYTES).splitlines(keepends=True)
    except (OSError, ValueError, RuntimeError):
        return None
    for index, line in enumerate(lines[:12]):
        if not line.endswith(b"\n"):
            break
        try:
            event = json.loads(line.decode("utf-8-sig"))
            if not isinstance(event, dict) or not isinstance(stamp := event.get("timestamp"), str):
                continue
            if len(stamp) > 40 or datetime.fromisoformat(stamp).tzinfo is None:
                continue
        except (ValueError, OverflowError, RecursionError):
            continue
        phase = event.get("phase")
        if event.get("kind") == "training":
            return "training"
        if event.get("kind") is not None:
            continue
        if wrapper:
            if index == 0 and phase == "starting" and event.get("status") == "loading":
                return "native"
            continue
        if (phase == "inputs_verified" and type(event.get("train_rows")) is int
                and type(event.get("dev_rows")) is int):
            return "improve"
        if isinstance(phase, str) and phase in TRAINING_PHASES:
            return "improve" if phase in ("lora_fit", "fit_epoch") else "experiment"
    return None


def _display_name(name, kind):
    if kind == "improve" and re.fullmatch(r"round3-improve(?:-\d{8})?", name):
        return "第三轮 LoRA 训练"
    return re.sub(r"[^\w -]+", " ", name).strip()[:80] or "自动发现的训练"


class TrainingDiscovery:
    """Return stable ID, display name and absolute journal path for recent runs."""

    def __init__(self, root, excluded=(), *, interval=SCAN_INTERVAL):
        self.root = Path(os.path.abspath(root))
        self.interval = interval
        self.excluded = set()
        for value in excluded:
            path = Path(os.path.abspath(value))
            self.excluded.add(path)
            try:
                self.excluded.add(path.resolve())
            except (OSError, RuntimeError):
                pass
        self._lock = Lock()
        self._scanned = float("-inf")
        self._sources = ()

    def _progress_stat(self, path):
        relative = path.relative_to(self.root)
        if path in self.excluded or not _plain_directory(self.root):
            raise ValueError("Training journal is outside the automatic discovery scope")
        if len(relative.parts) == 2 and relative.name == PROGRESS_NAME and _plain_directory(path.parent):
            return _plain_file(path)
        if len(relative.parts) == 1 and relative.name.endswith(WRAPPER_SUFFIX):
            return _plain_file(path)
        raise ValueError("Training journal is outside the automatic discovery scope")

    def safe_progress(self, path):
        """Recheck the boundary before a previously discovered file is sampled."""
        try:
            self._progress_stat(Path(os.path.abspath(path)))
        except (OSError, ValueError, RuntimeError):
            return False
        return True

    def _scan(self):
        if not _plain_directory(self.root) or self.root.resolve(strict=True) != self.root:
            return ()
        with os.scandir(self.root) as stream:
            entries = list(islice(stream, MAX_RUN_ENTRIES))
        candidates = []
        for entry in entries:
            try:
                directory = entry.is_dir(follow_symlinks=False)
                if directory:
                    path = self.root / entry.name / PROGRESS_NAME
                elif entry.name.endswith(WRAPPER_SUFFIX):
                    path = self.root / entry.name
                else:
                    continue
                info = self._progress_stat(path)
            except (OSError, ValueError, RuntimeError):
                continue
            candidates.append((info.st_mtime_ns, path, info, directory))
        run_journals = {path.parent.name: (path, info) for _, path, info, directory in candidates if directory}
        sources = []
        for _, path, info, directory in sorted(candidates, key=lambda entry: (entry[0], entry[1]), reverse=True):
            name = path.parent.name if directory else path.name[:-len(WRAPPER_SUFFIX)]
            # Prefer the trainer's own journal once a native cache wrapper's run is emitting one.
            if not directory and name in run_journals:
                run_path, run_info = run_journals[name]
                if _journal_kind(run_path, run_info, wrapper=False):
                    continue
            kind = _config_kind(path.parent / "resolved-config.json") if directory else None
            kind = kind or _journal_kind(path, info, wrapper=not directory)
            if not kind:
                continue
            key = os.path.normcase(str(path.relative_to(self.root)))
            source_id = "training-" + hashlib.sha256(os.fsencode(key)).hexdigest()[:16]
            sources.append((source_id, _display_name(name, kind), path))
            if len(sources) == MAX_TRAINING_SOURCES:
                break
        return tuple(sources)

    def sources(self):
        with self._lock:
            now = time.monotonic()
            if now - self._scanned >= self.interval:
                try:
                    self._sources = self._scan()
                except (OSError, ValueError, RuntimeError):
                    self._sources = ()
                self._scanned = now
            return self._sources

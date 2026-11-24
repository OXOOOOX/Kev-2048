"""List finished, locally recorded evaluation games for the replay directory."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from threading import Lock
import time

from gameplay_replay import MAX_LINE_BYTES, MAX_PAGE, MAX_TRACE_BYTES, file_stamp, valid_board
from gameplay_reuse import ARM, SHA, _final_move
from replay_library import MILESTONE_TILES
from training_discovery import _plain_directory, _plain_file, _prefix


INVALID = (OSError, ValueError, TypeError, KeyError, RuntimeError, OverflowError)


def _modified(stamp):
    return datetime.fromtimestamp(stamp / 1e9, timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _read(path, stamp, limit):
    info = _plain_file(path)
    if file_stamp(info) != stamp or not 0 < info.st_size <= limit:
        raise ValueError("Invalid evaluation file size or version")
    raw = _prefix(path, info, info.st_size)
    if len(raw) != info.st_size or file_stamp(_plain_file(path)) != stamp:
        raise ValueError("Evaluation file changed during indexing")
    return raw


def _completion_times(path, stamp):
    times = {}
    if stamp is None:
        return times
    try:
        raw = _read(path, stamp, 16 * 1024 * 1024)
    except INVALID:
        return times
    for line in raw.splitlines(keepends=True):
        if b'episode_completed' not in line or not line.endswith(b"\n"):
            continue
        try:
            e = json.loads(line)
            if e["phase"] != "episode_completed":
                continue
            when = datetime.fromisoformat(e["timestamp"])
            if when.tzinfo is None:
                continue
            key = (e["arm"], e["seed"], e["turns"], e["score"], e["max_tile"], e["stop"])
            times.setdefault(key, when.astimezone(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z"))
        except INVALID:
            continue
    return times


def _milestones(raw, seed, maximum):
    targets = [n for n in MILESTONE_TILES if n <= maximum]
    reached = {}
    if targets:
        for turn, line in enumerate(raw.splitlines(), 1):
            row = json.loads(line)
            if row["seed"] != seed or row["turn"] != turn:
                raise ValueError("Non-contiguous milestone recording")
            if row["max_tile"] >= targets[len(reached)]:
                if not valid_board(row["after"]) or max(map(max, row["after"])) != row["max_tile"]:
                    raise ValueError("Milestone and recorded board differ")
                for target in targets:
                    if row["max_tile"] >= target:
                        reached.setdefault(str(target), turn)
                if len(reached) == len(targets):
                    return reached
        raise ValueError("Missing recorded milestone")
    return reached


def _episode(folder, versions, identity, arm, seed, expected_head):
    result = json.loads(_read(folder/"result.json", versions[0], 256 * 1024))
    if (not isinstance(result, dict) or result.get("schema_version") != 1
            or result.get("split") != "gameplay_dev" or result.get("run_identity") != identity
            or result.get("arm") != arm or type(result.get("seed")) is not int or result["seed"] != seed
            or expected_head is not None and (not isinstance(expected_head, str) or not SHA.fullmatch(expected_head)
                                              or result.get("checkpoint_head_sha256") != expected_head)
            or any(type(result.get(k)) is not int for k in ("turn", "cap", "score", "max_tile"))
            or not 0 < result["turn"] <= result["cap"] or result["score"] < 0 or result["max_tile"] < 2
            or result.get("stop") == "capped" and result["turn"] != result["cap"]
            or not isinstance(result.get("moves_sha256"), str) or not SHA.fullmatch(result["moves_sha256"])):
        raise ValueError("Invalid single-game summary")
    raw = _read(folder/"moves.jsonl", versions[1], MAX_TRACE_BYTES)
    if (hashlib.sha256(raw).hexdigest() != result["moves_sha256"] or not raw.endswith(b"\n")
            or raw.count(b"\n") != result["turn"]):
        raise ValueError("Incomplete or changed evaluation trajectory")
    tail = raw[raw.rfind(b"\n", 0, -1)+1:]
    if len(tail) > MAX_LINE_BYTES:
        raise ValueError("Invalid final evaluation move")
    _final_move(tail, result, arm, seed, None)
    return dict(turns=result["turn"], score=result["score"], max_tile=result["max_tile"], status=result["stop"],
                milestones=_milestones(raw, seed, result["max_tile"]),
                replayIdentity=f"{identity}/{arm}/{seed}/{result['moves_sha256']}")


@dataclass
class _Stage:
    version: tuple
    scanned: float
    rows: list
    episodes: dict = field(default_factory=dict)


class EvaluationCatalog:
    """Cache compact summaries; the existing paged reader serves selected moves."""

    def __init__(self, runs_root, *, audit_seconds=30):
        self.root = Path(os.path.abspath(runs_root)) if runs_root is not None else None
        self.audit_seconds = audit_seconds
        self.lock = Lock()
        self.stages = {}

    def _row(self, watch, source_id, arm, seed, stamp, data):
        channel = watch.base
        return dict(id=f"eval/{source_id}/{arm}/{seed}", source="evaluation", run=channel["name"],
                    channel=channel["id"], arm=arm, model="官方基线" if arm == "official" else arm, seed=seed,
                    time=_modified(stamp), timeSource="modified", **data)

    def _structured(self, watch, source_id, prior, version, now):
        directory = watch.directory
        arms, batches, identity = watch.inputs()
        if not isinstance(identity, str) or not SHA.fullmatch(identity):
            raise ValueError("Missing evaluation identity")
        manifest = json.loads(_read(directory/"resolved-config.json", version[0], 4 * 1024 * 1024))
        if manifest.get("identity") != identity:
            raise ValueError("Evaluation manifest was replaced")
        pins = manifest.get("checkpoint_files", {})
        if not isinstance(pins, dict):
            raise ValueError("Invalid evaluation checkpoint metadata")
        seeds = {s for batch in batches for s in batch}
        skip_official = "official" in arms and bool(watch.reused_baseline(identity, len(seeds)))
        times = _completion_times(watch.path, version[1])
        rows, episodes = [], {}
        for arm in arms:
            if not ARM.fullmatch(arm) or (arm == "official" and skip_official):
                continue
            arm_folder = directory/arm
            arm_pins = pins.get(arm, {})
            if not isinstance(arm_pins, dict):
                continue
            try:
                if not _plain_directory(arm_folder) or arm_folder.resolve(strict=True) != arm_folder:
                    continue
                with os.scandir(arm_folder) as entries:
                    folders = [entry.name for entry in entries if entry.name.isascii() and entry.name.isdecimal()
                               and len(entry.name) <= 16 and int(entry.name) in seeds and str(int(entry.name)) == entry.name
                               and entry.is_dir(follow_symlinks=False)]
            except INVALID:
                continue
            for name in folders:
                seed = int(name)
                folder = arm_folder/name
                key = arm, seed
                try:
                    if not _plain_directory(folder):
                        continue
                    versions = tuple(file_stamp((folder/f).lstat()) for f in ("result.json", "moves.jsonl"))
                    if not all(0 <= stamp[3]/1e9 <= now+300 for stamp in versions):
                        continue
                    cached = prior.episodes.get(key) if prior else None
                    if cached and cached[0] == versions:
                        data = cached[1]
                    else:
                        try:
                            data = _episode(folder, versions, identity, arm, seed, arm_pins.get("head.pt"))
                        except OSError:
                            continue  # Retry temporary sharing errors even if the file stays unchanged.
                        except INVALID:
                            data = None
                    episodes[key] = versions, data
                    if data is None:
                        continue
                    row = self._row(watch, source_id, arm, seed, versions[0][3], data)
                    when = times.get((arm, seed, data["turns"], data["score"], data["max_tile"], data["status"]))
                    if when:
                        row.update(time=when, timeSource="logged")
                    rows.append(row)
                except INVALID:
                    continue
        return rows, episodes

    def _native(self, watch, source_id, result_stamp, now):
        reader = watch.native_replay
        overview = reader.overview(now)
        rows = []
        for seed in overview["seed_catalog"] if overview else ():
            end, = reader.seed_episodes(seed, now)
            targets = [n for n in MILESTONE_TILES if n <= end["max_tile"]]
            reached = {}
            for start in range(1, end["turn"]+1, MAX_PAGE):
                if len(reached) == len(targets):
                    break
                page = reader.page(seed, start, MAX_PAGE, now)
                for frame in page["tracks"][0]["frames"]:
                    for target in targets:
                        if frame["max_tile"] >= target:
                            reached.setdefault(str(target), frame["turn"])
            data = dict(turns=end["turn"], score=end["score"], max_tile=end["max_tile"], status=end["stop"],
                        milestones=reached)
            rows.append(self._row(watch, source_id, reader.arm, seed, result_stamp[3], data))
        return rows

    def catalog(self, watches):
        if self.root is None:
            return []
        rows, current = [], {}
        with self.lock:
            now, tick = time.time(), time.monotonic()
            for watch in watches:
                path = watch.path
                structured = watch.base.get("mode") == "gameplay" and hasattr(watch, "inputs")
                if (path in current or not path.is_relative_to(self.root) or watch.base.get("kind") != "evaluation"
                        or not structured and getattr(watch, "native_replay", None) is None):
                    continue
                try:
                    if watch.path_guard is not None and not watch.path_guard(path):
                        continue
                    # Bind cached rows to this watch, config, and active journal or native move file.
                    journal = file_stamp(_plain_file(path))
                    other = watch.directory/"resolved-config.json" if structured else watch.native_replay.moves
                    version = (file_stamp(_plain_file(other)), journal, watch.base["id"], watch.base["name"])
                    prior = self.stages.get(path)
                    if prior and prior.version == version and (not structured or tick-prior.scanned < self.audit_seconds):
                        stage = prior
                    else:
                        source_id = hashlib.sha256(path.relative_to(self.root).as_posix().encode()).hexdigest()[:24]
                        if structured:
                            # A replaced config must not inherit verified records from a different run.
                            previous = prior if prior and version[0] == prior.version[0] else None
                            games, episodes = self._structured(watch, source_id, previous, version, now)
                        else:
                            games, episodes = self._native(watch, source_id, journal, now), {}
                        stage = _Stage(version, tick, games, episodes)
                    current[path] = stage
                    rows.extend(stage.rows)
                except INVALID:
                    continue
            self.stages = current
        return sorted(rows, key=lambda row:(row["time"], row["id"]), reverse=True)

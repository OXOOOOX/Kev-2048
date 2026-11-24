"""Read the instant-play archive without accepting file paths from clients."""

from collections import OrderedDict
from contextlib import contextmanager
from datetime import UTC, datetime
from itertools import islice
import json
import os
from pathlib import Path
import re
import stat
from threading import Lock

from kev2048_release.game import DIRS, Game, legal, nested
from play_session import MAX_MODELS, MAX_SEED, MAX_TURNS, MODEL_ID


RUN_ID = re.compile(r"[0-9a-f]{32}")
MAX_CATALOG_DIRS = 2048
MAX_MANIFEST_BYTES = 16 * 1024
MAX_MOVES_BYTES = 16 * 1024 * 1024
MAX_ENDLESS_MOVES_BYTES = 128 * 1024 * 1024
# A standard 4×4 board holds at most 4*(2**16-1) tile mass. The opening
# contains at least 4 and each move adds at least 2.
MAX_ENDLESS_RECORD_TURNS = 131_068
MOVE_TAIL_BYTES = 8192
MILESTONE_TILES = (512, 1024, 2048)
MAX_MILESTONE_CACHE = MAX_CATALOG_DIRS * MAX_MODELS
REPARSE_POINT = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0)


def _is_reparse(info):
    return bool(getattr(info, "st_file_attributes", 0) & REPARSE_POINT)


def _plain_directory(path):
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or _is_reparse(info):
        raise ValueError("Archive path must be a regular directory")


@contextmanager
def _open_file(path, limit):
    before = path.lstat()
    if (not stat.S_ISREG(before.st_mode) or _is_reparse(before) or before.st_nlink != 1
            or not 0 < before.st_size <= limit or path.resolve(strict=True) != path):
        raise ValueError("Invalid archive file")
    with os.fdopen(os.open(path, OPEN_FLAGS), "rb") as stream:
        opened = os.fstat(stream.fileno())
        if (not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened)
                or (before.st_size, before.st_mtime_ns) != (opened.st_size, opened.st_mtime_ns)
                or path.resolve(strict=True) != path):
            raise ValueError("Archive file changed while opening")
        yield stream, opened
        after = os.fstat(stream.fileno())
        if (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError("Archive file changed while reading")


def _paths(directory, pattern, limit):
    _plain_directory(directory)
    with os.scandir(directory) as entries:
        names = list(islice(entries, limit + 1))
    return [entry.name for entry in names[:limit] if pattern.fullmatch(entry.name)], len(names) > limit


def _manifest(stream, run_id):
    value = json.loads(stream.read(MAX_MANIFEST_BYTES + 1))
    if not isinstance(value, dict):
        raise ValueError("Archive manifest must be an object")
    name = value.get("variant")
    seeds = value.get("seeds")
    cap = value.get("max_moves_per_game")
    mode = value.setdefault("mode", "classic")
    valid_limit = ((mode == "classic" and type(cap) is int and 1 <= cap <= MAX_TURNS)
                   or (mode == "endless" and "max_moves_per_game" in value and cap is None))
    if (type(value.get("schema_version")) is not int or value["schema_version"] != 1
            or value.get("kind") != "play" or value.get("run_id") != run_id
            or not isinstance(name, str) or not name.strip() or len(name.encode("utf-16-le")) > 320
            or re.search(r"[\x00-\x1f\x7f]", name)
            or not isinstance(seeds, list) or len(seeds) != 1
            or type(seeds[0]) is not int or not 0 <= seeds[0] <= MAX_SEED
            or not valid_limit
            or value.get("files") != {"moves": "moves.jsonl"}):
        raise ValueError("Invalid instant-play archive manifest")
    return value


def _same_rules(row, manifest):
    cap = manifest["max_moves_per_game"]
    return (row.get("mode", "classic") == manifest["mode"] and "cap" in row
            and type(row["cap"]) is type(cap) and row["cap"] == cap)


def _local_time(manifest, fallback):
    value = manifest.get("saved_at")
    if isinstance(value, str) and len(value) <= 40:
        try:
            saved_at = datetime.fromisoformat(value)
            if saved_at.tzinfo is not None:
                return saved_at.astimezone(UTC)
        except (ValueError, OverflowError):
            pass
    return datetime.fromtimestamp(fallback, UTC)


def _check_end_board(board, tile):
    if not isinstance(board, list) or len(board) != 4 or any(not isinstance(row, list) or len(row) != 4 for row in board):
        raise ValueError("Invalid end board")
    flat = [value for row in board for value in row]
    if (any(type(value) is not int or value < 0 or value > 2**30
            or (value != 0 and (value == 1 or value & (value - 1))) for value in flat)
            or max(flat) != tile):
        raise ValueError("Archive tile does not match its end board")
    return tuple(flat)


def _last_result(stream, size, manifest):
    start = max(0, size - MOVE_TAIL_BYTES)
    stream.seek(start)
    data = stream.read(MOVE_TAIL_BYTES)
    if not data.endswith(b"\n"):
        raise ValueError("Archive ends in an incomplete row")
    lines = data.splitlines()
    if start:
        lines = lines[1:]
    last = json.loads(lines[-1]) if lines else None
    if not isinstance(last, dict):
        raise ValueError("Archive has no end result")
    seed, = manifest["seeds"]
    cap = manifest["max_moves_per_game"]
    endless = manifest["mode"] == "endless"
    turns, score, tile = (last.get(key) for key in ("turn", "score", "tile"))
    if (last.get("run_id") != manifest["run_id"] or type(last.get("seed")) is not int or last["seed"] != seed
            or not _same_rules(last, manifest)
            or type(turns) is not int or not 1 <= turns <= (MAX_ENDLESS_RECORD_TURNS if endless else cap)
            or type(score) is not int or not 0 <= score < 2**53
            or type(tile) is not int or not 2 <= tile <= 2**30
            or type(last.get("seq")) is not int):
        raise ValueError("Invalid archive result")
    board = _check_end_board(last.get("board"), tile)
    outcome = last.get("stop")
    if endless:
        terminal = not legal(board)
        if (outcome == "terminal") != terminal or outcome in ("cap", "target"):
            raise ValueError("Endless result does not match the final board")
    if last.get("type") == "move":
        if (last["seq"] != turns or outcome not in ("terminal", "target", "cap")
                or (outcome == "cap" and turns != cap) or (outcome == "target" and tile < 2048)):
            raise ValueError("Archive does not end with a finalized move")
    elif (last.get("type"), outcome) in (("stopped", "stopped"), ("error", None)) and last["seq"] == turns + 1:
        previous = json.loads(lines[-2]) if len(lines) >= 2 else None
        if (not isinstance(previous, dict) or previous.get("type") != "move" or previous.get("seq") != turns
                or not _same_rules(previous, manifest) or previous.get("stop") is not None
                or any(previous.get(key) != last.get(key) for key in
                       ("run_id", "seed", "cap", "turn", "score", "tile", "board"))):
            raise ValueError("Archive final status does not match its last move")
    else:
        raise ValueError("Archive has no finalized result")
    return dict(seed=seed, turns=turns, score=score, max_tile=tile)


def _read_milestones(stream, manifest, metrics, *, through_end=False):
    targets = [tile for tile in MILESTONE_TILES if tile <= metrics["max_tile"]]
    game = Game.seeded(metrics["seed"])
    reached = {}
    stream.seek(0)
    for turn in range(1, metrics["turns"] + 1):
        line = stream.readline(MOVE_TAIL_BYTES + 1)
        if len(line) > MOVE_TAIL_BYTES or not line.endswith(b"\n"):
            raise ValueError("Archive has an incomplete move")
        row = json.loads(line)
        if (not isinstance(row, dict) or row.get("type") != "move" or row.get("run_id") != manifest["run_id"]
                or row.get("seed") != metrics["seed"] or not _same_rules(row, manifest)
                or row.get("seq") != turn or row.get("turn") != turn or row.get("before") != nested(game.state)
                or row.get("action") not in DIRS):
            raise ValueError("Archive moves are not a continuous native trace")
        gained, spawn = game.step(row["action"])
        tile = max(game.state)
        if (row.get("board") != nested(game.state) or row.get("score") != game.score or row.get("tile") != tile
                or row.get("gained") != gained or row.get("spawn") != spawn):
            raise ValueError("Archive move does not match its replayed position")
        if manifest["mode"] == "endless" and row.get("stop") is not None:
            if turn != metrics["turns"] or row["stop"] != "terminal":
                raise ValueError("Endless archive ends before the final move")
        for target in targets:
            if tile >= target:
                reached.setdefault(str(target), turn)
        if not through_end and len(reached) == len(targets):
            break
    if len(reached) != len(targets):
        raise ValueError("Archive milestones do not match the final result")
    if through_end:
        if row.get("stop") is None and not stream.readline(MOVE_TAIL_BYTES + 1).endswith(b"\n"):
            raise ValueError("Endless archive has no final status")
        if stream.read(1):
            raise ValueError("Endless archive has extra events after its final position")
    return reached


class ReplayLibrary:
    """Read only finished tracks under the configured instant-play directory."""

    def __init__(self, root):
        self.root = Path(os.path.abspath(root))
        self._milestone_cache = OrderedDict()
        self._milestone_lock = Lock()

    def _milestones(self, identifier, stream, info, manifest, metrics):
        if metrics["max_tile"] < MILESTONE_TILES[0]:
            return {}
        signature = (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
                     manifest["mode"], manifest["max_moves_per_game"], metrics["seed"], metrics["turns"], metrics["max_tile"])
        with self._milestone_lock:
            cached = self._milestone_cache.get(identifier)
            if cached is not None and cached[0] == signature:
                self._milestone_cache.move_to_end(identifier)
                return dict(cached[1])
        try:
            reached = _read_milestones(stream, manifest, metrics)
        except OSError:
            return {}
        except (ValueError, OverflowError, RecursionError):
            reached = {}
        with self._milestone_lock:
            self._milestone_cache[identifier] = (signature, tuple(reached.items()))
            self._milestone_cache.move_to_end(identifier)
            if len(self._milestone_cache) > MAX_MILESTONE_CACHE:
                self._milestone_cache.popitem(last=False)
        return reached

    def _track(self, run_id, model_id, *, include_moves=False):
        try:
            identifier = f"{run_id}/{model_id}"
            run = self.root / run_id
            folder = run / model_id
            for directory in (self.root, run, folder):
                _plain_directory(directory)
            with _open_file(folder / "run.json", MAX_MANIFEST_BYTES) as (stream, _):
                manifest = _manifest(stream, run_id)
            endless = manifest["mode"] == "endless"
            size_limit = MAX_ENDLESS_MOVES_BYTES if endless else MAX_MOVES_BYTES
            with _open_file(folder / "moves.jsonl", size_limit) as (stream, info):
                metrics = _last_result(stream, info.st_size, manifest)
                if include_moves:
                    if endless:
                        _read_milestones(stream, manifest, metrics, through_end=True)
                    stream.seek(0)
                    moves = stream.read(info.st_size).decode("utf-8")
                else:
                    metrics["milestones"] = self._milestones(identifier, stream, info, manifest, metrics)
            timestamp = _local_time(manifest, info.st_mtime).isoformat(timespec="milliseconds").replace("+00:00", "Z")
            name = manifest["variant"].strip()
            starred = manifest.get("starred") is True
            summary = dict(id=identifier, model=name, mode=manifest["mode"], time=timestamp, starred=starred, **metrics)
            if include_moves:
                # Expose the supported manifest fields and preserve the ordinary offline import path.
                document = dict(schema_version=1, kind="play", variant=name, run_id=run_id, mode=manifest["mode"],
                                seeds=manifest["seeds"], max_moves_per_game=manifest["max_moves_per_game"],
                                files={"moves": "moves.jsonl"}, starred=starred, saved_at=timestamp)
                return dict(id=summary["id"], manifest=document, moves=moves)
            return summary
        except (OSError, ValueError, OverflowError, RuntimeError):
            return None

    def catalog(self):
        games = []
        try:
            runs, truncated = _paths(self.root, RUN_ID, MAX_CATALOG_DIRS)
        except (OSError, ValueError):
            return dict(games=games, truncated=False)
        for run_id in runs:
            try:
                models, more = _paths(self.root / run_id, MODEL_ID, MAX_MODELS)
                truncated |= more
            except (OSError, ValueError):
                continue
            for model_id in models:
                row = self._track(run_id, model_id)
                if row is not None:
                    games.append(row)
        games.sort(key=lambda row: (row["time"], row["id"]), reverse=True)
        return dict(games=games, truncated=truncated)

    def record(self, identifier):
        if not isinstance(identifier, str):
            raise ValueError("Invalid replay ID")
        run_id, _, model_id = identifier.partition("/")
        if not RUN_ID.fullmatch(run_id) or not MODEL_ID.fullmatch(model_id):
            raise ValueError("Invalid replay ID")
        return self._track(run_id, model_id, include_moves=True)

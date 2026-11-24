"""Read bounded pages from the recorded moves of a selected evaluation seed."""

from array import array
from collections import OrderedDict
import json
import math
import os
from threading import Lock

from kev2048_release.game import Game, flatten, legal
from run_summaries import _native_games
from training_discovery import OPEN_FLAGS, _plain_file, _prefix


MAX_PAGE = 256
MAX_TRACE_BYTES = 16 * 1024 * 1024
MAX_CACHE_BYTES = 24 * 1024 * 1024
MAX_LINE_BYTES = 16 * 1024
FRAME_FIELDS = ("seed", "turn", "before", "after", "score", "max_tile", "action", "probabilities", "stop")


def file_stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def valid_board(board):
    return (isinstance(board, list) and len(board) == 4 and all(isinstance(line, list) and len(line) == 4
            and all(type(v) is int and 0 <= v <= 2**30 and (v == 0 or v >= 2 and not v & (v-1))
                    for v in line) for line in board))


def check_decision(row, options):
    probs = row.get("probabilities")
    if (row.get("legal_moves") != options or not options or not isinstance(probs, dict) or set(probs) != set(options)
            or not all(type(v) in (int, float) and 0 <= v <= 1 for v in probs.values())
            or not math.isclose(sum(probs.values()), 1, abs_tol=1e-5)
            or row.get("action") != max(options, key=probs.get)):
        raise ValueError("Inconsistent recorded decision")


def _advance(game, seed, row, *, native=False):
    """Verify an actual recorded move against the seeded game, without predicting one."""
    if (not isinstance(row, dict) or type(row.get("seed")) is not int or row["seed"] != seed
            or type(row.get("turn")) is not int or row["turn"] != game.turn + 1
            or not valid_board(row.get("before")) or flatten(row["before"]) != game.state
            or not valid_board(row.get("after"))):
        raise ValueError("Inconsistent move sequence")
    options = legal(game.state)
    if native:
        row.setdefault("legal_moves", options)
        row["max_tile"] = row.get("tile")
        row.setdefault("stop", None)
    if row.get("stop", False) not in (None, "terminal", "target", "capped", "cap"):
        raise ValueError("Invalid recorded stop")
    check_decision(row, options)
    gained, spawn = game.step(row["action"])
    if (flatten(row["after"]) != game.state or type(row.get("score")) is not int
            or row["score"] != game.score or type(row.get("max_tile")) is not int
            or row["max_tile"] != max(game.state) or row.get("gained", gained) != gained
            or row.get("spawn", spawn) != spawn):
        raise ValueError("Recorded board or score disagrees with this move")


def _page(seed, start, limit, tracks):
    total = max((t["total_turns"] for t in tracks), default=0)
    return dict(seed=seed, start=start, total_turns=total,
                next_start=start+limit if start+limit <= total else None, tracks=tracks)


class _Trace:
    def __init__(self, seed):
        self.seed = seed
        self.raw = b""
        self.game = Game.seeded(seed)
        self.ends = array("I", [0])
        self.stamp = None
        self.stop = None

    def extend(self, raw, stamp):
        start = self.ends[-1]
        while start < len(raw):
            end = raw.find(b"\n", start, start + MAX_LINE_BYTES + 1)
            if end < 0:
                if len(raw)-start > MAX_LINE_BYTES:
                    raise ValueError("Move exceeds the supported line length")
                break  # An active writer can leave an incomplete last line.
            row = json.loads(raw[start:end])
            if self.stop:
                raise ValueError("Recorded rows follow a stopped game")
            _advance(self.game, self.seed, row)
            self.stop = row["stop"]
            start = end + 1
            self.ends.append(start)
        self.raw, self.stamp = raw, stamp

    def frame(self, turn):
        record = json.loads(self.raw[self.ends[turn-1]:self.ends[turn]])
        return {key:record[key] for key in FRAME_FIELDS}


class GameplayReplay:
    def __init__(self):
        self.lock = Lock()
        self.cache = OrderedDict()
        self.bytes_cached = 0

    def _load(self, path, seed, expected_stamp, now):
        info = _plain_file(path)
        stamp = file_stamp(info)
        if (expected_stamp is not None and stamp != expected_stamp or info.st_size > MAX_TRACE_BYTES
                or not 0 <= info.st_mtime <= now + 300):
            raise ValueError("Move file changed or exceeds the replay limit")
        key = (path, seed)
        prior = self.cache.get(key)
        if prior is not None and prior.stamp == stamp:
            self.cache.move_to_end(key)
            return prior
        raw = _prefix(path, info, info.st_size)
        after = _plain_file(path)
        if (len(raw) != info.st_size or file_stamp(after) != stamp
                and (expected_stamp is not None or after.st_size <= info.st_size or
                     (after.st_dev, after.st_ino) != (info.st_dev, info.st_ino))):
            raise ValueError("Move file changed while reading")
        if prior is not None:
            self.bytes_cached -= len(prior.raw)
            del self.cache[key]
        # Compare the bytes too: rewriting or truncating a file must invalidate its prefix.
        trace = prior if prior is not None and raw.startswith(prior.raw) else _Trace(seed)
        trace.extend(raw, stamp)
        self.cache[key] = trace
        self.bytes_cached += len(raw)
        while self.bytes_cached > MAX_CACHE_BYTES or len(self.cache) > 128:
            _, discarded = self.cache.popitem(last=False)
            self.bytes_cached -= len(discarded.raw)
        return trace

    def page(self, sources, seed, start, limit, now):
        tracks = []
        for arm, path, reused, expected_stamp in sources:
            track = dict(arm=arm, reused=reused, frames=[], total_turns=0, status="waiting")
            try:
                with self.lock:
                    trace = self._load(path, seed, expected_stamp, now)
                    length = trace.game.turn
                    if length:
                        track.update(status="recorded", total_turns=length,
                                     frames=[trace.frame(t) for t in range(start, min(length+1, start+limit))])
                        if length < start:
                            track["last_frame"] = trace.frame(length)
            except FileNotFoundError:
                if expected_stamp is not None:
                    track["status"] = "invalid"
            except (ValueError, OSError, RuntimeError, OverflowError, TypeError):
                track["status"] = "invalid"
            tracks.append(track)
        return _page(seed, start, limit, tracks)


class NativeGameplayReplay:
    """One native play result and its adjacent, multi-seed moves.jsonl."""

    def __init__(self, result, arm):
        self.result, self.moves = result, result.with_name("moves.jsonl")
        self.arm = arm
        self.lock = Lock()
        self.stamp = None
        self.records = {}

    def _build(self, result_info, moves_info):
        if result_info.st_size > 4*1024*1024 or not 0 < moves_info.st_size <= MAX_TRACE_BYTES:
            raise ValueError("Native recording is outside supported limits")
        result = json.loads(_prefix(self.result, result_info, result_info.st_size))
        if not isinstance(result, dict) or not _native_games(result, []):
            raise ValueError("Unrecognized native report")
        raw = _prefix(self.moves, moves_info, moves_info.st_size)
        if len(raw) != moves_info.st_size:
            raise ValueError("Native recording changed")
        offset, records = 0, {}
        for summary in result["games"]:
            seed = summary["seed"]
            if seed in records or summary["turn"] < 1:
                raise ValueError("Ambiguous native episode")
            game = Game.seeded(seed)
            ends = array("I", [offset])
            for turn in range(1, summary["turn"]+1):
                end = raw.find(b"\n", offset, offset+MAX_LINE_BYTES+1)
                if end < 0:
                    raise ValueError("Missing complete native move")
                row = json.loads(raw[offset:end])
                _advance(game, seed, row, native=True)
                if row["stop"] is not None and (turn != summary["turn"] or row["stop"] != summary["stop"]):
                    raise ValueError("Native move conflicts with its reported stop")
                offset = end + 1
                ends.append(offset)
            moves_remain = bool(legal(game.state))
            if (game.score != summary["score"] or max(game.state) != summary["tile"]
                    or moves_remain and (summary["stop"] != "cap" or game.turn != result["cap"])
                    or not moves_remain and summary["stop"] != "terminal"):
                raise ValueError("Native summary disagrees with its moves")
            final = {k:row[k] for k in FRAME_FIELDS}
            final.update(stop=summary["stop"], arm=self.arm)
            records[seed] = (ends, final)
        if offset != len(raw):
            raise ValueError("Unaccounted native moves")
        return records

    def _load(self, now):
        # Keep only verified offsets and final frames. Reload on replacement of either sibling.
        try:
            info = _plain_file(self.result), _plain_file(self.moves)
            stamp = tuple(file_stamp(s) for s in info)
            if not all(0 <= s.st_mtime <= now+300 for s in info):
                return {}
            if stamp != self.stamp:
                self.stamp, self.records = stamp, {}
                records = self._build(*info)
                if tuple(file_stamp(_plain_file(p)) for p in (self.result, self.moves)) == stamp:
                    self.records = records
            return self.records
        except (OSError, ValueError, RuntimeError, TypeError, OverflowError):
            return {}

    def overview(self, now):
        with self.lock:
            records = self._load(now)
            if records:
                return dict(mode="gameplay", arms=[self.arm], seed_catalog=list(records),
                            games=len(records), total_games=len(records), batch=None, episodes=[])

    def seed_episodes(self, seed, now):
        with self.lock:
            record = self._load(now).get(seed)
            return [record[1]] if record is not None else None

    def page(self, seed, start, limit, now):
        with self.lock:
            record = self._load(now).get(seed)
            if record is None:
                return None
            ends, final = record
            turns = final["turn"]
            track = dict(arm=self.arm, status="recorded", reused=False, total_turns=turns, frames=[])
            if start > turns:
                track["last_frame"] = final
            else:
                last = min(turns, start+limit-1)
                with os.fdopen(os.open(self.moves, OPEN_FLAGS), "rb") as stream:
                    # On Windows the fd and path APIs can expose different ctime semantics.
                    if file_stamp(os.fstat(stream.fileno()))[:4] != self.stamp[1][:4] or self.moves.resolve(strict=True) != self.moves:
                        raise ValueError("Native recording changed")
                    stream.seek(ends[start-1])
                    data = stream.read(ends[last]-ends[start-1])
                if file_stamp(_plain_file(self.moves)) != self.stamp[1] or file_stamp(_plain_file(self.result)) != self.stamp[0]:
                    raise ValueError("Native result changed during replay")
                for line in data.splitlines():
                    row = json.loads(line)
                    row.update(max_tile=row["tile"], stop=final["stop"] if row["turn"] == turns else None)
                    track["frames"].append({k:row[k] for k in FRAME_FIELDS})
            return _page(seed, start, limit, [track])

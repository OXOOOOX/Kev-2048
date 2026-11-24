"""Read paired episodes cryptographically linked by an evaluation comparison plan."""

from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from threading import Lock
import time

from kev2048_release.game import flatten, legal
from kev2048_release.artifacts import identity as object_identity
from gameplay_replay import check_decision, file_stamp, valid_board
from training_discovery import OPEN_FLAGS, _plain_file, _prefix


SHA = re.compile("[a-f0-9]{64}")
ARM = re.compile("[a-z0-9][a-z0-9-]{0,63}")
INVALID = (OSError, ValueError, TypeError, KeyError, AttributeError, RuntimeError, OverflowError)
METADATA_LIMIT = 16 * 1024 * 1024


def _sha(value):
    return isinstance(value, str) and SHA.fullmatch(value) is not None


def _final_move(raw, summary, arm, seed, updated):
    row = json.loads(raw)
    if (not isinstance(row, dict) or row.get("seed") != seed
            or not all(valid_board(row.get(k)) for k in ("before", "after"))
            or row.get("stop") not in ("terminal", "target", "capped")
            or any(type(row.get(k)) is not int or not 0 <= row[k] < 2**53 for k in ("turn", "score", "max_tile"))
            or any(row[k] != summary.get(k) for k in ("turn", "score", "max_tile", "stop"))
            or max(flatten(row["after"])) != row["max_tile"]):
        raise ValueError("Reused final move disagrees with its signed result")
    check_decision(row, legal(flatten(row["before"])))
    return dict(arm=arm, seed=seed, updated_at=updated, reused=True,
                **{k:row[k] for k in ("turn", "before", "after", "probabilities", "action", "score", "max_tile", "stop")})


class GameplayReuse:
    """Only expose completed, matching episodes from a declared source in the same runs root."""

    def __init__(self, runs_root, gameplay_dir):
        self.root = Path(os.path.abspath(runs_root)) if runs_root is not None else None
        self.directory = Path(os.path.abspath(gameplay_dir))
        self.lock = Lock()
        self.metadata = OrderedDict()
        self.moves = {}
        self.source_version = None
        self.catalog = None

    def _info(self, path):
        if self.root is None or not path.is_relative_to(self.root) or ".." in path.parts:
            raise ValueError("Archive outside registered runs root")
        return _plain_file(path)

    def _json(self, path):
        info = self._info(path)
        stamp = file_stamp(info)
        cached = self.metadata.get(path)
        if cached and cached[0] == stamp:
            self.metadata.move_to_end(path)
            return cached[1], cached[2]
        if info.st_size > METADATA_LIMIT:
            raise ValueError("Archive metadata too large")
        raw = _prefix(path, info, info.st_size)
        if len(raw) != info.st_size or file_stamp(self._info(path)) != stamp:
            raise ValueError("Archive metadata changed")
        document = json.loads(raw)
        if not isinstance(document, dict):
            raise ValueError("Invalid archive metadata")
        digest = hashlib.sha256(raw).hexdigest()
        self.metadata[path] = (stamp, document, digest)
        if len(self.metadata) > 8:
            self.metadata.popitem(last=False)
        return document, digest

    def _cohort(self, identity, arms, fresh):
        current, _ = self._json(self.directory/"resolved-config.json")
        if (not _sha(current.get("source_gameplay_identity")) or not _sha(current.get("comparison_plan_sha256"))
                or not _sha(identity) or current.get("identity") != identity
                or current["identity"] != object_identity({k:v for k,v in current.items() if k != "identity"})):
            return None
        plan_path = self.directory.parent/"plan.json"
        plan, plan_sha = self._json(plan_path)
        initial = plan.get("initial_seeds")
        if (plan_sha != current["comparison_plan_sha256"] or not isinstance(initial, list) or not initial
                or len(initial) > 10000 or not all(type(s) is int and 0 <= s < 2**53 for s in initial)
                or len(initial) != len(set(initial)) or set(initial).intersection(fresh)
                or not arms or not all(isinstance(a, str) and ARM.fullmatch(a) for a in arms)
                or set(arms) != set(current["checkpoints"]) or set(arms) != set(current["checkpoint_files"])
                or plan["additional_seeds"] != list(fresh) or current["seeds"] != list(fresh)
                or plan["reused_games"] != len(arms)*len(initial) or plan["fresh_games"] != len(arms)*len(fresh)
                or plan["models"] != current["checkpoints"] or plan["checkpoint_files"] != current["checkpoint_files"]
                or not all(_sha(current["checkpoint_files"][a].get("head.pt")) for a in arms)
                or not isinstance(plan.get("source"), str) or not plan["source"]):
            return None
        # Select only this fixed pair of archive manifests below the declared local source.
        source = Path(os.path.abspath(plan_path.parent/plan["source"]))/"gameplay"
        if source == self.directory or not source.is_relative_to(self.root):
            return None
        config_path, result_path = source/"resolved-config.json", source/"result.json"
        old, old_sha = self._json(config_path)
        report, report_sha = self._json(result_path)
        pins = current["inputs_sha256"]
        if (pins.get(str(config_path)) != old_sha or pins.get(str(result_path)) != report_sha
                or plan["archived_gameplay_sha256"] != report_sha or old.get("identity") != current["source_gameplay_identity"]
                or old["identity"] != object_identity({k:v for k,v in old.items() if k != "identity"})
                or report.get("status") != "completed" or report.get("identity") != old["identity"]
                or old.get("seeds") != initial or report.get("seeds") != initial
                or any(old.get(k) != current.get(k) for k in
                       ("config", "checkpoints", "checkpoint_files", "training_identity", "baseline_training_identity",
                        "reference_environment", "predictions", "dtype", "cap", "target_tile", "temperature",
                        "final_seed_list_sha256"))):
            return None
        hashes = report["episode_sha256"]
        if set(hashes) != set(arms) or not all(isinstance(hashes[a], dict) for a in arms):
            return None
        version = (source, identity, old_sha, report_sha)
        if self.source_version != version:
            self.source_version = version
            self.moves.clear()
        return source, old, hashes, initial

    def _episode(self, cohort, arm, seed, now):
        source, old, hashes, _ = cohort
        expected = hashes[arm].get(str(seed))
        if not _sha(expected):
            return None
        folder = source/arm/str(seed)
        summary_path, moves_path = folder/"result.json", folder/"moves.jsonl"
        summary_info, moves_info = self._info(summary_path), self._info(moves_path)
        if not 0 <= moves_info.st_mtime <= now + 300 or summary_info.st_size > 256*1024:
            return None
        stamp = file_stamp(summary_info), file_stamp(moves_info), expected
        cached = self.moves.get((arm, seed))
        if cached and cached[0] == stamp:
            return cached[1]
        raw = _prefix(summary_path, summary_info, summary_info.st_size)
        if hashlib.sha256(raw).hexdigest() != expected or file_stamp(self._info(summary_path)) != stamp[0]:
            return None
        result = json.loads(raw)
        if (not isinstance(result, dict) or result.get("run_identity") != old["identity"]
                or result.get("arm") != arm or type(result.get("seed")) is not int or result["seed"] != seed
                or result.get("checkpoint_head_sha256") != old["checkpoint_files"][arm]["head.pt"]
                or any(result.get(k) != old.get(k) for k in ("cap", "dtype", "temperature"))
                or not _sha(result.get("moves_sha256"))):
            return None
        digest, tail = hashlib.sha256(), b""
        with os.fdopen(os.open(moves_path, OPEN_FLAGS), "rb") as stream:
            if not os.path.samestat(moves_info, os.fstat(stream.fileno())) or moves_path.resolve(strict=True) != moves_path:
                return None
            for chunk in iter(lambda: stream.read(128*1024), b""):
                digest.update(chunk)
                tail = (tail+chunk)[-16384:]
        if (digest.hexdigest() != result["moves_sha256"] or not tail.endswith(b"\n")
                or file_stamp(self._info(moves_path)) != stamp[1]):
            return None
        updated = datetime.fromtimestamp(moves_info.st_mtime, timezone.utc).isoformat().replace("+00:00","Z")
        move = _final_move(tail.splitlines()[-1], result, arm, seed, updated)
        self.moves[(arm,seed)] = (stamp, move)
        return move

    def _both(self, cohort, arms, seed, now):
        try:
            moves = [self._episode(cohort, arm, seed, now) for arm in arms]
            return [m.copy() for m in moves] if all(moves) else None
        except INVALID:
            return None

    def seed_catalog(self, identity, arms, fresh_seeds):
        key = identity, tuple(arms), tuple(fresh_seeds)
        with self.lock:
            if self.catalog and self.catalog[0] == key and time.monotonic()-self.catalog[1] < 5:
                return list(self.catalog[2])
            available = []
            try:
                cohort = self._cohort(identity, arms, fresh_seeds)
                if cohort:
                    now = time.time()
                    available = [s for s in cohort[3] if self._both(cohort, arms, s, now)]
            except INVALID:
                pass
            self.catalog = key, time.monotonic(), available
            return list(available)

    def seed_episodes(self, identity, arms, fresh_seeds, seed, now):
        if type(seed) is not int:
            return None
        with self.lock:
            try:
                cohort = self._cohort(identity, arms, fresh_seeds)
                if cohort and seed in cohort[3]:
                    return self._both(cohort, arms, seed, now)
            except INVALID:
                pass
            return None

    def replay_sources(self, identity, arms, fresh_seeds, seed, now):
        """Resolve only the exact moves files whose bytes the paired report verified."""
        with self.lock:
            try:
                cohort = self._cohort(identity, arms, fresh_seeds)
                if cohort and seed in cohort[3] and self._both(cohort, arms, seed, now):
                    return [(arm, cohort[0]/arm/str(seed)/"moves.jsonl", True, self.moves[arm, seed][0][1])
                            for arm in arms]
            except INVALID:
                pass
            return None

"""Index same-seed comparisons from registered runs and their recorded episode identities."""

from collections import defaultdict
from functools import lru_cache
import json
from pathlib import Path

from evaluation_catalog import INVALID, _read
from gameplay_replay import file_stamp
from gameplay_reuse import ARM, _sha
from run_discovery import MAX_RESULT_BYTES, _date_group
from training_discovery import _plain_file


def _run_folder(path, root):
    if root is None or not path.is_relative_to(root):
        return None
    parts = path.parent.relative_to(root).parts
    if not parts or ".." in parts:
        return None
    if _date_group(parts[0]):
        return root.joinpath(*parts[:2]) if len(parts) >= 2 else None
    return root / parts[0]


def _valid_seed(value):
    return type(value) is int and 0 <= value < 2**53


@lru_cache(maxsize=64)
def _declared_games(directory, config_stamp, result_stamp):
    """Keep only the episode keys from a consistent, completed aggregate."""
    config = json.loads(_read(directory/"resolved-config.json", config_stamp, MAX_RESULT_BYTES))
    result = json.loads(_read(directory/"result.json", result_stamp, MAX_RESULT_BYTES))
    if (not isinstance(config, dict) or config.get("schema_version") != 1 or not _sha(config.get("identity"))
            or not isinstance(result, dict) or result.get("identity") != config["identity"]
            or result.get("status") not in ("complete", "completed") or result.get("complete") is False
            or not isinstance(result.get("results"), dict)):
        return ()
    keys = {}
    for arm, records in result["results"].items():
        if not ARM.fullmatch(arm) or not isinstance(records, list):
            continue
        for record in records:
            if not isinstance(record, dict) or not _valid_seed(record.get("seed")):
                continue
            seed, run, digest = record["seed"], record.get("run_identity"), record.get("moves_sha256")
            replay = f"{run}/{arm}/{seed}/{digest}" if record.get("arm") == arm and _sha(run) and _sha(digest) else None
            key = arm, seed
            # Contradictory declarations for one slot cannot select a source.
            keys[key] = replay if keys.get(key, replay) == replay else None
    return tuple((arm, seed, replay) for (arm, seed), replay in keys.items() if replay)


def _game_keys(watch, root):
    directory = getattr(watch, "directory", None)
    if directory != watch.path.parent or _run_folder(watch.path, root) is None:
        return ()
    try:
        config = file_stamp(_plain_file(directory/"resolved-config.json"))
        result = file_stamp(_plain_file(directory/"result.json"))
        return _declared_games(directory, config, result)
    except INVALID:
        return ()


def build_comparison_catalog(watches, channels, evaluations, runs_root):
    """List periods and route only explicitly claimed, verified cross-run episodes."""
    root = Path(runs_root).absolute() if runs_root is not None else None
    watches = {watch.base["id"]: watch for watch in watches}
    channels = list(channels)
    periods, paths, canonical = {}, {}, {}
    owners = defaultdict(list)
    for channel in channels:
        channel_id = channel["id"]
        watch = watches.get(channel_id)
        if watch is None or channel.get("mode") != "gameplay":
            continue
        owner = canonical[channel_id] = paths.setdefault(watch.path, channel_id)
        if owner != channel_id:
            periods[owner]["aliases"].append(channel_id)
            continue
        arms = list(dict.fromkeys(a for a in channel.get("arms", ()) if isinstance(a, str) and a))
        seeds = list(dict.fromkeys(s for s in channel.get("seed_catalog", ()) if _valid_seed(s)))
        periods[owner] = dict(id=owner, name=channel["name"], aliases=[], updated_at=channel.get("updated_at"),
                              direct_arms=arms, arms=arms.copy(), seeds=seeds, references={})
        group = _run_folder(watch.path, root)
        if group is not None:
            owners[group].append(owner)

    # A parent summary or sibling training stage can select its run's sole gameplay stage.
    # Display names can collide, particularly in date-prefixed directories.
    for channel in channels:
        channel_id = channel["id"]
        watch = watches.get(channel_id)
        if watch is not None and channel_id not in canonical:
            matches = owners.get(_run_folder(watch.path, root), ())
            if len(matches) == 1:
                periods[matches[0]]["aliases"].append(channel_id)

    sources, local = {}, set()
    for episode in evaluations:
        origin = canonical.get(episode.get("channel"))
        arm, seed, replay = episode.get("arm"), episode.get("seed"), episode.get("replayIdentity")
        if origin is None or not isinstance(arm, str) or not _valid_seed(seed) or not isinstance(replay, str):
            continue
        key = arm, seed, replay
        sources[key] = origin if sources.get(key, origin) == origin else None
        local.add((origin, arm, seed))
        if episode.get("embedded") is True:
            periods[origin]["embedded"] = True

    for period in periods.values():
        current = period["id"]
        seeds = set(period["seeds"])
        for arm, seed, replay in _game_keys(watches[current], root):
            source = sources.get((arm, seed, replay))
            if not source or source == current or seed not in seeds or (current, arm, seed) in local:
                continue
            period["references"].setdefault(str(seed), {})[arm] = source
            if arm not in period["arms"]:
                period["arms"].append(arm)
    ordered = sorted(periods.values(), key=lambda p: (p["updated_at"] or "", p["id"]), reverse=True)
    return {"periods": ordered}

"""Index completed multi-game gameplay evaluations for the workbench.

This module deliberately reads evaluation manifests only.  It does not load a
model, replay a game, or hash checkpoint contents.  A checkpoint is identified
by the complete ``head.pt`` plus encoder/tokenizer pin set recorded in a
gameplay resolved configuration.  Consequently an arm name is only a label;
renaming an arm does not create a second model and a same-named arm with a
different head does not alias an existing model.

The production API is a single cached index::

    index = PlayEvaluationIndex("runs")
    summary = index.summarize(checkpoint_files)  # or "teacher"

``summary["batches"]`` contains independent completed batches.  ``latest`` is
the newest batch with at least one newly measured game; if every batch is a
reused control it falls back to the newest batch and retains its ``reused``
marker.  ``overall`` is pooled only after checking the gameplay protocol and
deduplicating equal checkpoint/seed pairs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any


CHECKPOINT_FILE_NAMES = (
    "head.pt",
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
)
_CHECKPOINT_NAME_SET = set(CHECKPOINT_FILE_NAMES)
_SHA256 = re.compile(r"^[0-9a-fA-F]{64}$")
_RUN_NUMBER = re.compile(r"(?:^|-)(?:round)(\d+)(?:-|$)", re.IGNORECASE)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None


def _normal_name(name: Any) -> str:
    """Return a checkpoint basename for either slash convention."""

    return str(name).replace("\\", "/").rsplit("/", 1)[-1]


def _normalize_checkpoint_files(value: Any) -> dict[str, str] | None:
    """Normalize and validate the complete head+encoder fingerprint.

    Missing encoder pins are intentionally rejected.  Matching a head alone
    would permit a model with a changed encoder/tokenizer to be presented as
    the same model in the workbench.
    """

    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in value.items():
        name = _normal_name(raw_name)
        if name not in _CHECKPOINT_NAME_SET:
            continue
        if not isinstance(raw_digest, str) or not _SHA256.fullmatch(raw_digest):
            return None
        normalized[name] = raw_digest.lower()
    if set(normalized) != _CHECKPOINT_NAME_SET:
        return None
    return {name: normalized[name] for name in CHECKPOINT_FILE_NAMES}


def checkpoint_id(checkpoint_files: Mapping[str, Any]) -> str | None:
    """Return the stable public id for a complete checkpoint fingerprint."""

    normalized = _normalize_checkpoint_files(checkpoint_files)
    if normalized is None:
        return None
    canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode("utf-8")
    fingerprint = hashlib.sha256(canonical).hexdigest()
    return "checkpoint:" + fingerprint


def _arm_checkpoint_maps(document: Any, hint: str | None = None) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Collect pins from the few checkpoint metadata shapes used by the runs.

    This is intentionally shallow.  Evaluation discovery must not walk the
    large training data/episode JSON objects merely to find a checkpoint pin.
    """

    named: dict[str, dict[str, str]] = {}
    unlabelled: list[dict[str, str]] = []
    if not isinstance(document, Mapping):
        return named, unlabelled

    def add(value: Any, local_hint: str | None = None) -> None:
        direct = _normalize_checkpoint_files(value)
        if direct is not None:
            if local_hint:
                named.setdefault(local_hint, direct)
            else:
                unlabelled.append(direct)
            return
        if not isinstance(value, Mapping):
            return
        for arm, candidate in value.items():
            candidate_files = _normalize_checkpoint_files(candidate)
            if candidate_files is not None:
                named.setdefault(str(arm), candidate_files)

    hint_value = document.get("candidate_id") or document.get("trial") or document.get("arm")
    if not isinstance(hint_value, str):
        hint_value = hint
    for key in ("checkpoint_files", "reference_checkpoint_files"):
        add(document.get(key), hint_value if key == "checkpoint_files" else None)

    # candidate.json stores the pin map below {candidate: {id, checkpoint_files}}
    # while collection-plan.json stores it directly at the top level.
    for key in ("candidate", "reference", "model"):
        child = document.get(key)
        if not isinstance(child, Mapping):
            continue
        child_hint = child.get("id") or child.get("arm") or child.get("name")
        add(child.get("checkpoint_files"), child_hint if isinstance(child_hint, str) else hint_value)
    for key in ("candidates", "finalists"):
        children = document.get(key)
        if not isinstance(children, list):
            continue
        for child in children:
            if not isinstance(child, Mapping):
                continue
            child_hint = child.get("id") or child.get("arm") or child.get("trial")
            add(child.get("checkpoint_files"), child_hint if isinstance(child_hint, str) else hint_value)
    return named, unlabelled


def _candidate_document_paths(run_dir: Path) -> list[Path]:
    """Return small metadata files that may carry a reused arm's pins."""

    paths = [
        run_dir / "resolved-config.json",
        run_dir / "gameplay" / "resolved-config.json",
        run_dir / "candidate.json",
        run_dir / "data" / "collection-plan.json",
        run_dir / "training" / "result.json",
    ]
    return [path for path in paths if path.is_file()]


def _path_value(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_dir() else None


def _referenced_run_dirs(document: Mapping[str, Any]) -> list[Path]:
    """Find explicit source runs used for reused baseline episodes."""

    found: list[Path] = []
    keys = ("reference_run", "original_source", "source_run", "b18_run", "reference_source")
    documents = [document]
    nested = document.get("config")
    if isinstance(nested, Mapping):
        documents.append(nested)
    for current in documents:
        for key in keys:
            path = _path_value(current.get(key))
            if path is not None:
                found.append(path)
    result: list[Path] = []
    seen: set[str] = set()
    for path in found:
        token = str(path.resolve())
        if token not in seen:
            seen.add(token)
            result.append(path)
    return result


def _maps_for_run(run_dir: Path, visited: set[str] | None = None) -> tuple[dict[str, dict[str, str]], list[dict[str, str]]]:
    """Load named pins from a run and explicitly referenced source runs."""

    if visited is None:
        visited = set()
    try:
        token = str(run_dir.resolve())
    except OSError:
        token = str(run_dir)
    if token in visited:
        return {}, []
    visited.add(token)

    named: dict[str, dict[str, str]] = {}
    unlabelled: list[dict[str, str]] = []
    documents: list[tuple[Path, Any]] = []
    for path in _candidate_document_paths(run_dir):
        document = _read_json(path)
        if document is not None:
            documents.append((path, document))
            local_named, local_unlabelled = _arm_checkpoint_maps(document)
            for arm, pins in local_named.items():
                named.setdefault(arm, pins)
            unlabelled.extend(local_unlabelled)
    for _path, document in documents:
        if isinstance(document, Mapping):
            for source_dir in _referenced_run_dirs(document):
                source_named, source_unlabelled = _maps_for_run(source_dir, visited)
                for arm, pins in source_named.items():
                    named.setdefault(arm, pins)
                unlabelled.extend(source_unlabelled)
    return named, unlabelled


def _run_order(run_name: str) -> tuple[int, str]:
    match = _RUN_NUMBER.search(run_name)
    if match:
        return int(match.group(1)), run_name
    # Development rounds predate the numbered runs.  Lexical order remains a
    # deterministic tie-breaker, while numbered rounds are always newer.
    if run_name.startswith("gameplay-dev"):
        return 0, run_name
    return -1, run_name


def _batch_sort_key(entry: Mapping[str, Any]) -> tuple[int, str, str]:
    run = str(entry.get("batch") or entry.get("source") or "")
    number, name = _run_order(run)
    return number, name, str(entry.get("source_id") or "")


def _run_label(run_name: str) -> str:
    """Short Chinese batch label suitable for a model card."""

    match = _RUN_NUMBER.search(run_name)
    if match:
        number = int(match.group(1))
        names = {
            13: "新开发种子",
            12: "固定开发种子",
            11: "新增20局",
            10: "固定开发种子",
            9: "固定开发种子",
            8: "复测",
            7: "开发种子",
            6: "开发种子",
        }
        return f"第 {number} 轮 · {names.get(number, '整局评测')}"
    if run_name.startswith("gameplay-dev-head8000"):
        return "开发评测 · 8000头筛选"
    if run_name.startswith("gameplay-dev-head2000"):
        return "开发评测 · 2000头"
    return "整局评测"


def _regime(config: Mapping[str, Any]) -> tuple[Any, ...]:
    """The gameplay protocol fields required before pooling batches."""

    return tuple(config.get(name) for name in ("cap", "target_tile", "temperature", "dtype"))


def _reference_arms(config: Mapping[str, Any], result: Mapping[str, Any], arms: set[str]) -> set[str]:
    """Identify arms whose episodes are reused controls in a batch."""

    explicit_reference = result.get("reference_id") or config.get("reference_arm")
    candidate_names: set[str] = set()
    candidates = result.get("candidates")
    if isinstance(candidates, Mapping):
        candidate_names.update(str(name) for name in candidates)
    elif isinstance(config.get("candidates"), list):
        candidate_names.update(str(name) for name in config["candidates"])
    candidate_id = result.get("candidate_id")
    if not isinstance(candidate_id, str):
        candidate_id = config.get("candidate_id")
    nested_config = config.get("config")
    if not isinstance(candidate_id, str) and isinstance(nested_config, Mapping):
        candidate_id = nested_config.get("candidate_id")
    if isinstance(candidate_id, str):
        candidate_names.add(candidate_id)
    reused: set[str] = set()
    if isinstance(explicit_reference, str) and explicit_reference in arms:
        reused.add(explicit_reference)
    if candidate_names:
        reused.update(arms - candidate_names)
    return reused


def _valid_episode_rows(rows: Any, arm: str, pins: Mapping[str, str]) -> dict[int, float] | None:
    if not isinstance(rows, list) or not rows:
        return None
    scores: dict[int, float] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            return None
        seed = row.get("seed")
        score = row.get("score")
        if type(seed) is not int or isinstance(score, bool) or not isinstance(score, (int, float)):
            return None
        if not math.isfinite(float(score)) or seed in scores:
            return None
        row_arm = row.get("arm")
        if row_arm is not None and row_arm != arm:
            return None
        row_head = row.get("checkpoint_head_sha256")
        if row_head is not None and row_head != pins["head.pt"]:
            return None
        stop = row.get("stop")
        if stop is not None and stop not in {"target", "terminal", "capped"}:
            return None
        scores[seed] = float(score)
    return scores


def _entry(*, run_dir: Path, arm: str,
           rows: Mapping[int, float], checkpoint_key: str, stage: Any,
           pins: Mapping[str, str], result: Mapping[str, Any],
           regime: tuple[Any, ...], reused: bool = False) -> dict[str, Any]:
    run_name = run_dir.name
    source = run_name
    source_id = f"{run_name}:{arm}"
    label = _run_label(run_name)
    values = list(rows.values())
    reused_count = len(rows) if reused and result.get("reference_games_reused", 0) else 0
    entry = {
        "source": source,
        "source_id": source_id,
        "sourceId": source_id,
        "label": label,
        "batch": run_name,
        "arm": arm,
        "stage": stage,
        "checkpoint_id": checkpoint_key,
        "head_sha256": pins["head.pt"],
        "games": len(values),
        "mean_score": statistics.fmean(values),
        "unique_seeds": len(rows),
        "regime": regime,
        "reused": bool(reused_count),
        "reused_games": reused_count,
        "fresh_games": len(rows) - reused_count,
        "_scores": dict(rows),
        "_regime": regime,
        "_sort_key": _batch_sort_key({"batch": run_name, "source_id": source_id}),
    }
    return entry


def _is_batch_report(result: Any, config: Any) -> bool:
    if not isinstance(result, Mapping) or result.get("status") != "completed":
        return False
    if not isinstance(result.get("results"), Mapping):
        return False
    if not isinstance(config, Mapping) or not isinstance(config.get("seeds"), list):
        return False
    named, unlabelled = _arm_checkpoint_maps(config)
    return bool(named or unlabelled)


def _scan_batches(runs_root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    try:
        # Only these two manifest locations are formal multi-game reports.
        # Per-episode result.json files under gameplay/<arm>/<seed>/ are not
        # enumerated, and training trees are never searched.
        result_paths = sorted(set(runs_root.glob("*/result.json")) |
                              set(runs_root.glob("*/gameplay/result.json")))
    except OSError:
        return entries
    for result_path in result_paths:
        result = _read_json(result_path)
        config_path = result_path.parent / "resolved-config.json"
        config = _read_json(config_path)
        if not _is_batch_report(result, config):
            continue
        assert isinstance(result, Mapping)
        assert isinstance(config, Mapping)
        run_dir = result_path.parent.parent if result_path.parent.name == "gameplay" else result_path.parent
        named, unlabelled = _maps_for_run(run_dir)
        # Explicit config pins take precedence over auxiliary source metadata.
        config_named, config_unlabelled = _arm_checkpoint_maps(config)
        for arm, pins in config_named.items():
            named[arm] = pins
        unlabelled = config_unlabelled + unlabelled

        results = result.get("results")
        if not isinstance(results, Mapping):
            continue
        reference_arms = _reference_arms(config, result, {str(name) for name in results})
        for arm, rows in results.items():
            arm_name = str(arm)
            pins = named.get(arm_name)
            if pins is None and isinstance(rows, list):
                # A reused baseline may have no arm entry in this config, but
                # each episode carries a cryptographic head pin.
                heads = {
                    row.get("checkpoint_head_sha256")
                    for row in rows if isinstance(row, Mapping)
                    and isinstance(row.get("checkpoint_head_sha256"), str)
                }
                matching = [candidate for candidate in unlabelled if candidate["head.pt"] in heads]
                if len(matching) == 1:
                    pins = matching[0]
            if pins is None:
                continue
            scores = _valid_episode_rows(rows, arm_name, pins)
            if scores is None:
                continue
            key = checkpoint_id(pins)
            if key is None:
                continue
            entries.append(_entry(run_dir=run_dir, arm=arm_name,
                                  rows=scores, checkpoint_key=key,
                                  stage=result.get("stage"), pins=pins,
                                  result=result, regime=_regime(config),
                                  reused=arm_name in reference_arms))
    return entries


def _teacher_entries(runs_root: Path, teacher: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    candidates = list(runs_root.glob("*/calibration.json"))
    requested_path = None
    requested_depth = None
    if isinstance(teacher, Mapping):
        raw_path = teacher.get("path")
        if isinstance(raw_path, str):
            requested_path = Path(raw_path).resolve()
        raw_depth = teacher.get("depth")
        if type(raw_depth) is int:
            requested_depth = raw_depth
    entries: list[dict[str, Any]] = []
    for report_path in sorted(set(candidates), key=str):
        if requested_path is not None and report_path.resolve() != requested_path:
            continue
        report = _read_json(report_path)
        if not isinstance(report, Mapping) or report.get("passed") is not True:
            continue
        depth = report.get("selected_depth")
        if type(depth) is not int or depth not in {3, 4, 5}:
            continue
        if requested_depth is not None and depth != requested_depth:
            continue
        plan = _read_json(report_path.parent / "plan.json")
        if isinstance(plan, Mapping):
            report_manifest = report.get("teacher_manifest_sha256")
            plan_manifest = plan.get("teacher_manifest_sha256")
            if (isinstance(report_manifest, str) and isinstance(plan_manifest, str)
                    and report_manifest != plan_manifest):
                continue
        selected_gate = next((item for item in report.get("depths", [])
                              if isinstance(item, Mapping) and item.get("depth") == depth), None)
        if isinstance(selected_gate, Mapping) and selected_gate.get("passed") is not True:
            continue
        episode_dir = report_path.parent / "calibration" / f"d{depth}"
        rows: dict[int, float] = {}
        valid = True
        try:
            result_paths = sorted(episode_dir.glob("*/result.json"))
        except OSError:
            continue
        for result_path in result_paths:
            raw = _read_json(result_path)
            if not isinstance(raw, Mapping):
                valid = False
                break
            seed, score = raw.get("seed"), raw.get("score")
            if type(seed) is not int or isinstance(score, bool) or not isinstance(score, (int, float)):
                valid = False
                break
            if not math.isfinite(float(score)) or seed in rows:
                valid = False
                break
            if raw.get("source_policy") != "teacher" or raw.get("teacher_depth") != depth:
                valid = False
                break
            if raw.get("stop") not in {"target", "terminal", "capped"}:
                valid = False
                break
            rows[seed] = float(score)
        if not valid or not rows:
            continue
        source = report_path.parent.name
        source_id = "teacher"
        label = "老师校准"
        entries.append({
            "source": source,
            "source_id": source_id,
            "sourceId": source_id,
            "label": label,
            "batch": report_path.parent.name,
            "arm": "teacher",
            "stage": "calibration",
            "checkpoint_id": source_id,
            "games": len(rows),
            "mean_score": statistics.fmean(rows.values()),
            "unique_seeds": len(rows),
            "teacher_depth": depth,
            "reused": False,
            "reused_games": 0,
            "fresh_games": len(rows),
            "_regime": (None, None, None, None),
            "_scores": rows,
            "_sort_key": _batch_sort_key({"batch": report_path.parent.name, "source_id": source_id}),
        })
    entries.sort(key=lambda item: item["_sort_key"], reverse=True)
    return entries


def _public_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "source", "source_id", "sourceId", "label", "batch", "arm", "stage",
        "games", "mean_score", "unique_seeds", "reused", "reused_games",
        "fresh_games", "teacher_depth",
    )
    return {key: entry[key] for key in keys if key in entry}


def _pool(entries: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    if not entries:
        return None
    merged: dict[int, float] = {}
    regime = entries[0].get("_regime")
    for entry in entries:
        if entry.get("_regime") != regime:
            # A different cap, target tile, temperature, or dtype is a
            # different gameplay protocol; do not present it as one mean.
            return None
        scores = entry.get("_scores")
        if not isinstance(scores, Mapping):
            return None
        for seed, score in scores.items():
            if seed in merged and merged[seed] != score:
                # Same checkpoint and seed must replay identically before a
                # pooled mean is claimed.  Do not silently choose a run.
                return None
            merged[int(seed)] = float(score)
    if not merged:
        return None
    return {
        "source": "pooled",
        "source_id": "pooled",
        "sourceId": "pooled",
        "label": f"多轮去重汇总 · {len(merged)}局",
        "batch": "pooled",
        "games": len(merged),
        "mean_score": statistics.fmean(merged.values()),
        "unique_seeds": len(merged),
        "batches": len(entries),
    }


def _build_index(runs_root: Path, teacher: Mapping[str, Any] | None = None) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in _scan_batches(runs_root):
        grouped.setdefault(str(entry["checkpoint_id"]), []).append(entry)
    for key in grouped:
        grouped[key].sort(key=_batch_sort_key, reverse=True)
    grouped["teacher"] = _teacher_entries(runs_root, teacher)
    return grouped


def _latest_entry(entries: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    if not entries:
        return None
    fresh = [entry for entry in entries if entry.get("fresh_games", 0) > 0]
    return fresh[0] if fresh else entries[0]


class PlayEvaluationIndex:
    """One read-only index shared by all model-card summaries."""

    def __init__(self, runs_root: str | Path = "runs", *, teacher: Mapping[str, Any] | None = None):
        self.runs_root = Path(runs_root)
        self._index = _build_index(self.runs_root, teacher)

    def summarize(self, checkpoint_files: Mapping[str, Any] | str) -> dict[str, Any]:
        """Return ``latest``, pooled ``overall`` and all independent batches.

        ``checkpoint_files`` must be the complete five-file pin map from the
        model discovery record, or the literal ``"teacher"``.  Unknown or
        incomplete identities intentionally return empty summaries.
        """

        if isinstance(checkpoint_files, str):
            key = "teacher" if checkpoint_files.lower() == "teacher" else None
        else:
            pins = _normalize_checkpoint_files(checkpoint_files)
            key = checkpoint_id(pins) if pins is not None else None
        entries = self._index.get(key, []) if key is not None else []
        batches = [_public_entry(entry) for entry in entries]
        latest = _latest_entry(entries)
        return {
            "latest": _public_entry(latest) if latest is not None else None,
            "overall": _pool(entries),
            "batches": batches,
        }


__all__ = ["CHECKPOINT_FILE_NAMES", "PlayEvaluationIndex"]

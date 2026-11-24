"""Local model catalog and lazy inference for interactive games.

The workbench deliberately discovers *published* head checkpoints rather than
guessing from directory names.  A checkpoint is usable only when its sealed
trial report, identity, frozen encoder assets, and head digest agree.  The
catalog does not import torch or configure native Kev; those relatively costly
operations remain on the first ``load`` call.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
import threading
import time

from kev2048_release.assets import BASE_DIGEST, FILES, ORIGINAL_DIGESTS
from kev2048_release.data import digest
from kev2048_release.game import legal, nested
from kev2048_release.artifacts import identity, load_plan

if __package__:
    from .wide_teacher import PlayTeacher
else:
    from wide_teacher import PlayTeacher


CHECKPOINT_PINS = {
    **ORIGINAL_DIGESTS,
    "adapter_config.json": "748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854",
    "tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
}
BASE_PINS = {
    "model.safetensors": BASE_DIGEST,
    "config.json": "b90b86f35c8e6925ef74ee04d0e758f0a845c83a42089ad82bbaa948de9b4204",
    "tokenizer.json": "fe000e3ed39ed12b8d2481d527d44f93c65d37e87645d2dcc80d1bf9d50d2927",
    "tokenizer_config.json": "e611fbccc7c29ef3b1cafb1cb7ea548d189968632901d678fd62be68c47885de",
}
ARM = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
SHA = re.compile(r"^[a-f0-9]{64}$")

# Published layouts used by rounds 3–12.  ``result.json`` is retained for the
# old finalists and for round 3/5/6/7 manifests; it is not enough on its own
# unless a candidate record containing a checkpoint is found.
_PUBLISHED_FILES = (
    "candidate.json",
    "candidates.json",
    "gameplay-candidates.json",
    "selection.json",
    "training-result.json",
    "result.json",
)
_REQUIRED_CHECKPOINT_FILES = {"head.pt", *FILES}
_REFRESH_TTL_SECONDS = 8.0


class CandidateSpec(dict):
    """Rich candidate metadata with the old ``(name, trial, token)`` view."""

    def __getitem__(self, key):
        if key == 0:
            return dict.__getitem__(self, "name")
        if key == 1:
            return dict.__getitem__(self, "trial")
        if key == 2:
            return dict.__getitem__(self, "token")
        return dict.__getitem__(self, key)


def read_json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a local JSON object")
    return value


def _read_document(path):
    """Read a publication document, allowing a top-level candidate list."""

    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, (dict, list)):
        raise ValueError("Expected a publication JSON object or list")
    return value


def verify_pins(folder, pins):
    folder = Path(folder)
    for name, sha in pins.items():
        if digest(folder / name) != sha:
            raise ValueError(f"Local model checksum mismatch: {name}")


def _resolve_inside(value, root):
    """Resolve a recorded path and reject paths escaping its run directory."""

    if not isinstance(value, (str, os.PathLike)):
        return None
    try:
        root = Path(root).resolve()
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        path = path.resolve()
        path.relative_to(root)
        return path
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _sealed_resolved(path):
    """Read and verify the identity of a run's resolved configuration."""

    try:
        resolved = read_json(path)
        token = resolved.get("identity")
        if not isinstance(token, str) or not SHA.fullmatch(token):
            return None
        body = {key: value for key, value in resolved.items() if key != "identity"}
        if identity(body) != token:
            return None
        return resolved, token
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _candidate_like(value):
    return (isinstance(value, dict) and isinstance(value.get("checkpoint"), str)
            and isinstance(value.get("checkpoint_files"), dict)
            and isinstance(value.get("trial_result"), str))


def _walk_candidates(value, seen=None):
    """Yield candidate-shaped dictionaries from old nested result manifests."""

    if seen is None:
        seen = set()
    if isinstance(value, dict):
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)
        if _candidate_like(value):
            yield value
        for child in value.values():
            yield from _walk_candidates(child, seen)
    elif isinstance(value, list):
        marker = id(value)
        if marker in seen:
            return
        seen.add(marker)
        for child in value:
            yield from _walk_candidates(child, seen)


def _number(value):
    return value if type(value) is int and value > 0 else None


def _candidate_rows(candidate, report, resolved, root_report=None):
    for source in (candidate, report, resolved, root_report or {}):
        if not isinstance(source, dict):
            continue
        rows = _number(source.get("train_rows"))
        if rows is not None:
            return rows
    match = re.search(r"(?<![0-9])([0-9]{2,6})(?![0-9])", str(candidate.get("id", "")))
    return int(match.group(1)) if match else None


def _teacher_depth(root, rows=None):
    """Read the sealed first training row, when present, for the real teacher depth."""

    data = Path(root) / "data"
    paths = []
    if rows:
        paths.append(data / f"train-{rows}.jsonl")
    paths.append(data / "train.jsonl")
    try:
        paths.extend(sorted(data.glob("train-*.jsonl")))
    except OSError:
        pass
    seen = set()
    for path in paths:
        try:
            path = path.resolve()
            if path in seen or not path.is_file():
                continue
            seen.add(path)
            with path.open("r", encoding="utf-8") as stream:
                row = json.loads(stream.readline())
            teacher = row.get("teacher_identity")
            depth = teacher.get("depth") if isinstance(teacher, dict) else None
            if type(depth) is int and depth in (3, 4, 5):
                return depth
        except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
            continue
    return None


def _training_summary(spec):
    candidate = spec.get("candidate", {})
    report = spec.get("report", {})
    resolved = spec.get("resolved", {})
    root_report = spec.get("root_report", {})
    model_id = str(spec.get("id", candidate.get("id", "")))
    arm = str(report.get("arm") or resolved.get("arm") or candidate.get("arm") or model_id)
    rows = spec.get("train_rows") or _candidate_rows(candidate, report, resolved, root_report)
    hparams = report.get("hyperparameters") or report.get("recipe") or resolved.get("training_recipe") or {}
    if not isinstance(hparams, dict):
        hparams = {}
    kl = hparams.get("kl_weight", report.get("kl_weight", resolved.get("kl_weight")))
    best = report.get("best_epoch") or candidate.get("best_epoch")
    depth = spec.get("teacher_depth")
    if depth is None:
        depth = _teacher_depth(spec.get("root", "."), rows)

    replaced = report.get("replaced_rows")
    stage_counts = report.get("replacement_stage_counts")
    if arm.startswith("onpolicy-") and rows == 4000 and (replaced == 1000 or stage_counts):
        source = "3,000 张教师盘面，加上模型自走的早期和中期各 500 张"
    elif arm.startswith("teacher-") and rows:
        source = f"{rows:,} 张教师盘面"
    elif arm.startswith("head") and rows:
        source = f"{rows:,} 张训练盘面"
    elif rows:
        source = f"{rows:,} 张训练盘面"
    else:
        source = "已封存训练盘面"

    details = [source]
    if depth is not None and ("教师" in source or "老师" in source):
        details.append(f"用深度 {depth} 教师标注")
    details.append("只训练策略头")
    facts = []
    if kl is not None:
        try:
            facts.append(f"KL={float(kl):g}")
        except (TypeError, ValueError):
            facts.append(f"KL={kl}")
    if best is not None:
        facts.append(f"最佳第 {best} 轮")
    if facts:
        details[-1] += "（" + "，".join(facts) + "）"
    return "，".join(details) + "。"


def _candidate_report(candidate, root, token):
    report_path = _resolve_inside(candidate.get("trial_result"), root)
    checkpoint = _resolve_inside(candidate.get("checkpoint"), root)
    if report_path is None or checkpoint is None or not checkpoint.is_dir() or not report_path.is_file():
        return None
    try:
        report = read_json(report_path)
        if report.get("status") != "completed" or report.get("identity") != token:
            return None
        pins = report.get("checkpoint_files")
        if not isinstance(pins, dict) or set(pins) != _REQUIRED_CHECKPOINT_FILES:
            return None
        if any(pins.get(name) != CHECKPOINT_PINS[name] for name in FILES):
            return None
        if not SHA.fullmatch(str(pins.get("head.pt", ""))):
            return None
        recorded = candidate.get("checkpoint_files")
        if recorded is not None and (not isinstance(recorded, dict)
                                     or set(recorded) != _REQUIRED_CHECKPOINT_FILES
                                     or recorded != pins):
            return None
        recorded_result = candidate.get("trial_result_sha256")
        if recorded_result is not None:
            if not SHA.fullmatch(str(recorded_result)) or digest(report_path) != recorded_result:
                return None
        # The head is small and is the identity-bearing artifact.  The large
        # adapter file is checked against the pinned official assets at load;
        # avoiding a repeated 100 MB hash keeps refresh comfortably interactive.
        if not all((checkpoint / name).is_file() for name in _REQUIRED_CHECKPOINT_FILES):
            return None
        if digest(checkpoint / "head.pt") != pins["head.pt"]:
            return None
        return report, report_path, checkpoint
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _normalise_candidate(candidate, experiment, resolved, token, root_report=None, source_file=None):
    """Validate one sealed candidate and return its internal catalog record."""

    if not isinstance(candidate, dict):
        return None
    training_kind = candidate.get("training_kind")
    if training_kind is not None and str(training_kind).lower() not in {
        "head", "head_only", "frozen_encoder_head_only"
    }:
        # In particular, never expose a LoRA checkpoint as a head-only model.
        return None
    for source in (candidate, resolved):
        scope = source.get("model_scope") or source.get("training_scope") or source.get("mode")
        if isinstance(scope, str) and "lora" in scope.lower():
            return None
    candidate_id = candidate.get("id")
    if not isinstance(candidate_id, str):
        trial_name = candidate.get("trial")
        rows = _number(candidate.get("train_rows")) or _number((root_report or {}).get("train_rows"))
        if not isinstance(trial_name, str) or not rows:
            return None
        candidate_id = f"{rows}-{trial_name}"
    if not ARM.fullmatch(candidate_id):
        return None
    checked = _candidate_report(candidate, experiment, token)
    if checked is None:
        return None
    report, report_path, checkpoint = checked
    rows = _candidate_rows(candidate, report, resolved, root_report)
    spec = CandidateSpec({
        "id": candidate_id,
        "name": f"{rows:,} 张 · {candidate_id}" if rows else candidate_id,
        "kind": "model",
        "trial": checkpoint.parent,
        "trial_result": report_path,
        "checkpoint": checkpoint,
        "token": token,
        "report": report,
        "candidate": dict(candidate),
        "resolved": resolved,
        "root_report": root_report or {},
        "root": Path(experiment).resolve(),
        "source_file": Path(source_file).resolve() if source_file else None,
        "train_rows": rows,
        "checkpoint_files": dict(report["checkpoint_files"]),
    })
    spec["teacher_depth"] = _teacher_depth(experiment, rows)
    spec["training_summary"] = _training_summary(spec)
    return spec


def _old_finalist_candidates(experiment, resolved, token, report):
    if report.get("status") != "completed" or report.get("model_scope") != "frozen_encoder_head_only":
        return []
    rows = _number(report.get("train_rows"))
    if not rows or not isinstance(report.get("finalists"), list):
        return []
    # The old rounds sealed the original head digest in resolved-config.
    files = resolved.get("files", {})
    if files.get("checkpoint/head.pt") != ORIGINAL_DIGESTS["head.pt"]:
        return []
    result = []
    for finalist in report["finalists"]:
        if not isinstance(finalist, dict) or not isinstance(finalist.get("trial"), str):
            continue
        arm = finalist["trial"]
        if not ARM.fullmatch(arm):
            continue
        checkpoint = finalist.get("checkpoint")
        if not isinstance(checkpoint, str):
            continue
        # Keep the old explicit path contract; a copied checkpoint must not be
        # silently associated with the original trial.
        trial = Path(experiment) / arm
        expected = (trial / "checkpoint").resolve()
        try:
            if Path(checkpoint).resolve() != expected:
                continue
        except (OSError, RuntimeError):
            continue
        public_id = f"{rows}-{arm}"
        # Round 4 intentionally kept the historical ``2000-lr8e-5-kl015``
        # id.  Give the otherwise hidden round-1 head a stable, distinguishable
        # alias so the two different checkpoint hashes never share scores.
        if experiment.name.startswith("round1-") and arm == "lr8e-5-kl015":
            public_id = f"round1-{public_id}"
        candidate = {
            "id": public_id,
            "trial": arm,
            "checkpoint": str(expected),
            "trial_result": str(trial / "result.json"),
            "training_kind": "head",
            "train_rows": rows,
        }
        spec = _normalise_candidate(candidate, experiment, resolved, token, report, experiment / "result.json")
        if spec is not None:
            result.append(spec)
    return result


def _checkpoint_key(spec):
    try:
        return (spec["report"]["checkpoint_files"]["head.pt"],
                tuple(spec["report"]["checkpoint_files"].get(name) for name in FILES))
    except (KeyError, TypeError):
        return None


def _store_candidate(candidates, spec, *, explicit=False):
    """Store a candidate without aliasing two different heads under one id."""

    model_id = spec["id"]
    current = candidates.get(model_id)
    if current is not None and _checkpoint_key(current) != _checkpoint_key(spec):
        head = spec.get("report", {}).get("checkpoint_files", {}).get("head.pt", "")
        alias = f"{model_id}-{head[:8]}" if SHA.fullmatch(str(head)) else f"{model_id}-alt"
        suffix = 2
        while alias in candidates:
            alias = f"{model_id}-{head[:8]}-{suffix}"
            suffix += 1
        spec["id"] = alias
        spec["name"] = f"{spec['train_rows']:,} 张 · {alias}" if spec.get("train_rows") else alias
        spec["training_summary"] = _training_summary(spec)
        model_id = alias
        current = None
    if current is None or explicit:
        candidates[model_id] = spec


def _discover_candidates(runs):
    runs = Path(runs)
    candidates = {}
    try:
        experiments = sorted((path for path in runs.iterdir() if path.is_dir()), key=lambda p: p.name, reverse=True)
    except OSError:
        return candidates
    for experiment in experiments:
        experiment = experiment.resolve()
        sealed = _sealed_resolved(experiment / "resolved-config.json")
        if sealed is None:
            continue
        resolved, token = sealed
        root_report = {}
        try:
            root_report = read_json(experiment / "result.json")
        except (OSError, ValueError, json.JSONDecodeError):
            pass

        # Preserve the exact old finalists contract (rounds 1 and 2).
        for spec in _old_finalist_candidates(experiment, resolved, token, root_report):
            _store_candidate(candidates, spec)

        for filename in _PUBLISHED_FILES:
            source = experiment / filename
            if not source.is_file():
                continue
            try:
                document = _read_document(source)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            source_identity = document.get("identity") if isinstance(document, dict) else None
            if source_identity is not None and source_identity != token:
                continue
            for candidate in _walk_candidates(document):
                spec = _normalise_candidate(candidate, experiment, resolved, token, root_report, source)
                if spec is not None:
                    # Prefer the explicit publication manifest over duplicate
                    # copies embedded in result/training manifests.
                    _store_candidate(candidates, spec, explicit=filename in {
                        "candidate.json", "candidates.json", "gameplay-candidates.json"
                    })
    def sort_key(item):
        spec = item[1]
        return (spec.get("train_rows") or 0, item[0])
    return dict(sorted(candidates.items(), key=sort_key))


def candidate_models(runs):
    """Return verified candidate specs keyed by their stable public id."""

    return _discover_candidates(runs)


def _teacher_descriptor(runs, teacher_spec):
    if not teacher_spec:
        return None
    folder, depth = teacher_spec
    try:
        for calibration in sorted(Path(runs).glob("*/calibration.json"), reverse=True):
            plan = load_plan(calibration.parent)
            if Path(plan.get("teacher_dir", "")).resolve() == Path(folder).resolve():
                return {"path": str(calibration), "sourceId": "teacher", "depth": depth}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        pass
    return {"sourceId": "teacher", "depth": depth}


def _gameplay_statistics(runs, candidates, teacher_spec):
    """Build one cached score index and attach summaries by sealed head pins."""

    try:
        try:
            from play_evaluations import PlayEvaluationIndex
        except ModuleNotFoundError:
            from workbench.play_evaluations import PlayEvaluationIndex
        index = PlayEvaluationIndex(runs, teacher=_teacher_descriptor(runs, teacher_spec))
    except Exception:
        # Evaluation is an optional presentation layer.  A partially written
        # score manifest or an unavailable score plugin must never prevent the
        # playable head catalog from coming up.
        return {}
    result = {}
    try:
        result["official"] = index.summarize(CHECKPOINT_PINS)
    except (OSError, TypeError, ValueError):
        result["official"] = None
    for model_id, spec in candidates.items():
        try:
            result[model_id] = index.summarize(spec["report"]["checkpoint_files"])
        except (KeyError, OSError, TypeError, ValueError):
            result[model_id] = None
    if teacher_spec:
        try:
            result["teacher"] = index.summarize("teacher")
        except (OSError, TypeError, ValueError):
            result["teacher"] = None
    return result


def teacher_setup(runs, preparation):
    paths = ([Path(preparation) / "calibration.json"] if preparation else
             sorted(Path(runs).glob("*/calibration.json"), reverse=True))
    for path in paths:
        try:
            plan = load_plan(path.parent)
            report = read_json(path)
            depth = report.get("selected_depth")
            if (report.get("passed") is not True or depth not in (3, 4, 5)
                    or report.get("teacher_manifest_sha256") != plan["teacher_manifest_sha256"]):
                continue
            gate = next((item for item in report["depths"] if item.get("depth") == depth), {})
            folder = Path(plan["teacher_dir"])
            if (gate.get("passed") is True and gate.get("completed") == plan["teacher_games"]
                    and gate.get("successes", 0) >= plan["teacher_required_successes"]
                    and (folder / "teacher.dll").is_file()):
                return folder, depth
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    return None


class ModelLibrary:
    """Thread-safe, lazily loaded catalog for the interactive game server."""

    def __init__(self, args, runs):
        self.args = args
        self.runs = Path(runs).resolve()
        self.model, self.tokenizer, self.meta = None, None, None
        self.heads = {}
        self._loaded_head_pins = {}
        self.teacher = None
        self._loaded_teacher_spec = None
        self.configured = False
        # PlaySession can compare snapshots by identity between publications.
        self.models = ()
        self.candidates = {}
        self.teacher_spec = None
        self._evaluations = {}
        self._lock = threading.RLock()
        self._last_refresh = 0.0
        self._last_audit = 0.0
        self._catalog_stamp = None
        self.refresh(force=True)

    def _assets_present(self):
        base = Path(self.args.base)
        checkpoint = Path(self.args.checkpoint)
        return (all((base / name).is_file() for name in BASE_PINS)
                and all((checkpoint / name).is_file() for name in CHECKPOINT_PINS))

    def _public_models(self, candidates, teacher_spec, evaluations):
        with self._lock:
            loaded = set(self.heads)
            old_by_id = {item.get("id"): item for item in self.models if isinstance(item, dict)}
        items = []
        if self._assets_present() or self.model is not None or "official" in loaded:
            items.append(dict(
                id="official",
                name="官方基线",
                kind="model",
                training_summary="官方发布的 Kev 策略，使用原版编码器和决策头。",
                evaluation=copy.deepcopy(evaluations.get("official")),
            ))
            # Most recent published runs are deliberately first.  The official
            # baseline remains first so existing clients retain their default;
            # the remaining entries retain candidate_models' stable order.
            def round_number(item):
                match = re.match(r"round(\d+)(?:-|$)", item[1]["root"].name)
                return int(match[1]) if match else 0

            ordered = sorted(candidates.items(), key=round_number, reverse=True)
            for model_id, spec in ordered:
                if model_id == "onpolicy-4000-kl0":
                    display_name = "自走纠错 · 4,000 张"
                elif model_id == "teacher-4000-kl0":
                    display_name = "纯教师 · 4,000 张"
                else:
                    display_name = spec["name"]
                items.append(dict(
                    id=model_id,
                    name=display_name,
                    kind="model",
                    training_summary=spec["training_summary"],
                    evaluation=copy.deepcopy(evaluations.get(model_id)),
                ))
            # A loaded model remains visible for the lifetime of an active
            # session even if a publisher temporarily removes its manifest.
            for model_id in sorted(loaded - {"official"}):
                if model_id not in candidates and model_id in old_by_id:
                    items.append(old_by_id[model_id])
        if teacher_spec:
            depth = teacher_spec[1]
            items.append(dict(
                id="teacher",
                name=f"搜索老师 · 深度 {depth}",
                kind="teacher",
                training_summary=f"按深度 {depth} 搜索各合法方向，取评分最高的走法。",
                evaluation=copy.deepcopy(evaluations.get("teacher")),
            ))
        elif "teacher" in loaded and "teacher" in old_by_id:
            items.append(old_by_id["teacher"])
        return items

    def _published_files_stamp(self, runs, preparation):
        names = (*_PUBLISHED_FILES, "resolved-config.json", "calibration.json",
                 "gameplay/result.json", "gameplay/resolved-config.json")
        paths = {path for name in names for path in runs.glob("*/" + name)}
        if preparation:
            paths.add(Path(preparation) / "calibration.json")
        with self._lock:
            for spec in self.candidates.values():
                paths.add(spec["trial_result"])
                paths.add(spec["checkpoint"] / "head.pt")
        stamps = []
        for path in sorted(paths):
            try:
                stat = path.stat()
                stamps.append((path, stat.st_mtime_ns, stat.st_size))
            except OSError:
                continue
        return stamps

    def refresh(self, runs=None, *, force=False, evaluations=None, preparation=None):
        """Refresh the published catalog and return a snapshot of its models.

        Calls without explicit inputs are cheap for eight seconds.  ``force``
        is useful after the publisher atomically installs a new candidate.
        Loaded encoder/head/teacher objects are never discarded by a refresh.
        """

        explicit = runs is not None or evaluations is not None or preparation is not None
        now = time.monotonic()
        with self._lock:
            if not force and not explicit and now - self._last_refresh < _REFRESH_TTL_SECONDS:
                return self.models
        root = self.runs if runs is None else Path(runs).resolve()
        prep = preparation if preparation is not None else getattr(self.args, "teacher_preparation", None)
        stamp = self._published_files_stamp(root, prep)
        with self._lock:
            # Periodically recheck assets that may finish copying after their manifest.
            if not force and not explicit and stamp == self._catalog_stamp and now - self._last_audit < 120:
                self._last_refresh = now
                return self.models
        candidates = candidate_models(root) if self._assets_present() else {}
        teacher_spec = teacher_setup(root, prep)
        stats = (dict(evaluations) if isinstance(evaluations, dict)
            else _gameplay_statistics(root, candidates, teacher_spec))
        models = self._public_models(candidates, teacher_spec, stats)
        with self._lock:
            self.runs = root
            self.candidates = candidates
            self.teacher_spec = teacher_spec
            self._evaluations = stats
            snapshot = tuple(models)
            if snapshot != self.models:
                self.models = snapshot
            self._catalog_stamp = stamp
            self._last_refresh = time.monotonic()
            self._last_audit = self._last_refresh
            return self.models

    def set_evaluations(self, evaluations):
        """Install a score-agent mapping without changing loaded weights."""

        return self.refresh(evaluations=evaluations, force=True)

    def load_teacher(self):
        with self._lock:
            spec = self.teacher_spec
            if self.teacher is not None and self._loaded_teacher_spec == spec:
                return
        if spec is None:
            raise ValueError("No calibrated search teacher is available")
        teacher = PlayTeacher(spec[0], depth=spec[1])
        with self._lock:
            self.teacher = teacher
            self._loaded_teacher_spec = spec

    def load_encoder(self):
        with self._lock:
            if self.model is not None:
                return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        from kev2048_release import native

        verify_pins(self.args.base, BASE_PINS)
        verify_pins(self.args.checkpoint, CHECKPOINT_PINS)
        with self._lock:
            if not self.configured:
                native.configure(self.args)
                self.configured = True
            self.tokenizer, self.model, self.meta = native.local_model(
                self.args.base, self.args.checkpoint, self.args.device)
            self.heads["official"] = self.model.head

    @staticmethod
    def _load_trial_report(spec):
        report = read_json(spec["trial_result"])
        token = spec["token"]
        if report.get("status") != "completed" or report.get("identity") != token:
            raise ValueError("Invalid completed candidate identity/status")
        pins = report.get("checkpoint_files")
        if not isinstance(pins, dict) or set(pins) != _REQUIRED_CHECKPOINT_FILES:
            raise ValueError("The candidate checkpoint manifest is incomplete")
        if pins != spec["report"].get("checkpoint_files"):
            raise ValueError("The candidate changed after its catalog entry was published")
        if any(pins.get(name) != CHECKPOINT_PINS[name] for name in FILES):
            raise ValueError("The candidate does not share the official encoder assets")
        for name, sha in pins.items():
            if digest(Path(spec["checkpoint"]) / name) != sha:
                raise ValueError("Candidate checkpoint was modified: " + name)
        # Standard trials additionally seal every evidence artifact.  A few
        # historical rounds used export.json instead of result.json; their
        # candidate manifest and checkpoint hashes still receive the checks
        # above, and no unverified path is loaded.
        if Path(spec["trial_result"]).resolve() == (Path(spec["trial"]) / "result.json").resolve():
            from kev2048_release.artifacts import completed_trial
            verified = completed_trial(Path(spec["trial"]), token)
            if verified is None:
                raise ValueError("Candidate trial is not complete")
            report = verified
        return report

    def load_head(self, name):
        with self._lock:
            spec = self.candidates.get(name)
            pin = spec["report"]["checkpoint_files"]["head.pt"] if spec else None
            if name in self.heads and (spec is None or self._loaded_head_pins.get(name) == pin):
                return
            model_meta = self.meta
            model = self.model
        if spec is None:
            raise ValueError(f"Unknown published model: {name}")
        if model is None or model_meta is None:
            raise RuntimeError("The official encoder must be loaded before a head")
        import torch
        from kev.checkpoint import Meta

        self._load_trial_report(spec)
        meta = Meta.from_dict(torch.load(Path(spec["checkpoint"]) / "head.pt",
                                         map_location="cpu", weights_only=True))
        origin = meta.extra.get("kev2048_experiment", {})
        mode = origin.get("mode")
        if (origin.get("identity") != spec["token"]
                or origin.get("initialization_head_sha256") != ORIGINAL_DIGESTS["head.pt"]
                or (mode is not None and mode != "frozen_encoder_head_only")
                or meta.temperature != 1.0
                or any(getattr(meta, key) != getattr(model_meta, key) for key in (
                    "weights", "head_dim", "option_isolation", "special_embeddings", "weights_dtype",
                ))):
            raise ValueError("Candidate head identity does not agree with the official encoder")
        head = copy.deepcopy(model.head)
        head.load_state_dict(meta.head)
        with self._lock:
            # A refresh may have replaced the catalog while this checkpoint was
            # loading.  The loaded object remains safe for the active session.
            self.heads[name] = head.requires_grad_(False).eval()
            self._loaded_head_pins[name] = pin

    def load(self, names):
        names = list(names)
        if "teacher" in names:
            self.load_teacher()
        neural = [name for name in names if name != "teacher"]
        if neural:
            self.load_encoder()
            for name in neural:
                self.load_head(name)
        return self.decide

    def decide(self, requests):
        results, shared = {}, {}
        with self._lock:
            teacher = self.teacher
            tokenizer, model = self.tokenizer, self.model
            heads = {name: self.heads.get(name) for name, _ in requests if name != "teacher"}
            device = self.args.device
        for name, game in requests:
            if name == "teacher":
                if teacher is None:
                    raise RuntimeError("Teacher has not been loaded")
                started = time.perf_counter()
                values = teacher.values(game.state)
                results[name] = dict(action_values=values, decision_ms=(time.perf_counter() - started) * 1000)
            else:
                if heads.get(name) is None or model is None or tokenizer is None:
                    raise RuntimeError(f"Model has not been loaded: {name}")
                shared.setdefault(game.state, []).append(name)
        if shared:
            import torch
            from kev2048_release.artifacts import official_features

            with torch.no_grad():
                for state, names in shared.items():
                    options = legal(state)
                    started = time.perf_counter()
                    (features,) = official_features(tokenizer, model, [
                        dict(board=nested(state), valid_moves=options)
                    ])
                    if device == "cuda":
                        torch.cuda.synchronize()
                    encode_ms = (time.perf_counter() - started) * 1000 / len(names)
                    for name in names:
                        started = time.perf_counter()
                        probabilities = heads[name](*features).softmax(-1).cpu().tolist()
                        results[name] = dict(
                            probabilities=dict(zip(options, probabilities, strict=True)),
                            decision_ms=encode_ms + (time.perf_counter() - started) * 1000,
                        )
        return results

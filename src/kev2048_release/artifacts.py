"""Shared artifact verification, game termination, and serving feature helpers."""

import hashlib
import json
from pathlib import Path

from .data import digest
from .game import legal


def identity(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def stop_reason(game, cap):
    if max(game.state) >= 2048:
        return "target"
    if not legal(game.state):
        return "terminal"
    if game.turn >= cap:
        return "capped"
    return None


def load_plan(out):
    out = Path(out)
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    lock = json.loads((out / "plan-lock.json").read_text(encoding="utf-8"))
    if digest(out / "plan.json") != lock["plan_sha256"]:
        raise ValueError("Preparation plan changed; use a new run directory")
    if digest(Path(plan["teacher_dir"]) / "manifest.json") != plan["teacher_manifest_sha256"]:
        raise ValueError("Teacher build changed; use a new run directory")
    for name, sha in plan["implementation_sha256"].items():
        source = Path(__file__).with_name(name)
        if not source.is_file() or digest(source) != sha:
            raise ValueError("Preparation implementation changed: " + name)
    return plan


def completed_trial(path, run_identity):
    path = Path(path)
    report_path = path / "result.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("identity") != run_identity or report.get("status") != "completed":
        raise ValueError("Invalid completed trial identity/status")
    for name, sha in report["checkpoint_files"].items():
        if digest(path / "checkpoint" / name) != sha:
            raise ValueError("Completed checkpoint was modified")
    for name, sha in report["artifacts"].items():
        if digest(path / name) != sha:
            raise ValueError("Completed trial evidence was modified: " + name)
    return report


def official_features(tok, model, rows):
    """Capture the exact FP32 head inputs used by official serving inference.

    Qwen3.5's cached-prefix serving path and full-row training encoder have
    slightly different rounding. A trained head can amplify that difference.
    The hook observes inputs only; it never replaces model outputs or weights.
    """
    import torch
    from . import native

    if model.training or model.head.training:
        raise ValueError("Official feature extraction requires evaluation mode")
    captured = []

    def capture(_head, args):
        if len(args) != 2:
            raise ValueError("Unexpected official pointer input signature")
        captured.append(tuple(tensor.detach().clone() for tensor in args))

    handle = model.head.register_forward_pre_hook(capture)
    try:
        with torch.no_grad():
            for row in rows:
                before = len(captured)
                probabilities = model.probs(native.record(tok, model, row))
                if len(captured) != before + 1 or len(probabilities) != 1:
                    raise ValueError("Expected exactly one official pointer call per board")
                q, k = captured[-1]
                if q.ndim != 1 or k.shape != (len(row["valid_moves"]), q.shape[0]):
                    raise ValueError("Official head feature dimensions differ from legal options")
    finally:
        handle.remove()
    return captured

"""Content-addressed storage for the frozen encoder's training features."""

import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from .assets import FILES
from .data import digest


FORMAT_VERSION = 1


def _snapshot_files(folder, ancestors=frozenset()):
    # Directory links are valid for local weights. Do not recurse through cycles.
    target = folder.resolve()
    if target in ancestors:
        return
    ancestors = ancestors | {target}
    for path in folder.iterdir():
        if path.is_dir() and path.name not in (".cache", ".git"):
            yield from _snapshot_files(path, ancestors)
        elif path.is_file():
            yield path


def frozen_inputs(base, checkpoint):
    """Track local snapshot files, excluding the checkpoint's pointer head."""
    base, checkpoint = base.resolve(), checkpoint.resolve()
    files = {f"checkpoint/{name}": checkpoint / name for name in FILES}
    for path in _snapshot_files(base):
        if path != checkpoint / "head.pt" and path.suffix not in (".features", ".tmp"):
            files[f"base/{path.relative_to(base).as_posix()}"] = path

    snapshot = {}
    for name, path in files.items():
        stat = path.stat()
        snapshot[name] = (path, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
    return snapshot


def _json_digest(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def cache_key(inputs, lock, data, encoder):
    files = {}
    hardlinks = {}
    for name, (path, device, inode, *stamp) in sorted(inputs.items()):
        # The pinned snapshot can expose the same weight through two hard links.
        file_id = (device, inode, *stamp) if inode else path
        if file_id not in hardlinks:
            hardlinks[file_id] = digest(path)
        files[name] = hardlinks[file_id]
    return _json_digest({
        "format": FORMAT_VERSION,
        "files": files,
        "data": {name: lock["splits"][name]["sha256"] for name in ("train", "dev")},
        "records": _json_digest(data),
        "encoder": encoder,
    })


def read_features(path, key, options_per_row, hidden_size, device):
    import torch
    from safetensors import SafetensorError
    from safetensors.torch import load

    q_shape = (len(options_per_row), hidden_size)
    k_shape = (sum(options_per_row), hidden_size)
    # Bound reads of malformed files by the FP32 payload size plus the tensor header.
    byte_limit = 4 * (q_shape[0] + k_shape[0]) * hidden_size + 16384
    try:
        with path.open("rb") as stream:
            header = stream.readline(131)
            payload = stream.read(byte_limit + 1)
    except FileNotFoundError:
        return None, "missing"

    if len(payload) > byte_limit or header != _header(key, payload):
        return None, "invalid checksum or key"
    try:
        tensors = load(payload)
    except SafetensorError:
        return None, "invalid tensor file"
    if tensors.keys() != {"queries", "options"}:
        return None, "invalid tensor names"
    queries, options = tensors["queries"], tensors["options"]
    if queries.shape != q_shape or options.shape != k_shape:
        return None, "invalid tensor shapes"
    if any(t.dtype != torch.float32 or not torch.isfinite(t).all() for t in (queries, options)):
        return None, "invalid FP32 features"

    queries = queries.to(device).unbind()
    options = options.to(device).split(options_per_row)
    return list(zip(queries, options, strict=True)), None


def _header(key, payload):
    return f"{key} {hashlib.sha256(payload).hexdigest()}\n".encode("ascii")


def _publish(tmp, path):
    # On Windows a concurrent reader can briefly prevent atomic replacement.
    for delay in (0.025, 0.05, 0.1):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in (5, 32, 33):
                raise
            time.sleep(delay)
    tmp.replace(path)


def write_features(path, key, features, options_per_row, hidden_size):
    import torch
    from safetensors.torch import save

    for (query, options), count in zip(features, options_per_row, strict=True):
        if query.shape != (hidden_size,) or options.shape != (count, hidden_size):
            raise ValueError("Unexpected training feature shape")
    payload = save({
        "queries": torch.stack([query for query, _ in features]).cpu(),
        "options": torch.cat([options for _, options in features]).cpu(),
    })
    header = _header(key, payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    tmp = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(header)
            stream.write(payload)
        _publish(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)

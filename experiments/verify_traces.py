"""Normalize project traces into a portable schema or verify frozen releases (Python 3.9+)."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BOARDS = {0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096}
ARMS = {"d4": ("identity", "d4", "greedy"), "wide": ("v2", "hard"), "stock": ("v2", "candidate", "greedy")}
CAPS = {"d4": 240, "wide": 400, "stock": 400}
ACTIONS = ("up", "down", "left", "right")


def read_rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def flat(board):
    if len(board) == 4 and isinstance(board[0], list):
        return [tile for row in board for tile in row]
    return board


def slide(board, action):
    if action not in ACTIONS:
        raise ValueError("invalid direction")
    after = board.copy()
    gain = 0
    vertical = action in ("up", "down")
    for lane in range(4):
        cells = [4 * i + lane if vertical else 4 * lane + i for i in range(4)]
        if action in ("down", "right"):
            cells.reverse()
        source = iter(board[i] for i in cells if board[i])
        merged, carry = [], next(source, 0)
        for tile in source:
            if carry == tile:
                carry *= 2
                merged.append(carry)
                gain += carry
                carry = 0
            else:
                if carry:
                    merged.append(carry)
                carry = tile
        if carry:
            merged.append(carry)
        for index, tile in zip(cells, merged + [0] * (4 - len(merged))):
            after[index] = tile
    return after, gain


def game_over(board):
    return all(slide(board, action)[0] == board for action in ACTIONS)


def spawn_diff(before, after, action):
    slid, gain = slide(before, action)
    diff = [i for i, (old, new) in enumerate(zip(slid, after)) if old != new]
    if slid == before or len(diff) != 1 or slid[diff[0]] != 0 or after[diff[0]] not in (2, 4):
        raise ValueError("invalid move or spawn transition")
    return diff[0], gain


def normalized(rows, experiment):
    out = []
    for row in rows:
        before, after = flat(row["before"]), flat(row["after"])
        spawn, _ = spawn_diff(before, after, row["action"])
        # Only these named numeric/game fields are exported; raw runtime/config
        # paths, model locations and other metadata never leave the original logs.
        item = {
            "seed": row["seed"],
            "policy": row["policy"],
            "turn": row["turn"],
            "action": row["action"],
            "before": before,
            "after": after,
            "spawn_index": spawn,
            "score": row["score"],
            "max_tile": row["tile"],
            "stop": "terminal" if row["terminal"] else "capped" if row["turn"] == CAPS[experiment] else None,
        }
        probs = row.get("probabilities") or row.get("probs")
        if probs:
            item["choice_probabilities"] = probs
        out.append(item)
    return out


def verify(rows, experiment):
    states, openings = {}, {}
    required = {"seed", "policy", "turn", "action", "before", "after", "spawn_index", "score", "max_tile", "stop"}
    for row in rows:
        if set(row) - required - {"choice_probabilities"} or required - set(row):
            raise ValueError("invalid trace schema")
        seed, policy, turn = row["seed"], row["policy"], row["turn"]
        if type(seed) is not int or type(turn) is not int or policy not in ARMS[experiment] or not 1 <= turn <= CAPS[experiment]:
            raise ValueError("unexpected seed, policy or turn")
        before, after = row["before"], row["after"]
        if any(len(x) != 16 or any(type(n) is not int or n not in BOARDS for n in x) for x in (before, after)):
            raise ValueError("invalid tiles")
        prev = states.get((seed, policy))
        if prev is None:
            if turn != 1 or before.count(0) != 14 or tuple(x for x in before if x) not in ((2, 2), (2, 4), (4, 2), (4, 4)):
                raise ValueError("invalid opening")
            if openings.setdefault(seed, before) != before:
                raise ValueError("different openings within seed")
        elif prev["turn"] + 1 != turn or prev["after"] != before or prev["stop"] is not None:
            raise ValueError("invalid inter-turn transition")
        spawn, gain = spawn_diff(before, after, row["action"])
        if row["spawn_index"] != spawn or row["score"] != gain + (prev["score"] if prev else 0) or row["max_tile"] != max(after):
            raise ValueError("incorrect spawn, score or tallest tile")
        expected = "terminal" if game_over(after) else "capped" if turn == CAPS[experiment] else None
        if expected != row["stop"]:
            raise ValueError("incorrect stop")
        probs = row.get("choice_probabilities")
        if policy != "greedy":
            legal = {a for a in ACTIONS if slide(before, a)[0] != before}
            if (not isinstance(probs, dict) or set(probs) != legal
                    or any(not math.isfinite(p) or p < 0 or p > 1 for p in probs.values())
                    or abs(sum(probs.values()) - 1.0) >= 1e-5
                    or probs[row["action"]] != max(probs.values())):
                raise ValueError("invalid native choice probabilities")
        elif probs:
            raise ValueError("greedy run does not report native probabilities")
        states[(seed, policy)] = row
    if (len(openings) != {"d4": 4, "wide": 10, "stock": 4}[experiment]
            or {(seed, a) for seed in openings for a in ARMS[experiment]} != set(states)
            or any(not r["stop"] for r in states.values())):
        raise ValueError("incomplete or unexpected experiment")
    endpoints = [
        {key: row[key] for key in ("seed", "policy", "turn", "score", "max_tile", "stop")}
        for _, row in sorted(states.items())
    ]
    return {"rows": len(rows), "endpoints": endpoints}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_release():
    manifest = json.loads((ROOT / "SHA256.json").read_text(encoding="utf-8"))
    for name, sha in manifest.items():
        if digest(ROOT / name) != sha:
            raise ValueError(f"Checksum mismatch: {name}")
    for experiment in ARMS:
        result = verify(read_rows(ROOT / "traces" / f"{experiment}.jsonl"), experiment)
        summary = json.loads((ROOT / "results" / f"{experiment}.json").read_text(encoding="utf-8"))
        if summary["rows"] != result["rows"] or summary["endpoints"] != result["endpoints"]:
            raise ValueError(f"Summary discrepancy: {experiment}")
        print(f"{experiment}: verified {result['rows']} moves, {len(result['endpoints'])} endpoints")
    print(f"Verified SHA-256 checksums for {len(manifest)} published trace/result files.")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cmd = cli.add_subparsers(dest="task", required=True)
    cmd.add_parser("verify")
    exp = cmd.add_parser("normalize", help="project a compatible original log into the release schema")
    exp.add_argument("experiment", choices=ARMS)
    exp.add_argument("source", type=Path)
    exp.add_argument("destination", type=Path)
    exp.add_argument("--extra-source", type=Path, action="append", default=[])
    args = cli.parse_args()
    if args.task == "verify":
        verify_release()
    else:
        rows = read_rows(args.source)
        for source in args.extra_source:
            rows.extend(read_rows(source))
        result = normalized(rows, args.experiment)
        verify(result, args.experiment)
        args.destination.write_bytes(("".join(json.dumps(x, separators=(",", ":")) + "\n" for x in result)).encode("utf-8"))
        print(f"{len(result)} completed moves exported to {args.destination}")


if __name__ == "__main__":
    main()

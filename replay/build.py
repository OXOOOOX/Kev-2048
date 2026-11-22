"""Rebuild the dependency-free HTML replay from the documented trace and template."""
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
from verify_traces import read_rows, verify  # noqa: E402


def build():
    path = HERE.parent / "experiments" / "traces" / "d4.jsonl"
    rows = read_rows(path)
    verify(rows, "d4")
    arms = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row["policy"] in ("identity", "d4"):
            arms[row["seed"]][row["policy"]].append(row)
    seeds = []
    for seed, policies in sorted(arms.items()):
        left, right = policies["identity"], policies["d4"]
        entry = {"seed": seed, "start": left[0]["before"]}
        for policy in ("identity", "d4"):
            moves = policies[policy]
            entry[policy] = {
                "end": moves[-1]["stop"],
                "frames": [[r["after"], r["score"], r["action"], r["spawn_index"]] for r in moves],
            }
        entry["divergence"] = next((i for i, (a, b) in enumerate(zip(left, right), 1) if a["action"] != b["action"]), None)
        if entry["divergence"] is not None:
            idx = entry["divergence"] - 1
            if left[idx]["before"] != right[idx]["before"]:
                raise ValueError("trajectory diverged before the first different action")
        seeds.append(entry)
    page = (HERE / "template.html").read_text(encoding="utf-8")
    encoded = json.dumps(seeds, ensure_ascii=False, separators=(",", ":"))
    if page.count("/*__SERIALIZED_DATA__*/") != 1 or "</script" in encoded.lower():
        raise ValueError("incorrect or unsafe replay template")
    result = page.replace("/*__SERIALIZED_DATA__*/", encoded).encode("utf-8")
    output = HERE / "2048-replay.html"
    output.write_bytes(result)
    print(f"Built {output.name}: {len(seeds)} paired seeds, SHA-256 {hashlib.sha256(result).hexdigest()}")


if __name__ == "__main__":
    build()

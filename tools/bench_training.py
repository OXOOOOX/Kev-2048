"""Compare CPU and CUDA with the same offline native Kev training preset."""
import argparse
import json
from pathlib import Path
from statistics import median
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PHASES = (
    "initial_model_load_seconds", "train_encoding_seconds", "head_fit_seconds",
    "checkpoint_save_seconds", "development_evaluation_seconds", "official_reload_seconds",
)


def options():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=["smoke", "stock"], default="smoke")
    p.add_argument("--base", type=Path, default=ROOT / "assets" / "qwen-base")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "assets" / "official-kev")
    p.add_argument("--data", type=Path, default=ROOT / "data" / "stock_scale")
    p.add_argument("--out", type=Path, required=True, help="New output directory, preferably under runs/")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--repeats", type=int, default=1, help="Runs per device; even repeats alternate CPU/CUDA order")
    p.add_argument("--dry-run", action="store_true", help="Print the command plan without importing ML dependencies or creating files")
    args = p.parse_args()
    if min(args.repeats, args.threads) < 1:
        p.error("--repeats and --threads must be positive")
    return args


def plan(args):
    for number in range(1, args.repeats + 1):
        order = ("cpu", "cuda") if number % 2 else ("cuda", "cpu")
        for device in order:
            name = f"{number:02}-{device}"
            tag = dict(name=name, repeat=number, device=device)
            argv = [
                sys.executable, "-m", "kev2048_release.cli", "train", "--preset", args.preset,
                "--base", str(args.base.resolve()), "--checkpoint", str(args.checkpoint.resolve()),
                "--data", str(args.data.resolve()), "--out", str(args.out.resolve() / name),
                "--threads", str(args.threads), "--device", device,
            ]
            yield tag, argv


def record(path, item):
    path.write_text(json.dumps(item, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def benchmark(args, runner=subprocess.run, clock=time.perf_counter):
    commands = list(plan(args))
    if args.dry_run:
        for _, argv in commands:
            print(json.dumps(argv, ensure_ascii=False))
        return 0
    args.out.mkdir(parents=True, exist_ok=False)
    file = args.out / "benchmark.json"
    runs = []
    report = dict(preset=args.preset, threads=args.threads, repeats=args.repeats, completed=False, runs=runs)
    reference = None
    for tag, argv in commands:
        name = tag["name"]
        with (args.out / (name + ".log")).open("wb") as log:
            print(f"{name}: see {name}.log", flush=True)
            started = clock()
            result = runner(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
            elapsed = clock() - started
        info = dict(**tag, process_wall_seconds=elapsed, returncode=result.returncode)
        runs.append(info)
        if result.returncode:
            report["error"] = f"{name} failed; inspect {name}.log. No speed ratio reported."
            record(file, report)
            print(report["error"], file=sys.stderr)
            return 1
        try:
            data = json.loads((args.out / name / "result.json").read_text(encoding="utf-8"))
            if (data["preset"], data["device"], data["dtype"]) != (args.preset, tag["device"], "fp32"):
                raise ValueError("unexpected preset, device or precision")
            protocol = {k: data[k] for k in ("train", "dev", "fixture_manifest", "initialization", "initial_head_sha256")}
            if reference is not None and protocol != reference:
                raise ValueError("train/dev fixture or initialization mismatch across runs")
            reference = protocol
            info["timing"] = {k: data["timing"][k] for k in PHASES}
            if not all(isinstance(v, (int, float)) and 0 <= v <= elapsed for v in info["timing"].values()):
                raise ValueError("invalid phase timing")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            report["error"] = f"{name}: invalid results: {exc}. No speed ratio reported."
            record(file, report)
            print(report["error"], file=sys.stderr)
            return 1
        record(file, report)
    report["completed"] = True
    report["protocol"] = reference
    report["median_wall_seconds"] = {
        d: median(r["process_wall_seconds"] for r in runs if r["device"] == d) for d in ("cpu", "cuda")
    }
    report["median_phase_seconds"] = {
        d: {phase: median(r["timing"][phase] for r in runs if r["device"] == d) for phase in PHASES}
        for d in ("cpu", "cuda")
    }
    if report["median_wall_seconds"]["cuda"] > 0:
        report["wall_speedup_cpu_over_cuda"] = report["median_wall_seconds"]["cpu"] / report["median_wall_seconds"]["cuda"]
    record(file, report)
    print(f"Comparison saved: {file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(benchmark(options()))

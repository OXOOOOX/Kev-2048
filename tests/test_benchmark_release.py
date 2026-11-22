import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("bench_training", ROOT / "tools" / "bench_training.py")
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class BenchmarkTest(unittest.TestCase):
    def make_args(self, root, repeats=1, dry_run=False):
        return argparse.Namespace(
            preset="smoke", base=root / "base with space", checkpoint=root / "checkpoint with space",
            data=root / "data", out=root / "runs with space", threads=2, repeats=repeats,
            dry_run=dry_run,
        )

    def fake(self, failure=None, wrong_sha=False):
        def run(argv, **kwargs):
            out = Path(argv[argv.index("--out") + 1])
            device = argv[argv.index("--device") + 1]
            if failure == device:
                return SimpleNamespace(returncode=7)
            out.mkdir()
            data = dict(
                preset="smoke", device=device, dtype="fp32", train=8, dev=8,
                fixture_manifest={"train": "train-hash", "dev": "dev-hash"},
                initialization="substitute_local_checkpoint",
                initial_head_sha256="different" if wrong_sha and device == "cuda" else "same",
                timing={k: 0.1 for k in BENCH.PHASES},
            )
            (out / "result.json").write_text(json.dumps(data))
            return SimpleNamespace(returncode=0)
        return run

    def test_dry_run_with_spaces_and_alternating_order(self):
        with tempfile.TemporaryDirectory() as td:
            args = self.make_args(Path(td), repeats=2, dry_run=True)
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(BENCH.benchmark(args, lambda *_, **__: self.fail("dry-run called runner")), 0)
            commands = [json.loads(s) for s in buf.getvalue().splitlines()]
            self.assertEqual([x[-1] for x in commands], ["cpu", "cuda", "cuda", "cpu"])
            self.assertEqual(commands[0][commands[0].index("--base") + 1], str(args.base))
            self.assertFalse(args.out.exists())

    def test_routes_same_protocol_and_reports_separate_medians(self):
        with tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()):
            args = self.make_args(Path(td), repeats=2)
            timings = iter((0, 12, 12, 18, 18, 22, 22, 32))
            self.assertEqual(BENCH.benchmark(args, self.fake(), lambda: next(timings)), 0)
            r = json.loads((args.out / "benchmark.json").read_text())
            self.assertTrue(r["completed"])
            self.assertEqual(r["median_wall_seconds"], {"cpu": 11.0, "cuda": 5.0})
            self.assertEqual(r["wall_speedup_cpu_over_cuda"], 2.2)
            self.assertEqual(r["median_phase_seconds"]["cpu"]["initial_model_load_seconds"], 0.1)

    def test_does_not_calculate_speedup_for_failure_or_different_head(self):
        for bad in ({"failure": "cuda"}, {"wrong_sha": True}):
            with self.subTest(bad=bad), tempfile.TemporaryDirectory() as td, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                args = self.make_args(Path(td))
                times = iter((0, 8, 8, 11))
                self.assertEqual(BENCH.benchmark(args, self.fake(**bad), lambda: next(times)), 1)
                r = json.loads((args.out / "benchmark.json").read_text())
                self.assertFalse(r["completed"])
                self.assertIn("No speed ratio", r["error"])
                self.assertNotIn("wall_speedup_cpu_over_cuda", r)


if __name__ == "__main__":
    unittest.main()

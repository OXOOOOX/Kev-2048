"""Exercise automatic discovery with dynamic runs, mixed stages and archived summaries."""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from run_discovery import RunDiscovery
from training_discovery import TrainingDiscovery


class RunDiscoveryTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve() / "runs"
        self.discovery = RunDiscovery(self.root, interval=0)

    def write(self, relative, value):
        path = self.root / relative
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(json.dumps(value) + "\n", encoding="utf-8")
        return path

    def progress(self, rel, phase, **details):
        return self.write(rel, dict(phase=phase, timestamp=datetime.now(UTC).isoformat(), status="running", **details))

    def sources(self):
        return {path: (key, kind) for key, _, kind, path in self.discovery.sources()}

    def test_new_run_and_nested_stages_are_identified_by_log_content(self):
        self.assertEqual(self.discovery.sources(), ())
        train = self.progress("new/progress.jsonl", "something_new_head_fit", train_rows=150)
        data = self.progress("new/data/progress.jsonl", "training_episode_completed", seed=25, tile=256, turns=10)
        inference = self.progress("arbitrary/alternate/progress.jsonl", "another_phase", kind="inference")
        result = self.write("past/training/result.json", dict(train_rows=150, dev_rows=30, completed_epochs=2))
        phases = self.sources()
        self.assertEqual({p: k for p, (_, k) in phases.items()},
                         {train:"training", data:"inference", inference:"inference", result:"training"})
        # Growing and replacing the primary file retain the identity users bookmarked.
        self.progress("new/progress.jsonl", "optimizer", kind="training")
        self.assertEqual(phases[train], self.sources()[train])
        old = {p: k for k, _, p in TrainingDiscovery(self.root).sources()}
        self.assertEqual(old[train], phases[train][0])

    def test_gameplay_uses_the_directory_and_other_evaluations_keep_the_journal(self):
        game = self.progress("project/gameplay/progress.jsonl", "loading_model")
        manifest = dict(schema_version=1, seeds=[11], checkpoints={str(i): {} for i in range(5)},
                        config=dict(seed_group_size=4), notes="x" * 70000)
        self.write("project/gameplay/resolved-config.json", manifest)
        other = self.progress("another/progress.jsonl", "whatever", kind="evaluation")
        self.assertEqual({p: k for p, (_, k) in self.sources().items()}, {game.parent:"gameplay", other:"evaluation"})
        self.assertTrue(self.discovery.safe_progress(game))
        self.assertEqual([s[3] for s in RunDiscovery(self.root, excluded=(game,)).sources()], [other])
        self.assertEqual([s[3] for s in RunDiscovery(self.root, excluded=(game.parent,)).sources()], [other])

    def test_all_history_remains_and_aggregate_benchmarks_are_discovered(self):
        expected = {self.progress(f"train-{index}/progress.jsonl", "optimizer", kind="training") for index in range(15)}
        benchmark = self.write("speed/benchmark.json", dict(completed=True, preset="stock", repeats=3,
                                                              protocol={}, runs=[], median_wall_seconds={}))
        existing = self.write("cache/comparison.json", dict(runs={}, checkpoints_identical=True, process_speedup=2))
        sources = self.sources()
        self.assertEqual(set(sources), expected | {benchmark, existing})
        self.assertEqual(sources[benchmark][1], "benchmark")
        self.assertEqual(sources[existing][1], "benchmark")

    def test_grouped_candidates_can_publish_separate_logs_or_fitting_results(self):
        fit = dict(completed_epochs=4, best_epoch=3, optimizer_steps=40, status="completed")
        saved = self.write("run/arms/candidate/result.json", fit)
        current = self.progress("run/trials/fresh/progress.jsonl", "optimizer", kind="training")
        ignored = self.write("run/gameplay/1/result.json", fit)
        self.assertEqual(set(self.sources()), {saved, current})
        self.assertTrue(self.discovery.safe_progress(saved))
        self.assertFalse(self.discovery.safe_progress(ignored))

    def test_dated_cli_runs_and_final_training_artifacts_share_the_source_identity(self):
        training = dict(candidate_head_sha256="f" * 64, train=40, dev=20,
                        metrics={"all": {}}, timing={"head_fit_seconds": 1.25})
        archive = self.write("2026-11-23/run-train/artifacts/result.json", training)
        undated = self.write("another-train/artifacts/result.json", training)
        benchmark = self.write("2026-11-23/run-bench/benchmark.json",
                               dict(preset="smoke", repeats=2, completed=False, runs=[]))
        ignored = self.write("2026-11-23/run-train/artifacts/checkpoint/result.json", training)
        self.write("2026-11-23/run-train/artifacts/identity.json", training)
        expected = {archive: "training", undated: "training", benchmark: "benchmark"}
        first = self.sources()
        self.assertEqual({p: kind for p, (_, kind) in first.items()}, expected)
        self.assertTrue(self.discovery.safe_progress(archive))
        self.assertFalse(self.discovery.safe_progress(ignored))
        journal = self.write("2026-11-23/run-train/progress.jsonl",
                             dict(timestamp=datetime.now(UTC).isoformat(), phase="starting", status="loading", elapsed_seconds=0))
        active = self.sources()
        self.assertEqual(active[journal], first[archive])
        self.assertNotIn(archive, active)
        self.assertTrue(self.discovery.safe_progress(journal))
        explicit = RunDiscovery(self.root, excluded=(archive,), interval=0)
        self.assertEqual({s[3] for s in explicit.sources()}, {undated, benchmark})

    def test_skips_single_games_dataset_files_and_already_observed_sources(self):
        outer = self.progress("training/progress.jsonl", "optimizer", kind="training")
        self.write("training/training/result.json", dict(train_rows=150, dev_rows=30, completed_epochs=2))
        self.write("training/1234/result.json", dict(kind="evaluation", seed=1234, max_tile=512, score=1290))
        self.write("group/arm/result.json", dict(kind="evaluation", seed=1234, max_tile=512, score=1290))
        self.write("group/identity.json", dict(kind="inference", status="completed"))
        self.progress("training/episodes/progress.jsonl", "training_episode_completed")
        self.progress("training.progress.jsonl", "starting")
        assessment = self.write("assessment/result.json", dict(kind="evaluation", status="completed"))
        self.assertEqual(set(self.sources()), {outer, assessment})
        self.progress("assessment/progress.jsonl", "completed", kind="evaluation")
        self.assertEqual([s[3] for s in RunDiscovery(self.root, excluded=(assessment,)).sources()], [outer])

    def test_rechecks_replaced_or_linked_inputs(self):
        path = self.progress("model/training/progress.jsonl", "optimizer", kind="training")
        outside = self.root.parent/"outside.jsonl"
        path.rename(outside)
        try:
            os.link(outside, path)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(str(exc))
        self.assertFalse(self.discovery.safe_progress(path))
        self.assertEqual(self.discovery.sources(), ())
        self.assertFalse(self.discovery.safe_progress(outside))

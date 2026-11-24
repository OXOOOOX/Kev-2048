"""Test bounded discovery of standard training and wrapper journals."""

from datetime import UTC, datetime
import json
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from training_discovery import MAX_TRAINING_SOURCES, TrainingDiscovery


def journal(path, phase, status="running", **fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(timestamp=datetime.now(UTC).isoformat(), status=status, phase=phase, **fields)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row) + "\n")
    return path


def resolved(path, **fields):
    (path.parent / "resolved-config.json").write_text(json.dumps(fields), encoding="utf-8")


class TrainingDiscoveryTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.root = (Path(workspace.name) / "runs").resolve()

    def test_creates_a_stable_source_for_a_run_added_after_start(self):
        discovery = TrainingDiscovery(self.root, interval=0)
        self.assertEqual(discovery.sources(), ())
        path = journal(self.root / "test.v2_训练" / "progress.jsonl", "verify_data")
        [(source_id, name, event_path)] = discovery.sources()
        self.assertRegex(source_id, r"^training-[0-9a-f]{16}$")
        self.assertEqual(name, "test v2_训练")
        self.assertEqual(event_path, path)
        self.assertTrue(discovery.safe_progress(path))
        journal(path, "completed", "completed")
        self.assertEqual(discovery.sources(), ((source_id, name, event_path),))

    def test_recognizes_resolved_training_and_ignores_eval_or_nested_logs(self):
        improvement = journal(self.root / "round3-improve-20261123" / "progress.jsonl", "completed", "completed")
        resolved(improvement, config=dict(head_reference="ref", cache_run="cache", lora_planning_seconds=60),
                 diagnostic_indices_sha256="1" * 64)
        experiment = journal(self.root / "future-head" / "progress.jsonl", "loading_model")
        resolved(experiment, training_scope="pointer_head_only", config=dict(train_file="train.jsonl", trials=[{}]))
        evaluation = journal(self.root / "new-eval" / "progress.jsonl", "loading_model")
        resolved(evaluation, config=dict(lora_planning_seconds=60, official="ref"), diagnostic_indices_sha256="2" * 64)
        journal(evaluation, "gameplay")
        nested = journal(self.root / "new-eval" / "sub" / "progress.jsonl", "head_fit")
        native = journal(self.root / "stock-progress.progress.jsonl", "starting", "loading")
        journal(self.root / "inference.progress.jsonl", "native_inference")
        journal(self.root / "unmarked" / "progress.jsonl", "completed", "completed")
        journal(self.root / "explicit-schema" / "progress.jsonl", "initializing", "loading", kind="training")

        discovery = TrainingDiscovery(self.root, excluded=(native,), interval=0)
        rows = {path: name for _, name, path in discovery.sources()}
        self.assertEqual(rows, {improvement: "第三轮 LoRA 训练", experiment: "future-head",
                                self.root / "explicit-schema" / "progress.jsonl": "explicit-schema"})
        self.assertFalse(discovery.safe_progress(nested))
        self.assertFalse(discovery.safe_progress(native))
        self.assertIn(("stock-progress", native), [(name, path) for _, name, path in TrainingDiscovery(self.root).sources()])

    def test_respects_scan_interval_and_recent_run_limit(self):
        with patch("training_discovery.time.monotonic", return_value=1) as clock:
            discovery = TrainingDiscovery(self.root, interval=2)
            self.assertEqual(discovery.sources(), ())
            paths = []
            for index in range(MAX_TRAINING_SOURCES + 2):
                path = journal(self.root / f"run-{index}" / "progress.jsonl", "head_fit")
                os.utime(path, (1_700_000_000 + index,) * 2)
                paths.append(path)
            clock.return_value = 2.5
            self.assertEqual(discovery.sources(), ())
            clock.return_value = 3
            self.assertEqual([row[2] for row in discovery.sources()], list(reversed(paths[-MAX_TRAINING_SOURCES:])))

    def test_native_journal_replaces_the_matching_cache_wrapper(self):
        discover=TrainingDiscovery(self.root,interval=0)
        wrapper=journal(self.root / 'stock.progress.jsonl','starting','loading')
        self.assertEqual([path for _,_,path in discover.sources()],[wrapper])
        native=journal(self.root / 'stock' / 'progress.jsonl','optimizer',kind='training',schema='kev2048.training.v1')
        journal(wrapper,'train_cache')
        self.assertEqual([path for _,_,path in discover.sources()],[native])

    def test_full_first_batch_and_multistage_training_entrypoints(self):
        sample=dict(board=[[2,16,256,1024]]*4,best_actions=['left','right'],stage='middle',
                    source_policy='perturbed',candidate=dict(choice='left',probabilities={
                        move:.25 for move in ('up','down','left','right')}))
        samples=[dict(sample,row=index+1) for index in range(64)]
        large=journal(self.root/'first-batch'/'progress.jsonl','optimizer',kind='training',
                      schema='kev2048.training.v1',trial='head',batch_size=64,samples=samples)
        self.assertGreater(large.stat().st_size,16*1024)
        expected={large}
        for phase in ('prepare_order_features','scale_inputs','retest_head_fit','head_fit'):
            expected.add(journal(self.root/phase/'progress.jsonl',phase))
        evaluation=journal(self.root/'checkpoint-only'/'progress.jsonl','verify_inputs')
        journal(evaluation,'checkpoint_verified')
        journal(evaluation,'board_screening')
        journal(self.root/'explicit-eval'/'progress.jsonl','verify_data',kind='evaluation')
        self.assertEqual({path for _,_,path in TrainingDiscovery(self.root).sources()},expected)

    def test_refuses_linked_journals_and_junction_like_paths(self):
        outside = journal(self.root.parent / "outside" / "progress.jsonl", "lora_fit")
        hardlink = self.root / "hardlink" / "progress.jsonl"
        hardlink.parent.mkdir(parents=True)
        try:
            os.link(outside, hardlink)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(str(exc))
        discovery = TrainingDiscovery(self.root, interval=0)
        self.assertFalse(discovery.safe_progress(hardlink))
        self.assertEqual(discovery.sources(), ())
        symlink_run = self.root / "symlinked-run"
        try:
            symlink_run.symlink_to(outside.parent, target_is_directory=True)
        except (OSError, NotImplementedError):
            return
        self.assertFalse(discovery.safe_progress(symlink_run / "progress.jsonl"))
        self.assertEqual(discovery.sources(), ())

    def test_real_round_three_is_recognized_when_available(self):
        root = Path(__file__).resolve().parents[1] / "runs"
        journal_path = root / "round3-improve-20261123" / "progress.jsonl"
        if not journal_path.is_file():
            self.skipTest("Third-round local training artifacts are not installed")
        path_to_name = {path: name for _, name, path in TrainingDiscovery(root).sources()}
        self.assertEqual(path_to_name[journal_path], "第三轮 LoRA 训练")
        self.assertNotIn(root / "gameplay-dev-head2000-20261123" / "progress.jsonl", path_to_name)
        for name in path_to_name.values():
            self.assertIsNotNone(re.fullmatch(r"[\w -]{1,80}", name))


if __name__ == "__main__":
    unittest.main()

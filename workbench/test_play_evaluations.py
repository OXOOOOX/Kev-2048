"""Unit and read-only evidence tests for the workbench gameplay index."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

try:
    from play_evaluations import PlayEvaluationIndex
except ModuleNotFoundError:  # Running the test module from the repository root.
    from workbench.play_evaluations import PlayEvaluationIndex


def pins(head: str = "a"):
    return {
        "head.pt": head * 64,
        "adapter_config.json": "b" * 64,
        "adapter_model.safetensors": "c" * 64,
        "tokenizer_config.json": "d" * 64,
        "tokenizer.json": "e" * 64,
    }


def row(arm, seed, score, head):
    return {
        "arm": arm,
        "seed": seed,
        "score": score,
        "max_tile": 128,
        "turn": 20,
        "stop": "terminal",
        "checkpoint_head_sha256": head * 64,
    }


def write_batch(root, name, arm_rows, checkpoint_files, *, status="completed", stage="paired_30", **result_extra):
    gameplay = root / name / "gameplay"
    gameplay.mkdir(parents=True, exist_ok=True)
    seeds = sorted({item["seed"] for rows in arm_rows.values() for item in rows})
    (gameplay / "resolved-config.json").write_text(json.dumps({
        "schema_version": 1,
        "seeds": seeds,
        "checkpoint_files": checkpoint_files,
        "checkpoints": {arm: "unused" for arm in checkpoint_files},
    }), encoding="utf-8")
    result = {
        "status": status,
        "stage": stage,
        "results": arm_rows,
    }
    result.update(result_extra)
    (gameplay / "result.json").write_text(json.dumps(result), encoding="utf-8")
    return gameplay


class PlayEvaluationTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def index(self):
        return PlayEvaluationIndex(self.root)

    def test_latest_is_arithmetic_mean_and_aliases_share_full_identity(self):
        model_pins = pins("a")
        write_batch(self.root, "round1-first", {
            "old-alias": [row("old-alias", 1, 10, "a"), row("old-alias", 2, 20, "a")],
        }, {"old-alias": model_pins}, stage="fixed_10")
        write_batch(self.root, "round2-latest", {
            "new-alias": [row("new-alias", 2, 20, "a"), row("new-alias", 3, 40, "a")],
        }, {"new-alias": model_pins}, stage="paired_2")

        summary = self.index().summarize(model_pins)
        batches = summary["batches"]
        self.assertEqual([item["arm"] for item in batches], ["new-alias", "old-alias"])
        self.assertEqual(batches[0]["games"], 2)
        self.assertEqual(batches[0]["mean_score"], 30.0)
        self.assertNotIn("head_sha256", batches[0])
        self.assertNotIn("checkpoint_id", batches[0])
        self.assertNotIn("seeds", batches[0])
        self.assertEqual(summary["latest"]["label"], "第 2 轮 · 整局评测")
        self.assertEqual(summary["overall"]["games"], 3)
        self.assertAlmostEqual(summary["overall"]["mean_score"], 70 / 3)

    def test_head_plus_encoder_identity_does_not_alias_changed_encoder(self):
        first = pins("a")
        changed_encoder = dict(first, **{"adapter_model.safetensors": "f" * 64})
        write_batch(self.root, "round1-first", {
            "first": [row("first", 1, 10, "a")],
        }, {"first": first})
        write_batch(self.root, "round2-changed", {
            "changed": [row("changed", 1, 20, "a")],
        }, {"changed": changed_encoder})
        index = self.index()
        self.assertEqual(len(index.summarize(first)["batches"]), 1)
        self.assertEqual(index.summarize(first)["batches"][0]["mean_score"], 10.0)
        self.assertEqual(len(index.summarize(changed_encoder)["batches"]), 1)

    def test_incomplete_and_training_manifests_are_not_evaluations(self):
        model_pins = pins("a")
        write_batch(self.root, "round1-incomplete", {
            "candidate": [row("candidate", 1, 10, "a")],
        }, {"candidate": model_pins}, status="running")
        training = self.root / "round2-training"
        (training / "training").mkdir(parents=True)
        (training / "training" / "result.json").write_text(json.dumps({
            "status": "completed",
            "results": {"candidate": [row("candidate", 1, 99, "a")]},
        }), encoding="utf-8")
        self.assertEqual(self.index().summarize(model_pins)["batches"], [])

    def test_invalid_episode_is_discarded_instead_of_published(self):
        model_pins = pins("a")
        bad = row("candidate", 1, 10, "a")
        bad["stop"] = "not-a-real-stop"
        write_batch(self.root, "round1-invalid", {"candidate": [bad]}, {"candidate": model_pins})
        self.assertEqual(self.index().summarize(model_pins)["batches"], [])

    def test_teacher_calibration_has_stable_source_id(self):
        report_dir = self.root / "preparation-20261123"
        episode_dir = report_dir / "calibration" / "d3"
        episode_dir.mkdir(parents=True)
        (report_dir / "calibration.json").write_text(json.dumps({
            "passed": True,
            "selected_depth": 3,
        }), encoding="utf-8")
        for seed, score in ((100, 1000), (101, 2000)):
            (episode_dir / str(seed)).mkdir()
            (episode_dir / str(seed) / "result.json").write_text(json.dumps({
                "seed": seed,
                "score": score,
                "source_policy": "teacher",
                "teacher_depth": 3,
                "stop": "target",
            }), encoding="utf-8")
        result = self.index().summarize("teacher")
        self.assertEqual(result["latest"]["sourceId"], "teacher")
        self.assertEqual(result["latest"]["games"], 2)
        self.assertEqual(result["latest"]["mean_score"], 1500.0)

    def test_reused_reference_is_marked_per_arm(self):
        reference = pins("a")
        candidate = pins("b")
        write_batch(self.root, "round1-reuse", {
            "reference": [row("reference", 1, 10, "a")],
            "candidate": [row("candidate", 1, 20, "b")],
        }, {"reference": reference, "candidate": candidate},
        reference_id="reference", candidate_id="candidate", reference_games_reused=1)
        index = self.index()
        self.assertTrue(index.summarize(reference)["batches"][0]["reused"])
        self.assertEqual(index.summarize(reference)["batches"][0]["fresh_games"], 0)
        self.assertFalse(index.summarize(candidate)["batches"][0]["reused"])

    def test_latest_prefers_new_measurement_over_newer_reused_control(self):
        model = pins("a")
        write_batch(self.root, "round1-fresh", {
            "candidate": [row("candidate", 1, 10, "a"), row("candidate", 2, 20, "a")],
        }, {"candidate": model})
        write_batch(self.root, "round2-reused", {
            "candidate": [row("candidate", 1, 10, "a")],
        }, {"candidate": model}, reference_id="candidate", reference_games_reused=1)
        result = self.index().summarize(model)
        self.assertEqual(result["latest"]["batch"], "round1-fresh")
        self.assertFalse(result["latest"]["reused"])
        self.assertEqual([batch["batch"] for batch in result["batches"]],
                         ["round2-reused", "round1-fresh"])

    @unittest.skipUnless(
        Path("runs/round13-fresh-paired-4k-20261123/gameplay/result.json").is_file(),
        "repository evidence run is not present",
    )
    def test_repository_r13_means_are_read_as_means(self):
        common = {
            "adapter_config.json": "748acb2cda88454cb1ba69d745ba336f3fcb5486eac349e90960c8b8d8d3e854",
            "adapter_model.safetensors": "9b908623acb162118575f4e7a94524f9c139c335be4bfb74d6cfceca01e1885a",
            "tokenizer_config.json": "8671bed7c852ce9e661be94f179a7b4ffd091c2a65aea0363e5501c20318ee45",
            "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        }
        expected = {
            "d2c495a71408ee7d4d1a10d9710c75922b36200a022737633ba22e27146133a3": 1446.2667,
            "a6ba731e4a1db4b7dd8e130efc723204949a4e71cae499ab39768f6756e791f3": 1521.7333,
        }
        index = PlayEvaluationIndex("runs")
        for head, mean in expected.items():
            result = index.summarize(dict(common, **{"head.pt": head}))
            self.assertEqual(result["latest"]["games"], 30)
            self.assertAlmostEqual(result["latest"]["mean_score"], mean, places=4)


if __name__ == "__main__":
    unittest.main()

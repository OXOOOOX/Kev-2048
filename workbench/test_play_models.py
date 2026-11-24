"""CPU/read-only tests for the playable model catalog."""

from pathlib import Path
import argparse
import unittest

try:
    from play_models import ModelLibrary, candidate_models
except ModuleNotFoundError:  # Running unittest from the repository root.
    from workbench.play_models import ModelLibrary, candidate_models


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


class PlayModelCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        required = (
            RUNS / "round10-teacher-4k-20261123" / "candidate.json",
            RUNS / "round12-onpolicy-4k-20261123" / "candidate.json",
        )
        if not RUNS.is_dir() or not all(path.is_file() for path in required):
            raise unittest.SkipTest("published R10/R12 training evidence is not present")
        cls.candidates = candidate_models(RUNS)

    def test_published_teacher_and_onpolicy_heads_are_discovered(self):
        self.assertIn("teacher-4000-kl0", self.candidates)
        self.assertIn("onpolicy-4000-kl0", self.candidates)
        self.assertNotIn("round13-fresh-paired-4k-20261123", self.candidates)
        for name in ("teacher-4000-kl0", "onpolicy-4000-kl0"):
            spec = self.candidates[name]
            self.assertEqual(spec["kind"], "model")
            self.assertTrue(spec["checkpoint"].joinpath("head.pt").is_file())
            self.assertIn("只训练策略头", spec["training_summary"])

    def test_summaries_use_sealed_recipe_and_real_source_mix(self):
        teacher = self.candidates["teacher-4000-kl0"]["training_summary"]
        onpolicy = self.candidates["onpolicy-4000-kl0"]["training_summary"]
        self.assertIn("4,000", teacher)
        self.assertIn("4,000 张教师盘面", teacher)
        self.assertIn("深度 3", teacher)
        self.assertIn("KL=0", teacher)
        self.assertIn("最佳第 10 轮", teacher)
        self.assertIn("3,000", onpolicy)
        self.assertIn("早期和中期各 500 张", onpolicy)
        self.assertIn("KL=0", onpolicy)
        self.assertIn("最佳第 9 轮", onpolicy)

    def test_conflicting_round_one_id_is_explicit_and_round_four_keeps_legacy_id(self):
        self.assertIn("round1-2000-lr8e-5-kl015", self.candidates)
        self.assertIn("2000-lr8e-5-kl015", self.candidates)
        first = self.candidates["round1-2000-lr8e-5-kl015"]
        fourth = self.candidates["2000-lr8e-5-kl015"]
        self.assertNotEqual(first["checkpoint"], fourth["checkpoint"])
        self.assertEqual(first["report"]["best_epoch"], 22)
        self.assertEqual(fourth["report"]["best_epoch"], 4)

    def test_r7_nested_candidates_are_sealed_without_cuda(self):
        expected = {"scale-1000", "scale-2000", "scale-4000", "scale-6000", "scale-8000"}
        self.assertTrue(expected <= set(self.candidates))
        for name in expected:
            spec = self.candidates[name]
            self.assertEqual(spec["root"].name, "round7-scale-20261123")
            self.assertEqual(spec["report"]["status"], "completed")

    @unittest.skipUnless((ROOT / "assets" / "qwen-base").is_dir(), "pinned assets are not present")
    def test_refresh_has_ttl_snapshot_and_preserves_loaded_session(self):
        args = argparse.Namespace(
            base=ROOT / "assets" / "qwen-base",
            checkpoint=ROOT / "assets" / "official-kev",
            device="cpu",
            teacher_preparation=None,
        )
        library = ModelLibrary(args, RUNS)
        initial = library.models
        self.assertIs(library.refresh(), initial)
        by_id = {item["id"]: item for item in initial}
        self.assertEqual(by_id["onpolicy-4000-kl0"]["name"], "自走纠错 · 4,000 张")
        self.assertEqual(by_id["teacher-4000-kl0"]["name"], "纯教师 · 4,000 张")
        if by_id["official"]["evaluation"] is None:
            self.skipTest("the optional gameplay score index is not available")
        self.assertIsNotNone(by_id["official"]["evaluation"])
        library.heads["onpolicy-4000-kl0"] = object()
        refreshed = library.refresh(force=True)
        self.assertIs(library.models, refreshed)
        self.assertIn("onpolicy-4000-kl0", {item["id"] for item in refreshed})
        # A loaded head remains advertised even if a subsequent publication
        # snapshot omits its candidate.
        old_candidates = library.candidates
        library.candidates = {}
        try:
            advertised = library._public_models({}, library.teacher_spec, library._evaluations)
        finally:
            library.candidates = old_candidates
        self.assertIn("onpolicy-4000-kl0", {item["id"] for item in advertised})


if __name__ == "__main__":
    unittest.main()

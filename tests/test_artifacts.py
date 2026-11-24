"""Unit tests for shared artifact verification and game stop helpers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from kev2048_release.artifacts import completed_trial, identity, load_plan, stop_reason
from kev2048_release.data import digest
from kev2048_release.game import Game


class ArtifactsTest(unittest.TestCase):
    def test_identity_is_key_order_invariant(self):
        self.assertEqual(identity({"b": 1, "a": [2, 3]}), identity({"a": [2, 3], "b": 1}))
        with self.assertRaises(ValueError):
            identity({"bad": float("nan")})

    def test_stop_reason_priority(self):
        game = Game.seeded(1)
        game.state, game.turn = (2048,) + (0,) * 15, 5000
        self.assertEqual(stop_reason(game, 5000), "target")

        dead = Game.seeded(1)
        dead.state, dead.turn = (2, 4, 8, 16, 32, 64, 128, 256, 2, 4, 8, 16, 32, 64, 128, 256), 10
        self.assertEqual(stop_reason(dead, 5000), "terminal")

        running = Game.seeded(1)
        running.state, running.turn = (2, 0, 0, 0) + (0,) * 12, 5
        self.assertIsNone(stop_reason(running, 10))
        self.assertEqual(stop_reason(running, 5), "capped")

    def test_load_plan_verifies_plan_teacher_and_sources(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            teacher_dir = root / "teacher"
            teacher_dir.mkdir()
            (teacher_dir / "manifest.json").write_text('{"revision":"ok"}\n', encoding="utf-8")

            pkg_dir = Path(__file__).resolve().parents[1] / "src" / "kev2048_release"
            plan = {
                "teacher_dir": str(teacher_dir),
                "teacher_manifest_sha256": digest(teacher_dir / "manifest.json"),
                "implementation_sha256": {"game.py": digest(pkg_dir / "game.py")},
            }
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "plan-lock.json").write_text(
                json.dumps({"plan_sha256": digest(root / "plan.json")}), encoding="utf-8"
            )
            self.assertEqual(load_plan(root)["teacher_dir"], str(teacher_dir))

            (teacher_dir / "manifest.json").write_text('{"revision":"tampered"}\n', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Teacher build changed"):
                load_plan(root)

            (teacher_dir / "manifest.json").write_text('{"revision":"ok"}\n', encoding="utf-8")
            plan["implementation_sha256"]["missing_module.py"] = "0" * 64
            (root / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
            (root / "plan-lock.json").write_text(
                json.dumps({"plan_sha256": digest(root / "plan.json")}), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Preparation implementation changed"):
                load_plan(root)

    def test_completed_trial_verifies_identity_and_files(self):
        with TemporaryDirectory() as tmp:
            trial = Path(tmp)
            self.assertIsNone(completed_trial(trial, "id-1"))

            ckpt = trial / "checkpoint"
            ckpt.mkdir()
            (ckpt / "head.pt").write_bytes(b"head-bytes")
            (trial / "metrics.json").write_text('{"ok":true}\n', encoding="utf-8")
            report = {
                "identity": "id-1",
                "status": "completed",
                "checkpoint_files": {"head.pt": digest(ckpt / "head.pt")},
                "artifacts": {"metrics.json": digest(trial / "metrics.json")},
            }
            (trial / "result.json").write_text(json.dumps(report), encoding="utf-8")
            self.assertEqual(completed_trial(trial, "id-1")["identity"], "id-1")

            with self.assertRaisesRegex(ValueError, "Invalid completed trial"):
                completed_trial(trial, "wrong-id")

            (ckpt / "head.pt").write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError, "Completed checkpoint was modified"):
                completed_trial(trial, "id-1")


if __name__ == "__main__":
    unittest.main()

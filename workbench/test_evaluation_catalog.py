"""Exercise automatic replay discovery with isolated, completed evaluation files."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation_catalog import EvaluationCatalog, _milestones
from live import GameplayWatch, Watch
from test_gameplay_replay import encode, recorded


class EvaluationCatalogTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.directory = self.root/"runs"/"paired"/"gameplay"
        self.directory.mkdir(parents=True)
        self.log = self.directory/"progress.jsonl"
        self.log.touch()
        self.identity = "a"*64
        self.arm = "candidate"
        self.config()
        self.watch = GameplayWatch(3, "Paired", self.directory)
        self.catalog = EvaluationCatalog(self.root/"runs", audit_seconds=0)

    def config(self, arms=("candidate",), seeds=(17, 18, 19, 20)):
        config = dict(schema_version=1, identity=self.identity, config={"seed_group_size":2},
                      checkpoints={a:"unused" for a in arms}, seeds=seeds, screening_seeds=list(seeds[:2]))
        (self.directory/"resolved-config.json").write_text(json.dumps(config), encoding="utf8")

    def sample(self, seed, arm=None, *, complete=True):
        arm = arm or self.arm
        dest = self.directory/arm/str(seed)
        dest.mkdir(parents=True, exist_ok=True)
        moves = recorded(seed, 3, arm)
        raw = encode(moves)
        (dest/"moves.jsonl").write_bytes(raw)
        end = moves[-1]
        summary = dict(schema_version=1, run_identity=self.identity, split="gameplay_dev", arm=arm,
                       cap=3, moves_sha256=hashlib.sha256(raw).hexdigest(),
                       **{k:end[k] for k in ("seed", "turn", "stop", "score", "max_tile")})
        if complete:
            (dest/"result.json").write_text(json.dumps(summary), encoding="utf8")
        return dest, summary

    def emit(self, summary, when):
        event = dict(phase="episode_completed", timestamp=when, turns=summary["turn"],
                     **{k:summary[k] for k in ("arm", "seed", "score", "stop", "max_tile")})
        with self.log.open("ab") as stream:
            stream.write(encode([event]))

    def test_completed_new_games_appear_with_their_actual_log_times(self):
        before, first = self.sample(17)
        self.emit(first, "2026-01-01T05:00:01.500Z")
        pending, second = self.sample(18, complete=False)
        self.sample(42)  # An unscheduled seed with a valid looking trace.
        self.assertEqual([row["seed"] for row in self.catalog.catalog([self.watch])], [17])
        (pending/"result.json").write_text(json.dumps(second), encoding="utf8")
        self.emit(second, "2026-01-01T05:00:01Z")
        rows = self.catalog.catalog([self.watch])
        self.assertEqual([r["seed"] for r in rows], [17, 18])
        a = rows[0]
        self.assertEqual((a["channel"], a["arm"], a["model"], a["run"], a["source"]),
                         ("watch-3", self.arm, self.arm, "Paired", "evaluation"))
        self.assertEqual((a["turns"], a["score"], a["max_tile"], a["timeSource"]),
                         (3, first["score"], first["max_tile"], "logged"))
        self.assertEqual(a["replayIdentity"], f"{self.identity}/{self.arm}/17/{first['moves_sha256']}")
        self.assertNotIn("embedded", a)
        with patch("evaluation_catalog._episode", side_effect=AssertionError("Reread unchanged moves")):
            self.assertEqual(self.catalog.catalog([self.watch]), rows)
        (before/"moves.jsonl").write_bytes((before/"moves.jsonl").read_bytes()+b"torn")
        self.assertEqual([r["seed"] for r in self.catalog.catalog([self.watch])], [18])

    def test_duplicate_channels_and_reused_baselines_are_counted_once(self):
        self.directory = self.root/"runs"/"paired-archive"
        self.directory.mkdir()
        self.identity = "c"*64
        self.config(("official", self.arm))
        self.log = self.directory/"progress.jsonl"
        self.log.touch()
        self.sample(17)
        self.sample(17, "official")
        watch = GameplayWatch("primary", "Archive", self.directory)
        duplicate = GameplayWatch("alias", "Alias", self.directory)
        rows = self.catalog.catalog([watch, duplicate])
        self.assertEqual(len(rows),2)
        self.assertTrue(all("embedded" not in r and r["timeSource"] == "modified" for r in rows))
        (self.directory/"baseline-reuse.json").write_text(json.dumps(
            dict(schema_version=1, arm="official", identity=self.identity, reused_games=4)), encoding="utf8")
        only, = self.catalog.catalog([watch])
        self.assertEqual(only["arm"], self.arm)
        relocated = GameplayWatch("alias", "Alias", self.directory)
        self.assertEqual(self.catalog.catalog([relocated])[0]["id"], only["id"])
        relocated.path_guard = lambda _: False
        self.assertEqual(self.catalog.catalog([relocated]), [])
        self.assertEqual(EvaluationCatalog(self.root/"other").catalog([watch]), [])

    def test_missing_mismatched_and_replaced_source_files(self):
        first, _ = self.sample(17)
        (first/"moves.jsonl").unlink()
        second, _ = self.sample(18)
        (second/"result.json").write_bytes(b"{")
        third, info = self.sample(19)
        (third/"result.json").write_text(json.dumps(dict(info, arm="other")), encoding="utf8")
        fourth, info = self.sample(20)
        (fourth/"result.json").write_text(json.dumps(dict(info, turn=2)), encoding="utf8")
        self.assertEqual(self.catalog.catalog([self.watch]), [])
        self.sample(20)
        row, = self.catalog.catalog([self.watch])
        self.assertEqual(row["seed"], 20)
        self.identity = "b"*64
        self.config()
        self.assertEqual(self.catalog.catalog([self.watch]), [])

    def test_milestones_store_the_first_turn_for_each_absolute_tile(self):
        tiles = [128, 512, 512, 1024, 1024, 2048]
        raw = encode([dict(seed=17, turn=i, max_tile=t, after=[[t,0,0,0]]+[[0]*4 for _ in range(3)])
                      for i,t in enumerate(tiles,1)])
        self.assertEqual(_milestones(raw,17,2048), {"512":2,"1024":4,"2048":6})
        with self.assertRaises(ValueError):
            _milestones(raw,18,2048)

    def test_model_identity_is_checked_against_declared_pins_without_opening_weights(self):
        dest, info = self.sample(17)
        manifest = self.directory/"resolved-config.json"
        config = json.loads(manifest.read_bytes())
        config["checkpoint_files"] = {self.arm: {"head.pt": "f"*64}}
        manifest.write_text(json.dumps(config), encoding="utf8")
        (dest/"result.json").write_text(json.dumps(dict(info, checkpoint_head_sha256="c"*64)), encoding="utf8")
        self.assertEqual(self.catalog.catalog([self.watch]), [])
        (dest/"result.json").write_text(json.dumps(dict(info, checkpoint_head_sha256="f"*64)), encoding="utf8")
        self.assertEqual([r["seed"] for r in self.catalog.catalog([self.watch])], [17])

    def test_native_multiseed_summary_has_one_row_for_each_verified_recording(self):
        folder = self.root/"runs"/"native"
        folder.mkdir()
        rows, games = [], []
        for seed in (19,27):
            for move in recorded(seed,3,"native"):
                rows.append({k:move[k] for k in ("seed","turn","before","after","action","score","probabilities")}
                            | {"tile":move["max_tile"]})
            games.append(dict(seed=seed,turn=3,score=move["score"],tile=move["max_tile"],stop="cap"))
        path = folder/"result.json"
        path.write_text(json.dumps(dict(cap=3,checkpoint_head_sha256="a"*64,games=games)), encoding="utf8")
        (folder/"moves.jsonl").write_bytes(encode(rows))
        watch = Watch(1, f"Native:evaluation:{path}")
        catalog = self.catalog.catalog([watch])
        self.assertEqual({(r["seed"],r["turns"],r["arm"],r["timeSource"]) for r in catalog},
                         {(s,3,"native","modified") for s in (19,27)})
        self.assertTrue(all("replayIdentity" not in r for r in catalog))
        (folder/"moves.jsonl").write_bytes(encode(rows[:-1]))
        self.assertEqual(self.catalog.catalog([watch]), [])


if __name__ == "__main__":
    unittest.main()

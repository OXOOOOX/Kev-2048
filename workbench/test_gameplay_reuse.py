"""A prior evaluation becomes browsable only through a verified comparison plan."""

import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from gameplay_reuse import GameplayReuse
from kev2048_release.game import Game, legal, nested
from kev2048_release.artifacts import identity


def put(path, data):
    raw = (json.dumps(data, sort_keys=True)+"\n").encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def seal(config):
    config["identity"] = identity({k:v for k,v in config.items() if k != "identity"})
    return config["identity"]


class ReuseTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()/"runs"
        self.source = self.root/"source"/"gameplay"
        self.current = self.root/"experiment"/"gameplay"
        self.current.mkdir(parents=True)
        self.source.mkdir(parents=True)
        self.arms, self.fresh = ["model-a", "model-b"], [10]
        self.old = dict(seeds=[5], cap=1, target_tile=2048, dtype="fp32", temperature=1,
                        checkpoint_files={a:{"head.pt":str(i)*64} for i,a in enumerate(self.arms, 1)},
                        checkpoints={a:str(self.root/"models"/a) for a in self.arms})
        old_id = seal(self.old)
        source_config = self.source/"resolved-config.json"
        source_report = self.source/"result.json"
        source_config_sha = put(source_config, self.old)
        self.moves = {}
        episode_hashes = {}
        for arm in self.arms:
            directory = self.source/arm/"5"
            directory.mkdir(parents=True)
            game = Game.seeded(5)
            before, directions = nested(game.state), legal(game.state)
            action = directions[0]
            game.step(action)
            move = dict(seed=5, before=before, after=nested(game.state), legal_moves=directions, action=action,
                        probabilities={k:int(k == action) for k in directions}, score=game.score,
                        turn=1, stop="capped", max_tile=max(game.state))
            move_sha = put(directory/"moves.jsonl", move)
            self.moves[arm] = move
            summary = dict(run_identity=old_id, arm=arm, checkpoint_head_sha256=self.old["checkpoint_files"][arm]["head.pt"],
                           seed=5, moves_sha256=move_sha, turn=1, stop="capped", score=game.score, max_tile=max(game.state),
                           cap=1, dtype="fp32", temperature=1)
            episode_hashes[arm] = {"5":put(directory/"result.json", summary)}
        self.report = dict(status="completed", identity=old_id, seeds=[5], episode_sha256=episode_hashes)
        report_sha = put(source_report, self.report)
        self.plan = dict(source=str(self.source.parent), initial_seeds=[5], additional_seeds=self.fresh,
                         models=self.old["checkpoints"], checkpoint_files=self.old["checkpoint_files"],
                         fresh_games=2, reused_games=2, archived_gameplay_sha256=report_sha)
        self.config = dict(self.old, seeds=self.fresh, source_gameplay_identity=old_id,
                           inputs_sha256={str(source_config):source_config_sha, str(source_report):report_sha})
        self.save_plan()
        self.reuse = GameplayReuse(self.root, self.current)

    def save_plan(self):
        plan_sha = put(self.current.parent/"plan.json", self.plan)
        self.config["comparison_plan_sha256"] = plan_sha
        self.save_config()

    def save_config(self):
        self.identity = seal(self.config)
        put(self.current/"resolved-config.json", self.config)

    def catalog(self):
        return self.reuse.seed_catalog(self.identity, self.arms, self.fresh)

    def episodes(self, seed=5):
        return self.reuse.seed_episodes(self.identity, self.arms, self.fresh, seed, time.time())

    def test_verified_paired_final_frames_and_cached_catalog(self):
        self.assertEqual(self.catalog(), [5])
        with patch.object(self.reuse, "_cohort", side_effect=AssertionError("Rescanned inside catalog interval")):
            self.assertEqual(self.catalog(), [5])
        actual = self.episodes()
        self.assertEqual([m["arm"] for m in actual],self.arms)
        for m in actual:
            self.assertTrue(m["reused"])
            self.assertEqual({k:m[k] for k in ("score","max_tile","action","turn","before","after","probabilities","stop")},
                             {k:self.moves[m["arm"]][k] for k in ("score","max_tile","action","turn","before","after","probabilities","stop")})
        self.assertIsNone(self.episodes(10))

    def test_reverify_trace_and_summary_on_selection_not_just_the_pinned_manifest(self):
        self.assertEqual(self.catalog(), [5])
        moves = self.source/self.arms[0]/"5"/"moves.jsonl"
        content = moves.read_bytes()
        moves.write_bytes(b" "+content)
        self.assertIsNone(self.episodes())
        moves.write_bytes(content)
        self.assertIsNotNone(self.episodes())
        result = self.source/self.arms[1]/"5"/"result.json"
        result.write_bytes(result.read_bytes()+b" ")
        self.assertIsNone(self.episodes())
        self.assertEqual(GameplayReuse(self.root,self.current).seed_catalog(self.identity,self.arms,self.fresh), [])

    def test_missing_unpinned_or_incompatible_comparison_is_hidden(self):
        self.assertEqual(self.reuse.seed_catalog("0"*64,self.arms,self.fresh), [])
        self.config["dtype"] = "bf16"
        self.save_config()
        self.assertEqual(self.catalog(), [])
        self.config["dtype"] = "fp32"
        self.plan["initial_seeds"] = [10]
        self.save_plan()
        self.assertEqual(self.catalog(), [])
        self.plan["initial_seeds"] = [5]
        self.save_plan()
        self.config["inputs_sha256"] = []
        self.save_config()
        self.assertEqual(self.catalog(), [])
        (self.current/"resolved-config.json").write_text("{}", encoding="utf8")
        self.assertIsNone(self.episodes())

    def test_outside_run_roots_and_symlinked_trajectories_are_not_followed(self):
        self.plan["source"] = str(self.root.parent/"unregistered")
        self.save_plan()
        self.assertEqual(self.catalog(), [])
        self.assertIsNone(GameplayReuse(None,self.current).seed_episodes(self.identity,self.arms,self.fresh,5,time.time()))
        self.plan["source"] = str(self.source.parent)
        self.save_plan()
        moves = self.source/self.arms[0]/"5"/"moves.jsonl"
        elsewhere = self.root.parent/"elsewhere.jsonl"
        moves.rename(elsewhere)
        try:
            moves.symlink_to(elsewhere)
        except OSError as error:
            self.skipTest(f"Symlinks unavailable: {error}")
        self.assertEqual(self.catalog(), [])


if __name__ == "__main__":
    unittest.main()

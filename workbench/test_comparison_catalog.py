"""Exercise comparison discovery and provenance without starting a model or service."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from comparison_catalog import build_comparison_catalog


class ComparisonCatalogTests(unittest.TestCase):
    def setUp(self):
        scratch = TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name).resolve()
        self.watches, self.channels, self.evaluations = [], [], []

    def register(self, channel, relative, arms=(), seeds=(71, 72), date="2026-03-02", name="相同的名称"):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        watch = SimpleNamespace(base={"id": channel}, path=path)
        if arms:
            watch.directory = path.parent
        self.watches.append(watch)
        self.channels.append(dict(id=channel, name=name, updated_at=date, mode="gameplay" if arms else "static_result",
                                  arms=list(arms), seed_catalog=list(seeds)))
        return watch

    def aggregate(self, watch, records, *, config="c"*64, result="c"*64, status="completed"):
        def put(name, data):
            target = watch.directory / name
            pending = target.with_suffix(".tmp")
            pending.write_text(json.dumps(data), encoding="utf-8")
            pending.replace(target)
        put("resolved-config.json", dict(schema_version=1, identity=config))
        put("result.json", dict(status=status, identity=result, results=records))

    @staticmethod
    def claim(arm="official", seed=71, origin="a"*64, digest="b"*64):
        return dict(arm=arm, seed=seed, run_identity=origin, moves_sha256=digest)

    def record(self, channel, claim, **extra):
        replay = "{run_identity}/{arm}/{seed}/{moves_sha256}".format(**claim)
        self.evaluations.append(dict(channel=channel, arm=claim["arm"], seed=claim["seed"], replayIdentity=replay, **extra))

    def periods(self):
        return {period["id"]: period for period in
                build_comparison_catalog(self.watches, self.channels, self.evaluations, self.root)["periods"]}

    def test_open_then_refresh_discovers_new_period_and_resolves_unambiguous_bookmarks(self):
        parent = "watch-evaluation-1769f454d15c657a"
        self.register(parent, "2026-03-02/some-run/result.json")
        self.register("first-stage", "2026-03-02/some-run/games/progress.jsonl", ("left", "right"))
        self.register("unrelated", "2026-03-03/some-run/result.json")  # Same name in a different run.
        first = self.periods()["first-stage"]
        self.assertEqual(first["aliases"], [parent])
        self.assertEqual((first["arms"], first["seeds"]), (["left", "right"], [71, 72]))
        self.assertEqual(first["references"], {})
        self.register("new-stage", "future-arbitrary-name/play/progress.jsonl", ("new-a", "new-b"), date="2026-05-07")
        self.assertEqual(list(self.periods()), ["new-stage", "first-stage"])
        self.register("competing-stage", "2026-03-02/some-run/more-games/progress.jsonl", ("third",))
        self.assertNotIn(parent, [alias for p in self.periods().values() for alias in p["aliases"]])

    def test_resolves_only_claimed_full_identity_and_caches_unchanged_metadata(self):
        self.register("older", "prior/games/progress.jsonl", ("official",))
        current = self.register("new", "recent/games/progress.jsonl", ("candidate",))
        claim = self.claim()
        other = self.claim(seed=72)
        outside = self.claim(seed=88)
        for item in (claim, other, outside):
            self.record("older", item, embedded=True)
        self.aggregate(current, {"official": [claim, claim, dict(other, moves_sha256="d"*64), outside]})
        periods = self.periods()
        self.assertEqual(periods["new"]["references"], {"71": {"official": "older"}})
        self.assertEqual(periods["new"]["arms"], ["candidate", "official"])
        self.assertEqual(periods["new"]["direct_arms"], ["candidate"])
        self.assertTrue(periods["older"]["embedded"])
        self.assertNotIn("embedded", periods["new"])
        with patch("comparison_catalog._read", side_effect=AssertionError("Read unchanged aggregate")):
            self.assertEqual(self.periods(), periods)

    def test_mismatched_replaced_incomplete_or_ambiguous_metadata_cannot_attach_reference(self):
        self.register("a", "prior/gameplay/progress.jsonl", ("official",))
        current = self.register("b", "new/gameplay/progress.jsonl", ("candidate",))
        claim = self.claim()
        self.record("a", claim)
        for records, metadata in [
            ({"official": [claim]}, {"result": "f"*64}),
            ({"official": [claim]}, {"status": "running"}),
            ({"official": [dict(claim, arm="other")]}, {}),
            ({"official": [claim, dict(claim, moves_sha256="0"*64)]}, {}),
        ]:
            with self.subTest(records=records, metadata=metadata):
                self.aggregate(current, records, **metadata)
                self.assertEqual(self.periods()["b"]["references"], {})
        self.aggregate(current, {"official": [claim]})
        self.assertEqual(self.periods()["b"]["references"], {"71": {"official": "a"}})
        self.register("c", "other/gameplay/progress.jsonl", ("official",))
        self.record("c", claim)
        self.assertEqual(self.periods()["b"]["references"], {})

    def test_duplicate_watch_and_existing_local_track_are_not_replaced(self):
        self.register("first", "source/games/progress.jsonl", ("official",))
        self.register("copy", "source/games/progress.jsonl", ("official",))
        current = self.register("new", "new/games/progress.jsonl", ("official", "candidate"))
        claim = self.claim()
        self.record("copy", claim)
        self.aggregate(current, {"official": [claim]})
        periods = self.periods()
        self.assertEqual(set(periods), {"first", "new"})
        self.assertEqual(periods["first"]["aliases"], ["copy"])
        self.assertEqual(periods["new"]["references"]["71"], {"official": "first"})
        self.record("new", self.claim(origin="c"*64))
        self.assertEqual(self.periods()["new"]["references"], {})


if __name__ == "__main__":
    unittest.main()

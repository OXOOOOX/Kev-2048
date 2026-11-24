"""Exercise the saved-game list and the public archive load boundary."""

from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from kev2048_release.game import legal
from play_session import PlaySession
from replay_library import ReplayLibrary, _read_milestones


MODELS = [
    dict(id="official", name="官方基线", kind="model"),
    dict(id="trial-head", name="候选模型", kind="model"),
]


def decisions(requests):
    result = {}
    for model, game in requests:
        options = legal(game.state)
        result[model] = dict(probabilities={direction: 1 / len(options) for direction in options}, decision_ms=1)
    return result


class ReplayLibraryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name) / "archive"
        self.play = PlaySession(self.root, MODELS, lambda _: decisions)
        self.library = ReplayLibrary(self.root)

    def _end_game(self, models, seed, cap=2):
        rules = {"mode": "endless"} if cap is None else {"cap": cap}
        state = self.play.start(dict(models=models, seed=seed, **rules))
        while state["session"]["active"]:
            state = self.play.pending(state["revision"], 2)
            self.assertIsNotNone(state)
        return state["session"]

    def test_starred_comparison_and_legacy_archive(self):
        baseline = self._end_game(["official"], 123)
        self.assertEqual(self.library.catalog(), dict(games=[], truncated=False))
        self.play.save(dict(id=baseline["id"]))
        baseline_folder = self.root / baseline["id"] / "official"
        run_file = baseline_folder / "run.json"
        data = json.loads(run_file.read_text(encoding="utf-8"))
        del data["starred"], data["saved_at"]
        data.pop("mode", None)
        run_file.write_text(json.dumps(data), encoding="utf-8")
        legacy_moves = baseline_folder / "moves.jsonl"
        legacy_moves.write_bytes(legacy_moves.read_bytes().replace(b'"mode":"classic",', b""))
        os.utime(legacy_moves, (1_700_000_000, 1_700_000_000))
        result = self._end_game(["official", "trial-head"], 456)
        self.play.save(dict(id=result["id"]))
        catalog = self.library.catalog()
        self.assertFalse(catalog["truncated"])
        first, second, legacy = catalog["games"]
        self.assertEqual((first["starred"], second["starred"], legacy["starred"]), (True, True, False))
        self.assertEqual((first["time"], legacy["time"]), (second["time"], "2023-11-14T22:13:20.000Z"))
        self.assertIsNotNone(datetime.fromisoformat(first["time"]).tzinfo)
        for row in catalog["games"]:
            self.assertEqual(row["milestones"], {})
            record = self.library.record(row["id"])
            self.assertEqual(record["manifest"]["variant"], row["model"])
            self.assertEqual(record["manifest"]["seeds"], [row["seed"]])
            self.assertEqual((row["mode"], record["manifest"]["mode"]), ("classic", "classic"))
            last = json.loads(record["moves"].splitlines()[-1])
            self.assertEqual((row["turns"], row["score"], row["max_tile"]), (last["turn"], last["score"], last["tile"]))
            self.assertEqual(record["manifest"]["starred"], row["starred"])

    def test_endless_star_preserves_rules_and_can_pass_classic_limits(self):
        finished = self._end_game(["official"], 123, cap=None)
        self.play.save(dict(id=finished["id"]))
        with patch("replay_library.MAX_TURNS", 2), patch("replay_library.MAX_MOVES_BYTES", 8):
            row, = self.library.catalog()["games"]
            record = self.library.record(row["id"])
        self.assertEqual((row["mode"], record["manifest"]["mode"]), ("endless", "endless"))
        self.assertIsNone(record["manifest"]["max_moves_per_game"])
        moves = [json.loads(line) for line in record["moves"].splitlines()]
        self.assertGreater(row["turns"], 2)
        self.assertEqual(len(moves), row["turns"])
        self.assertEqual(moves[-1]["stop"], "terminal")
        self.assertFalse(legal(tuple(value for line in moves[-1]["board"] for value in line)))
        with patch("replay_library.MAX_ENDLESS_MOVES_BYTES", 8):
            self.assertIsNone(self.library.record(row["id"]))

    def test_endless_rejects_inconsistent_limits_and_premature_terminal_events(self):
        finished = self._end_game(["official"], 123, cap=None)
        self.play.save(dict(id=finished["id"]))
        identifier = f"{finished['id']}/official"
        track = self.root / identifier
        manifest = track / "run.json"
        original = json.loads(manifest.read_bytes())
        for change in ({"mode": "unknown"}, {"mode": "classic"}, {"max_moves_per_game": 5000}):
            with self.subTest(manifest=change):
                manifest.write_text(json.dumps({**original, **change}), encoding="utf-8")
                self.assertFalse(self.library.catalog()["games"])
        manifest.write_text(json.dumps(original), encoding="utf-8")
        moves = track / "moves.jsonl"
        rows = [json.loads(line) for line in moves.read_bytes().splitlines()]

        def write(events):
            moves.write_bytes(b"".join(json.dumps(event).encode("utf-8") + b"\n" for event in events))

        for change in ({"stop": "target"}, {"stop": "cap"}, {"mode": "classic"}, {"cap": 5000}):
            with self.subTest(final_move=change):
                write([*rows[:-1], {**rows[-1], **change}])
                self.assertFalse(self.library.catalog()["games"])
                self.assertIsNone(self.library.record(identifier))
        write([{**rows[0], "stop": "terminal"}])
        self.assertFalse(self.library.catalog()["games"])
        for change in ({"cap": 2}, {"mode": "classic"}, {"stop": "target"}):
            with self.subTest(early_move=change):
                write([rows[0], {**rows[1], **change}, *rows[2:]])
                self.assertIsNone(self.library.record(identifier))

        first = rows[0]
        for event, outcome in (("stopped", "stopped"), ("error", None)):
            with self.subTest(manual_or_error=event):
                write([first, {**first, "type": event, "seq": first["seq"] + 1, "stop": outcome}])
                self.assertEqual(self.library.catalog()["games"][0]["turns"], 1)
                self.assertIsNotNone(self.library.record(identifier))

    def test_milestones_use_confirmed_moves_and_refresh_when_archive_changes(self):
        finished = self._end_game(["official"], 123, cap=8)
        identifier = f"{finished['id']}/official"
        self.play.save(dict(id=finished["id"]))
        moves = self.root / identifier / "moves.jsonl"
        unchanged = moves.read_bytes()
        expected = {"4": 1, "8": 8}
        with patch("replay_library._read_milestones", wraps=_read_milestones) as scan:
            row, = self.library.catalog()["games"]
            self.assertEqual(row["milestones"], {})
            scan.assert_not_called()
            # A short, legal trace exercises multiple first reaches without storing a long fixture.
            with patch("replay_library.MILESTONE_TILES", (4, 8, 16)):
                row, = self.library.catalog()["games"]
                self.assertEqual(row["milestones"], expected)
                row["milestones"]["8"] = 3
                self.assertEqual(self.library.catalog()["games"][0]["milestones"], expected)
                self.assertEqual(scan.call_count, 1)
                fresh = ReplayLibrary(self.root)
                self.assertEqual(fresh.catalog()["games"][0]["milestones"], expected)
                self.assertEqual(scan.call_count, 2)
                damaged = unchanged.replace(b'"type":"move"', b'"type":"oops"', 1)
                parent = moves.parent
                replacement = parent / "replacement.jsonl"
                replacement.write_bytes(damaged)
                replacement.replace(moves)
                row, = self.library.catalog()["games"]
                self.assertEqual((row["id"], row["milestones"]), (identifier, {}))
                self.assertEqual(scan.call_count, 3)
                moves.write_bytes(unchanged)
                os.utime(moves, (1_700_000_000, 1_700_000_000))
                self.assertEqual(self.library.catalog()["games"][0]["milestones"], expected)
                self.assertEqual(scan.call_count, 4)
                self.assertEqual(self.library.record(identifier)["moves"], unchanged.decode("utf-8"))
                self.assertEqual(scan.call_count, 4)

    def test_invalid_ids_staging_and_damaged_files_are_ignored(self):
        result = self._end_game(["official"], 123)
        self.play.save(dict(id=result["id"]))
        identifier = f"{result['id']}/official"
        track = self.root / result["id"] / "official"
        for invalid in (None, "../official", identifier + "/moves.jsonl", f"{result['id']}/..", "official"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.library.record(invalid)
        self.assertIsNone(self.library.record(f"{result['id']}/unknown"))
        moves = track / "moves.jsonl"
        intact = moves.read_bytes()
        (self.root / ".instant-play-save-staged").mkdir()
        with patch("replay_library.MAX_MOVES_BYTES", 12):
            self.assertIsNone(self.library.record(identifier))
            self.assertFalse(self.library.catalog()["games"])
        moves.write_bytes(intact[:-1])
        self.assertFalse(self.library.catalog()["games"])
        moves.write_bytes(intact)
        self.assertEqual(len(self.library.catalog()["games"]), 1)
        with patch("replay_library.MAX_CATALOG_DIRS", 1):
            self.assertTrue(self.library.catalog()["truncated"])

    def test_stopped_game_uses_last_confirmed_position(self):
        deciding = threading.Event()
        continue_deciding = threading.Event()
        self.addCleanup(continue_deciding.set)

        def wait_after_first_move(requests):
            if requests[0][1].turn == 1:
                deciding.set()
                continue_deciding.wait(2)
            return decisions(requests)

        self.play = PlaySession(self.root, MODELS, lambda _: wait_after_first_move)
        game = self.play.start(dict(models=["official"], seed=123, cap=2))["session"]
        self.assertTrue(deciding.wait(2))
        stopped = self.play.stop(dict(id=game["id"]))
        continue_deciding.set()
        ended = self.play.pending(stopped["revision"], 2)["session"]
        self.assertEqual(ended["status"], "stopped")
        self.play.save(dict(id=ended["id"]))
        row, = self.library.catalog()["games"]
        self.assertEqual(row["turns"], 1)
        record = self.library.record(row["id"])
        last = json.loads(record["moves"].splitlines()[-1])
        self.assertEqual(last["stop"], "stopped")
        self.assertEqual((row["score"], row["max_tile"]), (last["score"], last["tile"]))

    def test_symlinked_archive_folder_is_not_exposed(self):
        result = self._end_game(["official"], 123)
        self.play.save(dict(id=result["id"]))
        actual = self.root / result["id"]
        relocated = self.root.parent / result["id"]
        actual.rename(relocated)
        try:
            actual.symlink_to(relocated, target_is_directory=True)
        except OSError:
            self.skipTest("Creating a directory symlink is unavailable")
        self.assertIsNone(self.library.record(f"{result['id']}/official"))
        self.assertFalse(self.library.catalog()["games"])


if __name__ == "__main__":
    unittest.main()

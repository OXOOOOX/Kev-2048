"""Exercise the local play session with deterministic CPU-only stand-ins."""

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from play_session import MAX_MODELS, MAX_SEED, MAX_TURNS, PlayRequestError, PlaySession, _Track
from kev2048_release.game import Game, legal, nested


MODELS = [
    dict(id="official", name="官方基线", kind="model"),
    dict(id="other-head", name="候选模型", kind="model"),
    dict(id="teacher", name="教师模型", kind="teacher"),
]


def decisions(requests):
    result = {}
    for key, game in requests:
        opts = legal(game.state)
        if key == "teacher":
            result[key] = dict(action_values={move: -10 + i * 2 for i, move in enumerate(opts)}, decision_ms=0.7)
        else:
            # Reverse the dict order so equal scores must follow the board's legal-move order.
            result[key] = dict(probabilities={move: 1 / len(opts) for move in reversed(opts)}, decision_ms=0.3)
    return result


class PlaySessionTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.out = Path(temp.name) / "play"

    def wait_for(self, session, predicate):
        state = session.latest()
        deadline = time.monotonic() + 5
        while not predicate(state["session"]):
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, state)
            state = session.pending(state["revision"], remaining)
            self.assertIsNotNone(state)
        return state

    def read_replay(self, identifier, model):
        folder = self.out / identifier / model
        manifest = json.loads((folder / "run.json").read_text(encoding="utf-8"))
        moves = [json.loads(line) for line in (folder / "moves.jsonl").read_text(encoding="utf-8").splitlines()]
        return manifest, moves

    def test_catalog_updates_publish_without_replacing_the_current_game(self):
        current = MODELS
        session = PlaySession(self.out, current, lambda _: decisions, refresh_models=lambda: current)
        session.start({"models": ["official"], "seed": 5, "cap": 1})
        finished = self.wait_for(session, lambda game: not game["active"])
        published = dict(id="new-head", name="新策略头", kind="model", training_summary="教师盘面训练。",
                         evaluation={"latest": {"games": 2, "mean_score": 14, "label": "开发评测"}})
        current = [*MODELS, published]
        update = session.pending(finished["revision"], 0)
        self.assertEqual(update["models"][-1], published)
        self.assertEqual(update["session"], finished["session"])
        self.assertIsNone(session.pending(update["revision"], 0))
        self.assertEqual(session.frame(finished["session"]["id"], 1)["games"][0]["turn"], 1)
        current = [*MODELS, dict(published, id="just-published")]
        started = session.start({"models": ["just-published"], "seed": 5, "cap": 1})
        self.assertEqual(started["session"]["games"][0]["model"], "just-published")
        self.wait_for(session, lambda game: not game["active"])

    def test_same_seed_independent_inference_and_replays(self):
        loads = []
        game_objects = {}

        def load(ids):
            loads.append(ids)

            def decide(requests):
                game_objects.update(requests)
                return decisions(requests)

            return decide

        session = PlaySession(self.out, MODELS, load)
        initial = session.latest()
        self.assertEqual(initial, dict(revision=0, session=None, models=MODELS, modes=["classic", "endless"],
                                       can_continue_endless=False))
        self.assertEqual(loads, [])
        self.assertIsNone(session.pending(0, 0))

        with patch("play_session.secrets.randbelow", return_value=123) as choose_seed:
            start = session.start(dict(models=["official", "teacher"], cap=3))
        choose_seed.assert_called_once_with(MAX_SEED + 1)
        self.assertEqual(start["session"]["seed"], 123)
        self.assertEqual(start["session"]["status"], "loading")
        self.assertEqual(start["session"]["games"][0]["board"], start["session"]["games"][1]["board"])
        result = self.wait_for(session, lambda play: not play["active"])
        self.assertGreater(result["revision"], start["revision"])
        self.assertEqual(result["session"]["status"], "completed")
        self.assertFalse(result["session"]["saved"])
        self.assertFalse(self.out.exists())
        self.assertEqual(loads, [("official", "teacher")])
        self.assertIsNot(game_objects["official"], game_objects["teacher"])
        self.assertNotEqual(game_objects["official"].state, game_objects["teacher"].state)

        identifier = result["session"]["id"]
        initial = session.frame(identifier, 0)
        self.assertEqual([game["turn"] for game in initial["games"]], [0, 0])
        self.assertEqual(initial["games"][0]["board"], start["session"]["games"][0]["board"])
        self.assertNotIn("action", initial["games"][0])
        saved = session.save(dict(id=identifier))
        self.assertTrue(saved["session"]["saved"])
        self.assertGreater(saved["revision"], result["revision"])
        self.assertEqual(session.save(dict(id=identifier)), saved)

        for snapshot in result["session"]["games"]:
            model = snapshot["model"]
            manifest, moves = self.read_replay(result["session"]["id"], model)
            self.assertEqual(manifest["variant"], snapshot["name"])
            self.assertEqual(manifest["seeds"], [123])
            self.assertEqual(manifest["files"], {"moves": "moves.jsonl"})
            self.assertEqual(manifest["run_id"], result["session"]["id"])
            self.assertEqual([row["seq"] for row in moves], [1, 2, 3])
            recorded = Game.seeded(123)
            label = "action_values" if model == "teacher" else "probabilities"
            other = "probabilities" if model == "teacher" else "action_values"
            for turn, row in enumerate(moves, 1):
                frame = next(view for view in session.frame(identifier, turn)["games"] if view["model"] == model)
                opts = legal(recorded.state)
                self.assertEqual(row["before"], nested(recorded.state))
                self.assertNotIn(other, row)
                self.assertEqual(list(row[label]), opts)
                self.assertEqual(row["action"], max(opts, key=row[label].get))
                gain, spawn = recorded.step(row["action"])
                self.assertEqual((row["turn"], row["board"], row["score"], row["tile"], row["gained"], row["spawn"]),
                                 (turn, nested(recorded.state), recorded.score, max(recorded.state), gain, spawn))
                self.assertEqual(row["stop"], "cap" if turn == 3 else None)
                self.assertEqual([frame[k] for k in ("before", "board", "score", "action", label)],
                                 [row[k] for k in ("before", "board", "score", "action", label)])
            self.assertEqual(snapshot["score"], recorded.score)
            self.assertEqual(snapshot["status"], "cap")

        # Returned state can be freely adapted for JSON clients.
        result["models"][0]["name"] = "changed"
        result["session"]["games"][0]["board"][0][0] = -1
        initial["games"][0]["board"][0][0] = -1
        self.assertEqual(session.latest()["models"], MODELS)
        self.assertGreaterEqual(session.latest()["session"]["games"][0]["board"][0][0], 0)
        self.assertGreaterEqual(session.frame(identifier, 0)["games"][0]["board"][0][0], 0)

    def test_stop_occurs_between_moves_and_stale_ids_are_rejected(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)

        def decide(requests):
            if requests[0][1].turn == 1:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
            return decisions(requests)

        session = PlaySession(self.out, MODELS, lambda ids: decide)
        start = session.start(dict(models=["teacher", "official"], seed=912, cap=100))
        current = start["session"]["id"]
        self.assertTrue(entered.wait(5))
        for call in [
            lambda: session.start(dict(models=["teacher"])),
            lambda: session.stop(dict(id="stale")),
            lambda: session.save(dict(id=current)),
        ]:
            with self.assertRaises(PlayRequestError) as rejected:
                call()
            self.assertEqual(rejected.exception.code, 409)
        paused = session.pause(dict(id=current))
        self.assertEqual(paused["session"]["status"], "paused")
        stopping = session.stop(dict(id=current))
        self.assertEqual(stopping["session"]["status"], "stopping")
        self.assertEqual(session.stop(dict(id=current))["revision"], stopping["revision"])
        self.assertEqual([game["turn"] for game in stopping["session"]["games"]], [1, 1])
        release.set()
        stopped = self.wait_for(session, lambda play: not play["active"])
        self.assertEqual(stopped["session"]["status"], "stopped")
        self.assertFalse(self.out.exists())
        self.assertTrue(session.save(dict(id=current))["session"]["saved"])
        for game in stopped["session"]["games"]:
            _, trace = self.read_replay(current, game["model"])
            self.assertEqual(game["status"], "stopped")
            self.assertEqual([(row["seq"], row["turn"], row["type"]) for row in trace],
                             [(1, 1, "move"), (2, 1, "stopped")])
            self.assertEqual(trace[-1]["board"], game["board"])

        restart = session.start(dict(models=["other-head"], seed=0, cap=1))
        self.assertNotEqual(restart["session"]["id"], current)
        self.assertEqual(self.wait_for(session, lambda play: not play["active"])["session"]["status"], "completed")
        self.assertTrue((self.out / current / "official" / "moves.jsonl").is_file())

    def test_pause_drops_uncommitted_decision_and_resume_uses_same_position(self):
        entered = threading.Event()
        release = threading.Event()
        self.addCleanup(release.set)
        called = []

        def decide(requests):
            game = requests[0][1]
            called.append((game.turn, game.state))
            if len(called) == 2:
                entered.set()
                if not release.wait(5):
                    raise RuntimeError("test timed out")
            return decisions(requests)

        session = PlaySession(self.out, MODELS, lambda _: decide)
        current = session.start(dict(models=["official"], seed=123, cap=3))["session"]["id"]
        self.assertTrue(entered.wait(5))
        paused = session.pause(dict(id=current))
        self.assertEqual([game["turn"] for game in paused["session"]["games"]], [1])
        self.assertEqual(session.pause(dict(id=current)), paused)
        release.set()
        self.assertIsNone(session.pending(paused["revision"], 0.05))
        self.assertEqual(session.frame(current, 1)["games"][0]["turn"], 1)
        for bad in (-1, 2, None, True):
            with self.subTest(turn=bad), self.assertRaises(PlayRequestError) as rejected:
                session.frame(current, bad)
            self.assertEqual(rejected.exception.code, 400)
        resumed = session.resume(dict(id=current))
        self.assertEqual(resumed["session"]["status"], "running")
        self.assertEqual(resumed["session"]["games"][0]["turn"], 1)
        ended = self.wait_for(session, lambda play: not play["active"])
        self.assertEqual(ended["session"]["games"][0]["turn"], 3)
        self.assertEqual([turn for turn, _ in called], [0, 1, 1, 2])
        self.assertEqual(called[1], called[2])

    def test_older_unsaved_record_is_discarded_and_shorter_game_is_held(self):
        session = PlaySession(self.out, MODELS, lambda _: decisions)
        first = session.start(dict(models=["teacher", "official"], seed=0, cap=100))["session"]["id"]
        result = self.wait_for(session, lambda play: not play["active"])["session"]
        self.assertEqual([game["turn"] for game in result["games"]], [91, 68])
        self.assertEqual([game["turn"] for game in session.frame(first, 65)["games"]], [65, 65])
        last_official = session.frame(first, 68)["games"][1]
        later = session.frame(first, 85)["games"]
        self.assertEqual(later[0]["turn"], 85)
        self.assertEqual(later[1], last_official)
        self.assertEqual(later[1]["status"], "terminal")
        latest = session.start(dict(models=["official"], seed=1, cap=1))["session"]
        self.assertNotEqual(latest["id"], first)
        for action in (lambda: session.frame(first, 0), lambda: session.save(dict(id=first))):
            with self.assertRaises(PlayRequestError) as rejected:
                action()
            self.assertEqual(rejected.exception.code, 409)
        self.assertFalse(self.out.exists())
        self.wait_for(session, lambda play: not play["active"])

    def test_endless_tracks_continue_past_2048_and_4096_and_star_at_their_own_terminal(self):
        opening = (1024, 1024, 1024, 1024, 16, 32, 64, 128, 256, 512, 16, 32, 64, 128, 256, 512)

        def decide(requests):
            result = {}
            for key, game in requests:
                moves = legal(game.state)
                preferred = "right" if key == "teacher" else "left"
                action = preferred if preferred in moves else moves[0]
                field = "action_values" if key == "teacher" else "probabilities"
                result[key] = {field: {move: float(move == action) for move in moves}, "decision_ms": 0}
            return result

        play = PlaySession(self.out, MODELS, lambda _: decide)
        request = dict(models=["official", "teacher"], seed=1)
        with patch.object(Game, "seeded", side_effect=lambda seed: Game(opening, random.Random(seed))):
            play.start(request)
            classic = self.wait_for(play, lambda session: not session["active"])["session"]
            self.assertEqual((classic["mode"], classic["cap"]), ("classic", MAX_TURNS))
            self.assertEqual([(g["status"], g["turn"], g["max_tile"]) for g in classic["games"]],
                             [("target", 1, 2048)] * 2)

            play.start(dict(request, mode="endless"))
            endless = self.wait_for(play, lambda session: not session["active"])["session"]
        self.assertEqual((endless["mode"], endless["cap"], endless["status"]), ("endless", None, "completed"))
        identifier = endless["id"]
        for step, tile in [(1, 2048), (2, 4096)]:
            self.assertEqual([(g["status"], g["max_tile"]) for g in play.frame(identifier, step)["games"]],
                             [("running", tile)] * 2)
        self.assertEqual([g["turn"] for g in endless["games"]], [6, 4])
        self.assertEqual(play.frame(identifier, 6)["games"][1], play.frame(identifier, 4)["games"][1])
        self.assertTrue(play.save(dict(id=identifier))["session"]["saved"])
        for game in endless["games"]:
            self.assertEqual(game["status"], "terminal")
            self.assertFalse(legal(tuple(tile for row in game["board"] for tile in row)))
            manifest, rows = self.read_replay(identifier, game["model"])
            self.assertEqual((manifest["mode"], manifest["max_moves_per_game"]), ("endless", None))
            self.assertTrue(all(row["mode"] == "endless" and row["cap"] is None for row in rows))
            self.assertEqual([row["stop"] for row in rows], [None] * (game["turn"] - 1) + ["terminal"])
            self.assertEqual(rows[-1]["board"], game["board"])

    def test_endless_move_at_and_beyond_the_classic_turn_limit(self):
        game = Game.seeded(123)
        game.turn = MAX_TURNS - 1
        track = _Track(game, {}, [{"board": nested(game.state)}])
        play = PlaySession(self.out, MODELS, lambda _: decisions)
        for turn in (MAX_TURNS, MAX_TURNS + 1):
            play._move({"mode": "endless", "cap": None}, track, legal(game.state)[0], {})
            self.assertEqual((track.view["turn"], track.view["status"]), (turn, "running"))

    def test_continue_winning_legacy_group_keeps_rng_replays_and_dead_tracks(self):
        opening = (64, 2, 32, 128, 8, 4, 256, 4, 512, 2, 0, 64, 128, 64, 1024, 1024)
        first = {"official": "left", "other-head": "down", "teacher": "up"}
        loads, batches = [], []

        def choose(key, game):
            return first[key] if game.turn == 0 else legal(game.state)[0]

        def load(ids):
            loads.append(ids)

            def decide(requests):
                batches.append([(key, game.turn) for key, game in requests])
                result = {}
                for key, game in requests:
                    action = choose(key, game)
                    field = "action_values" if key == "teacher" else "probabilities"
                    result[key] = {field: {m: float(m == action) for m in legal(game.state)}, "decision_ms": 0}
                return result

            return decide

        play = PlaySession(self.out, MODELS, load)
        with patch.object(Game, "seeded", side_effect=lambda seed: Game(opening, random.Random(seed))):
            original = play.start(dict(models=list(first), seed=1, cap=1))["session"]["id"]
        won = self.wait_for(play, lambda s: not s["active"])
        self.assertEqual([g["status"] for g in won["session"]["games"]], ["target", "cap", "terminal"])
        del play._session["mode"]  # The workbench may have restored a pre-mode in-memory session.
        saved = play.save(dict(id=original))
        self.assertTrue(saved["can_continue_endless"])
        previous_frames = play.frame(original, 1)["games"]
        files = {file: file.read_bytes() for file in (self.out / original).glob("*/*")}

        with patch.object(threading.Thread, "start", side_effect=RuntimeError("thread unavailable")):
            with self.assertRaises(RuntimeError):
                play.continue_endless(dict(id=original))
        self.assertEqual(play.latest(), saved)

        gate = threading.Barrier(2)
        def continue_once():
            gate.wait(3)
            try:
                return play.continue_endless(dict(id=original))
            except PlayRequestError as error:
                return error.code

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(continue_once) for _ in range(2)]
            replies = [job.result(timeout=5) for job in futures]
        self.assertEqual(replies.count(409), 1)
        start, = (reply["session"] for reply in replies if isinstance(reply, dict))
        new_id = start["id"]
        self.assertNotEqual(new_id, original)
        self.assertEqual((start["mode"], start["cap"], start["seed"], start["saved"]), ("endless", None, 1, False))
        self.assertEqual([g["turn"] for g in start["games"]], [1, 1, 1])
        final = self.wait_for(play, lambda s: not s["active"])
        self.assertFalse(final["can_continue_endless"])
        self.assertEqual(loads, [tuple(first), ("official", "other-head")])
        self.assertEqual(batches[1], [("official", 1), ("other-head", 1)])
        for old, new in zip(previous_frames, play.frame(new_id, 1)["games"], strict=True):
            self.assertEqual(new, dict(old, status="terminal" if old["model"] == "teacher" else "running"))
        self.assertTrue(play.save(dict(id=new_id))["session"]["saved"])
        for view in final["session"]["games"]:
            key = view["model"]
            replay = Game(opening, random.Random(1))
            manifest, rows = self.read_replay(new_id, key)
            self.assertEqual((manifest["mode"], manifest["run_id"]), ("endless", new_id))
            for row in rows:
                action = choose(key, replay)
                gain, spawn = replay.step(action)
                self.assertEqual((row["action"], row["gained"], row["spawn"], row["board"], row["score"]),
                                 (action, gain, spawn, nested(replay.state), replay.score))
            self.assertEqual([r["stop"] for r in rows], [None] * (len(rows)-1) + ["terminal"])
            self.assertFalse(legal(replay.state))
        self.assertTrue(all(file.read_bytes() == content for file, content in files.items()))

    def test_dead_target_is_ineligible_to_continue(self):
        opening = (1024, 1024, 8, 16, 2, 32, 64, 128, 256, 512, 16, 32, 64, 128, 256, 512)
        play = PlaySession(self.out, MODELS, lambda _: decisions)
        with patch.object(Game, "seeded", side_effect=lambda seed: Game(opening, random.Random(seed))):
            identifier = play.start(dict(models=["official"], seed=1))["session"]["id"]
        final = self.wait_for(play, lambda session: not session["active"])
        self.assertEqual(final["session"]["games"][0]["status"], "target")
        self.assertFalse(final["can_continue_endless"])
        with self.assertRaises(PlayRequestError) as rejected:
            play.continue_endless(dict(id=identifier))
        self.assertEqual(rejected.exception.code, 409)
        self.assertEqual(play.latest(), final)

    def test_starring_a_restored_session_that_predates_mode(self):
        play = PlaySession(self.out, MODELS, lambda _: decisions)
        identifier = play.start(dict(models=["official"], cap=1))["session"]["id"]
        self.wait_for(play, lambda session: not session["active"])
        del play._session["mode"]
        previous = play.frame(identifier, 1)
        self.assertTrue(play.save(dict(id=identifier))["session"]["saved"])
        manifest, rows = self.read_replay(identifier, "official")
        self.assertEqual((manifest["mode"], manifest["max_moves_per_game"]), ("classic", 1))
        self.assertEqual((rows[0]["mode"], rows[0]["stop"]), ("classic", "cap"))
        self.assertEqual(play.frame(identifier, 1), previous)

    def test_failed_save_leaves_no_partial_archive_and_can_retry(self):
        session = PlaySession(self.out, MODELS, lambda _: decisions)
        current = session.start(dict(models=["official", "teacher"], seed=5, cap=1))["session"]["id"]
        ended = self.wait_for(session, lambda play: not play["active"])
        original = session._write_replay

        def fail_second(directory, run, track):
            if track.view["kind"] == "teacher":
                raise OSError("simulated second track write failure")
            return original(directory, run, track)

        with patch.object(session, "_write_replay", side_effect=fail_second):
            with self.assertLogs("play_session", level="ERROR"), self.assertRaises(PlayRequestError) as rejected:
                session.save(dict(id=current))
            self.assertEqual(rejected.exception.code, 500)
        self.assertEqual(session.latest(), ended)
        self.assertFalse((self.out / current).exists())
        self.assertFalse(list(self.out.parent.glob(".instant-play-save-*")))
        self.assertTrue(session.save(dict(id=current))["session"]["saved"])
        self.assertEqual({item.name for item in (self.out / current).iterdir()}, {"official", "teacher"})

    def test_invalid_request_does_not_start_session(self):
        session = PlaySession(self.out, MODELS, lambda _: self.fail("Unexpected model load"))
        for body in [
            None, {}, {"models": []}, {"models": {"teacher": True}}, {"models": ["teacher", "teacher"]},
            {"models": [{}]}, {"models": ["../outside"]}, {"models": ["unregistered"]},
            {"models": ["teacher"], "seed": MAX_SEED + 1}, {"models": ["teacher"], "seed": True},
            {"models": ["teacher"], "cap": 5001}, {"models": ["teacher"], "cap": 0},
            {"models": ["teacher"], "mode": "endless", "cap": 100},
            {"models": ["teacher"], "mode": "classic", "cap": None},
            {"models": ["teacher"], "mode": "other"}, {"models": ["teacher"], "mode": {}},
            {"models": ["teacher"], "path": "forbidden"},
        ]:
            with self.subTest(body=body), self.assertRaises(PlayRequestError) as rejected:
                session.start(body)
            self.assertEqual(rejected.exception.code, 400)
        with self.assertRaises(PlayRequestError):
            session.stop({"path": "forbidden"})
        self.assertIsNone(session.latest()["session"])
        self.assertFalse(self.out.exists())

    def test_selection_limit_applies_to_the_number_of_active_policies(self):
        catalog = [dict(id=f"policy-{n}", name=f"Policy {n}", kind="model") for n in range(MAX_MODELS + 1)]
        session = PlaySession(self.out, catalog, lambda _: decisions)
        ids = [option["id"] for option in catalog]
        with self.assertRaises(PlayRequestError):
            session.start(dict(models=ids, cap=1))
        session.start(dict(models=ids[:MAX_MODELS], seed=17, cap=1))
        completed = self.wait_for(session, lambda play: not play["active"])["session"]
        self.assertEqual(completed["status"], "completed")
        self.assertEqual(len(completed["games"]), MAX_MODELS)

    def test_failed_inference_preserves_only_committed_moves(self):
        def decide(requests):
            result = decisions(requests)
            if requests[0][1].turn:
                result["teacher"]["action_values"] = {"up": float("nan")}
            return result

        session = PlaySession(self.out, MODELS, lambda _: decide)
        with self.assertLogs("play_session", level="ERROR"):
            start = session.start(dict(models=["official", "teacher"], seed=420, cap=4))
            result = self.wait_for(session, lambda play: not play["active"])["session"]
        self.assertEqual(result["status"], "error")
        self.assertTrue(result["message"])
        self.assertFalse(self.out.exists())
        self.assertTrue(session.save(dict(id=result["id"]))["session"]["saved"])
        for item in result["games"]:
            self.assertEqual((item["turn"], item["status"]), (1, "error"))
            _, trace = self.read_replay(start["session"]["id"], item["model"])
            self.assertEqual([(row["turn"], row["type"]) for row in trace], [(1, "move"), (1, "error")])


if __name__ == "__main__":
    unittest.main()

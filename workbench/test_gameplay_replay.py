"""Exercise the evaluation replay contract with recorded, synthetic local episodes."""

import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from live import Game, GameplayWatch, Server, Watch, legal, nested
from gameplay_replay import FRAME_FIELDS, GameplayReplay, file_stamp


def recorded(seed, count, name):
    game = Game.seeded(seed)
    rows = []
    for turn in range(1, count+1):
        before = nested(game.state)
        moves = legal(game.state)
        action = moves[-1] if name == "alternate" else moves[0]
        gained, spawn = game.step(action)
        rows.append(dict(seed=seed, turn=turn, before=before, after=nested(game.state), legal_moves=moves,
                         action=action, probabilities={k:int(k==action) for k in moves}, score=game.score,
                         max_tile=max(game.state), gained=gained, spawn=spawn,
                         stop="capped" if turn == count else None, private_extra="displayed nowhere"))
    return rows


def encode(rows):
    return b"".join(json.dumps(row).encode()+b"\n" for row in rows)


class GameplayReplayTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name).resolve()
        self.seed = 17
        self.arm = "official"
        self.file = self.root/self.arm/str(self.seed)/"moves.jsonl"
        self.file.parent.mkdir(parents=True)
        self.rows = recorded(self.seed, 5, self.arm)
        self.reader = GameplayReplay()
        self.source = (self.arm, self.file, False, None)

    def page(self, start=1, limit=256, source=None):
        return self.reader.page([source or self.source], self.seed, start, limit, time.time())["tracks"][0]

    def test_live_append_partial_rewrites_and_stable_page_cache(self):
        self.file.write_bytes(encode(self.rows[:2])+b'{"seed":')
        first = self.page()
        self.assertEqual([row["turn"] for row in first["frames"]], [1, 2])
        with patch("gameplay_replay._prefix", side_effect=AssertionError("Reopening unchanged moves")):
            self.assertEqual(self.page(2, 1)["frames"], first["frames"][1:])
        self.file.write_bytes(encode(self.rows[:4]))
        self.assertEqual([r["turn"] for r in self.page(3, 2)["frames"]], [3, 4])
        self.file.write_bytes(encode(self.rows))
        self.assertEqual(self.page(99, 1)["last_frame"], {k:self.rows[-1][k] for k in FRAME_FIELDS})
        self.file.write_bytes(encode(self.rows[:1]))
        self.assertEqual(self.page()["total_turns"], 1)

    def test_damaged_sequences_and_unverified_replacements_are_hidden(self):
        for rows in ([self.rows[1]], [dict(self.rows[0], seed=self.seed+1)],
                     [dict(self.rows[0], score=99)], [dict(self.rows[0], probabilities={"left":1})],
                     [dict(self.rows[0], stop="target"), self.rows[1]]):
            with self.subTest(rows=rows):
                self.file.write_bytes(encode(rows))
                result = self.page()
                self.assertEqual((result["status"], result["frames"]), ("invalid", []))
        self.file.write_bytes(encode(self.rows))
        pinned = (self.arm, self.file, True, file_stamp(self.file.stat()))
        self.assertEqual(self.page(source=pinned)["total_turns"], 5)
        self.file.write_bytes(encode(self.rows[:2]))
        self.assertEqual(self.page(source=pinned)["status"], "invalid")
        self.file.unlink()
        self.assertEqual(self.page()["status"], "waiting")

    def test_linked_moves_are_never_served(self):
        outside = self.root/"outside"
        outside.write_bytes(encode(self.rows))
        try:
            os.link(outside, self.file)
        except (OSError, NotImplementedError) as exc:
            self.skipTest(str(exc))
        self.assertEqual(self.page()["status"], "invalid")

    def test_http_pages_all_arms_and_short_tracks_in_the_same_turn_range(self):
        self.file.write_bytes(encode(self.rows))
        short = self.root/"alternate"/str(self.seed)/"moves.jsonl"
        short.parent.mkdir(parents=True)
        short.write_bytes(encode(recorded(self.seed, 2, "alternate")))
        arms = ["alternate", self.arm, "waiting"]
        (self.root/"resolved-config.json").write_text(json.dumps(dict(schema_version=1,
            checkpoints=dict.fromkeys(arms, "private/model/path"), seeds=[17, 23], screening_seeds=[17],
            config={"seed_group_size":2})), encoding="utf8")
        with Server(0, None, self.root/"replay.html", [GameplayWatch(1, "New evaluation", self.root)]) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            url = f"http://127.0.0.1:{server.server_port}/gameplay/replay?"
            def get(**params):
                query = dict(channel="watch-1", seed=self.seed)
                query.update(params)
                with urlopen(url+urlencode(query), timeout=5) as res:
                    return json.load(res)
            try:
                first = get(limit=2)
                self.assertEqual((first["channel"], first["seed"], first["start"], first["total_turns"], first["next_start"]),
                                 ("watch-1", self.seed, 1, 5, 3))
                self.assertEqual([track["arm"] for track in first["tracks"]], [self.arm, "alternate", "waiting"])
                self.assertNotIn("private", json.dumps(first))
                later = get(start=3, limit=2)
                long, short, pending = later["tracks"]
                self.assertEqual([r["turn"] for r in long["frames"]], [3, 4])
                self.assertEqual((short["frames"], short["last_frame"]["turn"], short["last_frame"]["stop"]), ([], 2, "capped"))
                self.assertEqual((pending["status"], pending["total_turns"]), ("waiting", 0))
                self.assertIsNone(get(start=5, limit=2)["next_start"])
                self.assertEqual(get(seed=23)["total_turns"], 0)
                for params, status in (({"seed":99}, 404), ({"channel":"watch-2"}, 404), ({"seed":"../17"}, 400),
                                       ({"limit":257}, 400), ({"start":0}, 400), ({"file":"moves.jsonl"}, 400)):
                    with self.subTest(params=params), self.assertRaises(HTTPError) as error:
                        get(**params)
                    self.assertEqual(error.exception.code, status)
            finally:
                server.shutdown()
                thread.join()


class NativeGameplayReplayTests(unittest.TestCase):
    def setUp(self):
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.model = Path(temp.name).resolve()/"official-kev"
        self.model.mkdir()
        self.path = self.model/"result.json"
        self.moves = self.model/"moves.jsonl"
        self.watch = Watch(1, f"Model:evaluation:{self.path}")
        self.games, self.rows = [], []
        for seed in (19, 27):
            records = recorded(seed, 3, "official")
            for full in records:
                row = {k:full[k] for k in ("seed", "turn", "before", "after", "action", "score", "probabilities")}
                self.rows.append(dict(row, tile=full["max_tile"]))
            self.games.append(dict(seed=seed, turn=3, score=full["score"], tile=full["max_tile"], stop="cap"))
        self.summary = dict(cap=3, checkpoint_head_sha256="a"*64, games=self.games, device="cpu")
        self.path.write_text(json.dumps(self.summary), encoding="utf8")
        self.moves.write_bytes(encode(self.rows))

    def test_native_result_routes_keep_one_model_and_check_full_aggregate(self):
        with Server(0, None, self.model.parent/"replay.html", [self.watch]) as server:
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            def get(route):
                with urlopen(f"http://127.0.0.1:{server.server_port}"+route, timeout=5) as res:
                    return json.load(res)
            try:
                shape, = get("/channels")["channels"]
                self.assertEqual((shape["mode"], shape["arms"], shape["seed_catalog"], shape["games"], shape["total_games"]),
                                 ("gameplay", ["official-kev"], [19, 27], 2, 2))
                args = "?channel=watch-1&seed=27"
                with patch.object(self.watch.native_replay, "_build", side_effect=AssertionError("Rebuilt stable recording")):
                    latest, = get("/gameplay/seed"+args)["episodes"]
                    page = get("/gameplay/replay"+args+"&start=2&limit=1")
                    frame, = page["tracks"][0]["frames"]
                    self.assertEqual((page["total_turns"], page["next_start"]), (3, 3))
                    self.assertEqual((frame["turn"], frame["stop"], frame["probabilities"]), (2, None, self.rows[4]["probabilities"]))
                    final, = get("/gameplay/replay"+args+"&start=3")["tracks"][0]["frames"]
                    self.assertEqual((final["score"], final["max_tile"], final["stop"]), (latest["score"], latest["max_tile"], "cap"))
                    self.assertEqual(get("/gameplay/replay"+args+"&start=4")["tracks"][0]["last_frame"], latest)
                self.games[0]["score"] += 1
                self.path.write_text(json.dumps(self.summary), encoding="utf8")
                self.assertEqual(get("/channels")["channels"][0]["mode"], "static_result")
                for route in ("/gameplay/seed", "/gameplay/replay"):
                    with self.assertRaises(HTTPError) as err:
                        get(route+args)
                    self.assertEqual(err.exception.code, 404)
            finally:
                server.shutdown()
                thread.join()

    def test_only_verified_trajectories_promote_a_native_summary(self):
        for key, value in (("seed", 17), ("turn", 2), ("tile", 256), ("stop", "terminal")):
            with self.subTest(key=key):
                invalid = dict(self.summary, games=[dict(self.games[0], **{key:value}), self.games[1]])
                self.path.write_text(json.dumps(invalid), encoding="utf8")
                self.assertEqual(self.watch.snapshot(time.time())["mode"], "static_result")
        self.path.write_text(json.dumps(self.summary), encoding="utf8")
        for damaged in (encode(self.rows[:-1]), encode(self.rows)+b"{", encode(self.rows+self.rows[-1:])):
            self.moves.write_bytes(damaged)
            self.assertEqual(self.watch.snapshot(time.time())["mode"], "static_result")
        self.moves.unlink()
        elsewhere = self.model.parent/"other"
        elsewhere.write_bytes(encode(self.rows))
        try:
            os.link(elsewhere, self.moves)
        except (OSError, NotImplementedError):
            return
        self.assertEqual(self.watch.snapshot(time.time())["mode"], "static_result")

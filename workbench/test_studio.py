"""Verify new runs and appended training samples through the public studio routes."""

from datetime import UTC, datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
import unittest
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import urlopen

from live import Server
from training_boards import TRAINING_SCHEMA


class StudioTests(unittest.TestCase):
    def test_new_training_is_discovered_and_appended_batches_are_browsable(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            runs = root / "runs"
            runs.mkdir()
            server = Server(0, None, root / "replay.html", training_root=runs)
            server.training_discovery.interval = 0
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()

            def get(route, **query):
                address = f"http://127.0.0.1:{server.server_port}{route}"
                if query:
                    address += "?" + urlencode(query)
                with urlopen(address, timeout=5) as response:
                    return json.load(response)

            try:
                self.assertEqual(get("/channels")["channels"], [])
                log = runs / "unseen-experiment" / "fit" / "progress.jsonl"
                log.parent.mkdir(parents=True)
                candidate = {"choice": "down", "probabilities": {"up": None, "down": .7, "left": None, "right": .3}}
                sample = {"row": 7, "best_actions": ["right"], "board": [[2, 4, 0, 0]] + [[0]*4 for _ in range(3)],
                          "candidate": candidate}
                step = dict(kind="training", schema=TRAINING_SCHEMA, phase="optimizer",
                            status="running", trial="next-model", epoch=1, optimizer_step=1, steps=3,
                            mean_training_loss=.42, batch_size=8, samples=[sample], timestamp=datetime.now(UTC).isoformat())
                log.write_text(json.dumps(step) + "\n", encoding="utf-8")
                watch, = get("/channels")["channels"]
                channel = watch["id"]
                self.assertEqual(watch["kind"], "training")
                first, = get("/training/batches", channel=channel)["batches"]
                detail = get("/training/batch", channel=channel, key=first["key"])
                self.assertEqual(detail["samples"], [dict(sample, candidate_source="recorded")])

                step.update(optimizer_step=3, mean_training_loss=.25)
                with log.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(step))
                    stream.flush()
                    self.assertEqual(get("/training/batches", channel=channel)["batches"], [first])
                    stream.write("\n")
                batches = get("/training/batches", channel=channel)["batches"]
                self.assertEqual([(b["step"], b["loss"]) for b in batches], [(1, .42), (3, .25)])
                self.assertEqual(get("/channels")["channels"][0]["id"], channel)
                for route, fields, status in (
                    ("/training/batches", {"channel": "unknown"}, 404),
                    ("/training/batch", {"channel": channel, "key": "../progress.jsonl"}, 400),
                ):
                    with self.assertRaises(HTTPError) as error:
                        get(route, **fields)
                    self.assertEqual(error.exception.code, status)
            finally:
                server.shutdown()
                thread.join()
                server.server_close()

"""Verify the read-only LoRA batch lookup against pinned synthetic inputs."""

import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

from kev2048_release.game import flatten, legal
from kev2048_release.artifacts import identity

from training_boards import LORA_IMPLEMENTATION, TrainingBoards, recorded_batch


def write_resolved(path, metadata):
    claimed = identity(metadata)
    path.write_text(json.dumps(dict(metadata, identity=claimed)), encoding="utf-8")
    return claimed


def fit(**fields):
    return dict({"phase": "lora_fit", "trial": "lora", "epoch": 1, "optimizer_step": 110, "steps": 125}, **fields)


class RecordedCandidateTests(unittest.TestCase):
    def test_candidate_is_optional_and_matches_the_legal_moves(self):
        board = [[2, 4, 0, 0], [0]*4, [0]*4, [0]*4]
        sample = dict(row=1, board=board, best_actions=["right"])
        probabilities = dict(up=None, down=.7, left=None, right=.3)
        record = dict(batch_size=1, samples=[sample], samples_seen=7, boards_per_second=3.5)
        def publish():
            return recorded_batch(record, fit())
        self.assertNotIn("candidate", publish()["samples"][0])
        candidate = dict(probabilities=probabilities, choice="down")
        sample["candidate"] = dict(candidate, unrelated=True)
        self.assertEqual(publish()["samples"][0]["candidate"], candidate)
        self.assertEqual((publish()["samples_seen"], publish()["boards_per_second"]), (7, 3.5))
        for invalid in (dict(candidate, choice="right"), dict(candidate, choice="left"),
                        dict(candidate, probabilities=dict(probabilities, up=0)),
                        dict(candidate, probabilities=dict(probabilities, right=float("nan"))),
                        dict(candidate, probabilities={"down": .7, "right": .3}),
                        dict(candidate, probabilities=dict(probabilities, right=.8))):
            with self.subTest(invalid=invalid):
                sample["candidate"] = invalid
                row = publish()["samples"][0]
                self.assertNotIn("candidate", row)
                self.assertEqual(row["best_actions"], ["right"])


class TrainingBoardsTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.workspace = Path(workspace.name).resolve()
        self.root = self.workspace / "runs"
        self.run = self.root / "round3"
        self.lora = self.run / "lora"
        self.lora.mkdir(parents=True)
        self.dataset = self.root / "preparation" / "dataset"
        self.dataset.mkdir(parents=True)
        self.path = self.dataset / "train-2000.jsonl"
        self.rows = []
        for index in range(2000):
            start = 2 ** (1 + index % 7)
            board = [[0, 0, 0, 0], [0, start, 2, 0], [0, 4, 0, 0], [0, 0, 0, 0]]
            self.rows.append(dict(board=board, split="train", stage=("early", "middle", "late")[index % 3],
                                  source_policy="official", valid_moves=legal(flatten(board)),
                                  best_actions=["up", "left"]))
        raw = ("\n".join(json.dumps(row) for row in self.rows) + "\n").encode()
        self.path.write_bytes(raw)
        self.parent = dict(config={"data": str(self.dataset)},
                           data_files={self.path.name: hashlib.sha256(raw).hexdigest()})
        self.branch = dict(train_sha256=identity(self.rows), implementation_sha256=LORA_IMPLEMENTATION,
                           seed=96729100, accumulation=16, microbatch=1, pilot_steps=32)
        self.write_identity_chain()
        self.boards = TrainingBoards(self.root)

    def write_identity_chain(self):
        parent_id = write_resolved(self.run / "resolved-config.json", self.parent)
        write_resolved(self.lora / "resolved-config.json", dict(self.branch, parent_identity=parent_id))

    def assert_batch_order(self, event):
        batch = self.boards.batch(self.run, event)
        self.assertIsNotNone(batch)
        self.assertEqual({key: batch[key] for key in ("trial", "epoch", "optimizer_step", "steps")},
                         {key: event[key] for key in ("trial", "epoch", "optimizer_step", "steps")})
        order = list(range(2000))
        random.Random(96729100 + event["epoch"]).shuffle(order)
        start = (event["optimizer_step"] - 1) * 16
        expected = order[start:start + 16]
        self.assertEqual([row["row"] for row in batch["samples"]], [index + 1 for index in expected])
        fields = ("board", "stage", "source_policy", "best_actions")
        for sample, index in zip(batch["samples"], expected, strict=True):
            self.assertEqual({key: sample[key] for key in fields}, {key: self.rows[index][key] for key in fields})
        return batch

    def test_exact_batches_and_cache_reuse(self):
        first = self.assert_batch_order(fit())
        first["samples"][0]["board"][0][0] = 99
        first["samples"][0]["best_actions"].append("right")
        with patch("training_boards._read", side_effect=AssertionError("Unchanged inputs should be cached")):
            self.assert_batch_order(fit())
            self.assert_batch_order(fit(trial="lora-head-control", epoch=2, optimizer_step=125))

    def test_pilot_truncations_and_unrecognized_events(self):
        self.assert_batch_order(fit(trial="lora-pilot", optimizer_step=32, steps=32))
        self.assert_batch_order(fit(trial="lora-control-timing", optimizer_step=4, steps=4))
        for event in (fit(trial="lora-pilot", optimizer_step=1, steps=32, epoch=2),
                      fit(trial="lora-control-timing", optimizer_step=4, steps=125),
                      fit(optimizer_step=126), fit(optimizer_step=True), fit(epoch=3),
                      fit(trial="unknown"), fit(phase="fit_epoch")):
            with self.subTest(event=event):
                self.assertIsNone(self.boards.batch(self.run, event))

    def test_tampered_subset_invalidates_validated_cache(self):
        self.assert_batch_order(fit())
        raw = self.path.read_bytes().replace(b'"early"', b'"late"', 1)
        self.path.write_bytes(raw)
        self.assertIsNone(self.boards.batch(self.run, fit()))
        self.parent["data_files"][self.path.name] = hashlib.sha256(raw).hexdigest()
        self.write_identity_chain()
        self.assertIsNone(self.boards.batch(self.run, fit()))

    def test_refuses_unknown_trainer_mismatched_parent_and_escaping_data(self):
        self.branch["implementation_sha256"] = "a" * 64
        self.write_identity_chain()
        self.assertIsNone(self.boards.batch(self.run, fit()))
        self.branch["implementation_sha256"] = LORA_IMPLEMENTATION
        write_resolved(self.lora / "resolved-config.json", dict(self.branch, parent_identity="b" * 64))
        self.assertIsNone(self.boards.batch(self.run, fit()))

        external = self.workspace / "external"
        external.mkdir()
        (external / self.path.name).write_bytes(self.path.read_bytes())
        self.parent["config"]["data"] = str(external)
        self.write_identity_chain()
        self.assertIsNone(self.boards.batch(self.run, fit()))
        self.assertIsNone(self.boards.batch(self.run / "lora", fit()))

    def test_refuses_linked_dataset_with_matching_checksums(self):
        link = self.root / "linked"
        try:
            link.symlink_to(self.dataset, target_is_directory=True)
        except OSError as error:
            self.skipTest(f"Cannot create symlinks on this host: {error}")
        self.parent["config"]["data"] = str(link)
        self.write_identity_chain()
        self.assertIsNone(self.boards.batch(self.run, fit()))


if __name__ == "__main__":
    unittest.main()

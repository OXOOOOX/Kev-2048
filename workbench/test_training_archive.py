"""Verify append-aware training archive reads using tiny isolated journals."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from training_archive import TrainingArchive
from training_boards import TRAINING_SCHEMA


BOARD = [[2, 4, 0, 0], [0]*4, [0]*4, [0]*4]
CANDIDATE = {"choice": "down", "probabilities": {"up": None, "down": .7, "left": None, "right": .3}}


def row(index=1, **fields):
    return dict({"row": index, "board": BOARD, "best_actions": ["right"]}, **fields)


def step(number=1, **fields):
    return dict({"kind": "training", "schema": TRAINING_SCHEMA, "phase": "optimizer",
                 "trial": "alpha", "epoch": 1, "optimizer_step": number, "steps": 10,
                 "batch_size": 1, "mean_training_loss": 1.2, "samples": [row()],
                 "timestamp": "2026-05-01T00:00:00Z"}, **fields)


def encoded(value):
    return json.dumps(value, separators=(",", ":")).encode()


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.log = self.directory/"progress.jsonl"
        self.watch = SimpleNamespace(path=self.log, path_guard=None)
        self.archive = TrainingArchive()

    def write(self, *records):
        self.log.write_bytes(b"".join(encoded(item)+b"\n" for item in records))

    def append(self, data):
        with self.log.open("ab") as stream:
            stream.write(data)

    def test_step_keys_are_stable_per_trial_and_epoch_and_sample_payloads_are_lazy(self):
        self.write(step(samples=[row(candidate=CANDIDATE)], batch_size=8),
                   step(trial="beta"), step(epoch=2), step(10, mean_training_loss=.6))
        metas = self.archive.index(self.watch)["batches"]
        self.assertEqual(len({m["key"] for m in metas}), 4)
        self.assertTrue(all("samples" not in m for m in metas))
        self.assertEqual((metas[0]["size"], metas[0]["sample_count"]), (8, 1))
        first = self.archive.batch(self.watch, metas[0]["key"])
        self.assertEqual(first["samples"], [row(candidate=CANDIDATE, candidate_source="recorded")])
        self.assertEqual(first["board_source"], "recorded")
        self.assertIsNone(self.archive.batch(self.watch, "a"*24))
        self.assertIsNone(self.archive.batch(self.watch, "../some/file"))

    def test_unchanged_index_not_reparsed_and_append_only_reads_new_committed_lines(self):
        self.write(step())
        initial = self.archive.index(self.watch)
        newer = encoded(step(10, mean_training_loss=.8))
        with patch.object(self.archive, "_add", wraps=self.archive._add) as add:
            self.assertEqual(self.archive.index(self.watch), initial)
            self.append(newer[:-4])
            self.assertEqual(self.archive.index(self.watch)["batches"], initial["batches"])
            self.assertEqual(add.call_count, 0)
            self.append(newer[-4:]+b"\n")
            after = self.archive.index(self.watch)["batches"]
            self.assertEqual([b["step"] for b in after], [1, 10])
            self.assertEqual(after[0], initial["batches"][0])
            self.assertEqual(add.call_count, 1)

    def test_duplicate_update_replaces_record_without_discarding_standard_data_for_legacy_line(self):
        first = step()
        self.write(first)
        old, = self.archive.index(self.watch)["batches"]
        self.append(encoded(first)+b"\n")
        self.assertEqual(self.archive.index(self.watch)["batches"], [old])
        self.append(encoded(step(samples=[row(6, candidate=CANDIDATE)], mean_training_loss=.9))+b"\n")
        self.append(encoded(step(phase="lora_fit", samples=None))+b"\n")
        new, = self.archive.index(self.watch)["batches"]
        self.assertEqual(new["key"], old["key"])
        self.assertNotEqual(new["revision"], old["revision"])
        self.assertEqual(self.archive.batch(self.watch, old["key"])["samples"][0]["row"], 6)

    def test_file_truncation_rotation_and_guard_invalidate_cached_batches(self):
        self.write(step(), step(3, trial="gone"))
        _, removed = self.archive.index(self.watch)["batches"]
        self.write(step(8))
        self.assertIsNone(self.archive.batch(self.watch, removed["key"]))
        self.assertEqual([b["step"] for b in self.archive.index(self.watch)["batches"]], [8])
        replacement = self.directory/"tmp"
        replacement.write_bytes(encoded(step(4))+b"\n")
        replacement.replace(self.log)
        key = self.archive.index(self.watch)["batches"][0]["key"]
        self.assertEqual(self.archive.batch(self.watch, key)["step"], 4)
        self.watch.path_guard = lambda _: False
        with self.assertRaises(ValueError):
            self.archive.batch(self.watch, key)

    def test_labels_and_candidates_stay_missing_when_absent_or_invalid(self):
        self.write(step(samples=[row(best_actions=[], candidate=CANDIDATE), row(2, candidate={"choice":"up"}), row(3)],
                        batch_size=3), step(2, batch_size=64, samples=None),
                   step(3, samples=[row(i) for i in range(1, 66)], batch_size=65))
        present, empty, oversized = self.archive.index(self.watch)["batches"]
        samples = self.archive.batch(self.watch, present["key"])["samples"]
        self.assertEqual(samples[0]["best_actions"], [])
        self.assertEqual(samples[0]["candidate"], CANDIDATE)
        self.assertNotIn("candidate", samples[1])
        self.assertNotIn("candidate", samples[2])
        self.assertEqual([m["sample_count"] for m in (empty, oversized)], [0, 0])
        self.assertEqual(self.archive.batch(self.watch, empty["key"])["samples"], [])

    def test_verified_legacy_lora_and_epoch_only_logs(self):
        boards = Mock()
        boards.batch.return_value = {"samples": [row()]}
        source = TrainingArchive(boards)
        self.write({"phase":"lora_fit","trial":"lora","epoch":1,"optimizer_step":8,"steps":12,
                    "mean_training_loss":.93},
                   {"phase":"fit_epoch","trial":"head","epoch":1,"train_teacher_loss":1.02,"dev_objective":.97},
                   {"phase":"arbitrary_fit_epoch","arm":"epsilon","epoch":3,"dev_teacher_loss":.89},
                   {"phase":"training_completed","epoch":3,"dev_teacher_loss":.89})
        index = source.index(self.watch)
        meta, = index["batches"]
        self.assertEqual((meta["board_source"], meta["size"], meta["sample_count"]), ("reconstructed", 1, 1))
        self.assertEqual(source.batch(self.watch, meta["key"])["samples"], [row()])
        self.assertEqual([(r["trial"],r["epoch"]) for r in index["epochs"]], [("epsilon",3),("head",1)])
        self.assertEqual(index["epochs"][0]["dev_loss"], .89)
        self.assertEqual(index["epochs"][1]["dev_objective"], .97)
        self.assertNotIn("dev_loss", index["epochs"][1])
        self.assertNotIn("candidate", source.batch(self.watch, meta["key"])["samples"][0])
        boards.batch.return_value = None
        self.assertEqual(source.batch(self.watch, meta["key"])["samples"], [])


if __name__ == "__main__":
    unittest.main()

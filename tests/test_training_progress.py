import json
from contextlib import nullcontext
from pathlib import Path
import random
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ModuleNotFoundError as error:
    raise unittest.SkipTest("Training progress tests require torch") from error

from kev2048_release import head_fit, improve_fit, improve_lora, round6_fit
from kev2048_release.data import digest
from kev2048_release.game import DIRS, flatten, legal
from kev2048_release.training_progress import SCHEMA, _Journal, watch_training


class Pointer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(-0.24))

    def forward(self, query, options):
        return self.weight * query[0] * options[:, 0]


def fixture(count):
    rows, features = [], []
    values = dict(zip(DIRS, (-1.0, 0.25, 0.5, 1.0), strict=True))
    for index in range(count):
        board = [[0, 0, 0, 0], [0, 2 ** (index % 10 + 1), 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        if index % 3 == 0:
            board = [[2, 0, 0, 0]] + [[0] * 4 for _ in range(3)]
        moves = legal(flatten(board))
        rows.append(dict(board=board, valid_moves=moves, best_actions=[moves[-1]], action=moves[-1],
                         action_values={move: number for number, move in enumerate(moves)},
                         stage=improve_fit.STAGES[index % 3], source_policy="teacher",
                         episode_id=str(index), episode_seed=index + 1, ply=0, crowded=False))
        features.append((torch.tensor([1.0 + index / count]), torch.tensor([[values[move]] for move in moves])))
    return rows, features


class TrainingProgressTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.out = self.root / "run"

    def events(self):
        lines = (self.out / "progress.jsonl").read_text(encoding="utf-8").splitlines()
        events = [json.loads(line) for line in lines]
        for row in events:
            self.assertEqual(row["schema"], SCHEMA)
            self.assertEqual(row["kind"], "training")
        return events

    def assert_batches(self, events, rows, order, sizes, steps):
        self.assertEqual([row["optimizer_step"] for row in events], steps)
        for row, indices in zip(events, sizes, strict=True):
            self.assertEqual(row["phase"], "optimizer")
            self.assertEqual(row["status"], "running")
            self.assertGreater(row["mean_training_loss"], 0)
            group = order[indices]
            self.assertEqual(row["batch_size"], len(group))
            self.assertEqual(row["samples_seen"], indices.stop or len(order))
            if row["optimizer_step"] == 1:
                self.assertNotIn("boards_per_second", row)
            else:
                self.assertGreater(row["boards_per_second"], 0)
            self.assertEqual([sample["row"] for sample in row["samples"]], [index + 1 for index in group])
            for sample, index in zip(row["samples"], group, strict=True):
                self.assertEqual(sample["board"], rows[index]["board"])
                self.assertEqual(sample["best_actions"], rows[index]["best_actions"])

    def assert_decision(self, sample, moves, scores):
        decision = sample["candidate"]
        expected = dict(zip(moves, scores.detach().float().softmax(-1).tolist(), strict=True))
        self.assertEqual(decision["choice"], moves[scores.argmax().item()])
        self.assertEqual(set(decision["probabilities"]), set(DIRS))
        for direction, value in decision["probabilities"].items():
            if direction in expected:
                self.assertAlmostEqual(value, expected[direction], places=6)
            else:
                self.assertIsNone(value)

    def assert_initial_head(self, event, rows, features):
        initial = Pointer()
        for sample in event["samples"]:
            index = sample["row"] - 1
            self.assert_decision(sample, rows[index]["valid_moves"], initial(*features[index]))

    def test_head_trainers_record_every_batch_and_its_pre_update_decisions_without_changing_fit(self):
        rows, features = fixture(23)
        config = head_fit.FitConfig(max_epochs=1, batch_size=2)
        order = list(range(len(rows)))
        random.Random(config.seed + 1).shuffle(order)
        for command, fit, extra in (("experiment", head_fit.fit, ()),
                                    ("improve", improve_fit.fit, (improve_fit.Recipe(),))):
            with self.subTest(command=command):
                self.out = self.root / command
                self.out.mkdir()
                pre_update = []
                baseline_head = Pointer()
                def remember(_head, _inputs, logits):
                    if logits.requires_grad:
                        pre_update.append(logits.detach().clone())
                baseline_head.register_forward_hook(remember)
                baseline = fit(baseline_head, rows, features, rows, features, config, *extra)
                with watch_training(self.out, command):
                    observed = fit(Pointer(), rows, features, rows, features, config, *extra)
                self.assertEqual(baseline["history"], observed["history"])
                self.assertEqual(baseline["head_state"], observed["head_state"])
                events = self.events()
                self.assertTrue(all(row["steps"] == 12 and row["epoch"] == 1 for row in events))
                batches = [slice(i, min(i + 2, len(rows))) for i in range(0, len(rows), 2)]
                self.assert_batches(events, rows, order, batches, list(range(1, 13)))
                samples = [sample for event in events for sample in event["samples"]]
                for sample, logits in zip(samples, pre_update, strict=True):
                    self.assert_decision(sample, rows[sample["row"] - 1]["valid_moves"], logits)

    def test_lora_reports_actual_accumulated_group(self):
        try:
            from kev2048_release import native
        except ModuleNotFoundError as error:
            self.skipTest(f"Native model runtime unavailable: {error}")
        self.out.mkdir()
        rows, features = fixture(19)
        order = list(reversed(range(len(rows))))
        parameter = torch.nn.Parameter(torch.tensor(-0.24))
        optimizer = torch.optim.SGD([parameter], lr=.01)
        pre_update = {}
        def forward(batch, shared_prefix):
            index = batch[0]["episode_seed"] - 1
            query, options = features[index]
            logits = parameter * query[0] * options[:, 0]
            self.assertNotIn(index, pre_update)
            pre_update[index] = logits.detach().clone()
            return ((logits,),)
        model = SimpleNamespace(train=lambda: None, forward_batch=forward)
        references = [{"probabilities": [1/len(row["valid_moves"])] * len(row["valid_moves"])} for row in rows]
        zeros = torch.zeros
        with (patch.object(improve_lora, "ACCUMULATION", 2),
              patch.object(improve_lora, "_optimizer", return_value=(optimizer, [parameter], [])),
              patch.object(torch.cuda, "synchronize"), patch.object(torch, "autocast", return_value=nullcontext()),
              patch.object(torch, "zeros", side_effect=lambda *args, **kwargs: zeros(*args)),
              patch.object(native, "record", side_effect=lambda tok, model, row: row), watch_training(self.out, "improve")):
            improve_lora._fit_steps(None, model, rows, references, order,
                                   improve_fit.Recipe(), False, 2, SimpleNamespace(emit=lambda *a, **k: None),
                                   "lora-head-control", time.time() + 30)
        events = self.events()
        self.assertTrue(all(row["epoch"] == 2 and row["trial"] == "lora-head-control" for row in events))
        batches = [slice(i, min(i + 2, len(rows))) for i in range(0, len(rows), 2)]
        self.assert_batches(events, rows, order, batches, list(range(1, 11)))
        self.assertEqual(len(pre_update), len(rows))
        for event in events:
            for sample in event["samples"]:
                index = sample["row"] - 1
                self.assert_decision(sample, rows[index]["valid_moves"], pre_update[index])

    def test_native_training_writes_its_successful_final_status(self):
        try:
            from kev2048_release import native
        except ModuleNotFoundError as error:
            self.skipTest(f"Native model runtime unavailable: {error}")
        rows, features = fixture(8)
        checkpoint = self.root / "initial"
        checkpoint.mkdir()
        for name in ("head.pt", *native.FILES):
            (checkpoint / name).write_text("fixture", encoding="utf-8")
        args = SimpleNamespace(base=self.root, checkpoint=checkpoint, out=self.out, device="cpu",
                               require_historical_init=False, preset="smoke", feature_cache=None)

        def load_model(_base, path, _device):
            head = Pointer()
            if path != checkpoint:
                head.load_state_dict(torch.load(path / "head.pt", weights_only=True))
            return None, SimpleNamespace(head=head), SimpleNamespace()

        with (patch.object(native, "configure"), patch.object(native, "local_model", side_effect=load_model),
              patch.object(native, "_training_features", return_value=(features, {}, {})),
              patch.object(native, "cache", return_value=features), patch.object(native, "parity", return_value={}),
              patch.object(native, "write_meta", side_effect=lambda path, meta: torch.save(meta.head, Path(path) / "head.pt")),
              watch_training(self.out, "train")):
            native.train(args, {"v2_head_sha256": digest(checkpoint / "head.pt"), "splits": {}},
                         {"train": rows, "dev": rows})
        events = self.events()
        self.assertEqual([row["status"] for row in events], ["running", "running", "completed"])
        for epoch, row in enumerate(events[:2], start=1):
            order = list(range(len(rows)))
            random.Random(96729100 + epoch).shuffle(order)
            self.assertEqual(row["epoch"], epoch)
            self.assert_batches([row], rows, order, (slice(None),), [1])
        self.assert_initial_head(events[0], rows, features)

    def test_round6_reverse_views_preserve_direction_mapping(self):
        self.out.mkdir()
        rows, normal = fixture(8)
        reverse_rows = [dict(row, valid_moves=row["valid_moves"][::-1]) for row in rows]
        reverse = [(q, k.flip(0)) for q, k in normal]
        with patch.object(round6_fit, "EPOCHS", 1), watch_training(self.out, "round6-order"):
            round6_fit.fit(Pointer(), rows, reverse_rows, normal, reverse, [0, 1] * 4,
                           rows, normal, self.out / "fit", "test")
        event, = self.events()
        self.assertEqual(event["trial"], "alternating")
        self.assert_initial_head(event, rows, normal)

    def test_batch_sampling_keeps_the_correct_first_64_predictions(self):
        self.out.mkdir()
        rows, features = fixture(73)
        config = head_fit.FitConfig(max_epochs=1, batch_size=72)
        with watch_training(self.out, "round9-kl"), watch_training(self.out, "standalone-head"):
            head_fit.fit(Pointer(), rows, features, rows, features, config)
        first, last = self.events()
        order = list(range(len(rows)))
        random.Random(config.seed + 1).shuffle(order)
        self.assertEqual([s["row"] for s in first["samples"]], [i + 1 for i in order[:64]])
        self.assertEqual((first["batch_size"], first["samples_seen"], last["samples_seen"]), (72, 72, 73))
        self.assert_initial_head(first, rows, features)

    def test_throughput_uses_actual_sample_delta_and_resets_on_new_trial_or_epoch(self):
        self.out.mkdir()
        rows, features = fixture(1)
        optimizer = torch.optim.SGD(Pointer().parameters(), lr=.01)
        journal = _Journal(self.out, "experiment")
        scores = [Pointer()(*features[0])]
        def publish(trial, epoch, step, seen, now):
            with patch("kev2048_release.training_progress.time.perf_counter", return_value=now):
                journal._publish_batch(optimizer, rows, [0], scores, trial=trial, epoch=epoch, step=step,
                                       steps=12, loss=1., seen=seen)
        publish("a", 1, 1, 16, 100)
        publish("a", 1, 10, 160, 102)
        publish("a", 2, 1, 16, 105)
        publish("b", 2, 10, 160, 110)
        publish("b", 2, 1, 16, 111)
        publish("b", 2, 10, 160, 111)
        self.assertEqual([row.get("boards_per_second") for row in self.events()], [None, 72., None, None, None, None])

    def test_unrelated_optimizer_does_not_write_to_output(self):
        self.out.mkdir()
        with watch_training(self.out, "train"):
            torch.optim.SGD(Pointer().parameters(), lr=.1).step()
        self.assertFalse((self.out / "progress.jsonl").exists())


if __name__ == "__main__":
    unittest.main()

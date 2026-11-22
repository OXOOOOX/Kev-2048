from copy import deepcopy
from importlib.util import find_spec
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from kev2048_release.assets import FILES
from kev2048_release.feature_cache import cache_key, frozen_inputs, read_features, write_features


class FeatureCacheKeyTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        self.base = root / "base"
        self.checkpoint = root / "checkpoint"
        for folder, names in (
            (self.base, ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json")),
            (self.checkpoint, (*FILES, "head.pt")),
        ):
            folder.mkdir()
            for name in names:
                (folder / name).write_bytes(b"old")
        self.lock = {"splits": {"train": {"sha256": "train"}, "dev": {"sha256": "dev"}}}
        self.data = {"train": [{"board": [[2, 0]], "valid_moves": ["up", "left"]}], "dev": []}
        self.encoder = {"prompt": "stock", "device": "cuda", "saved_weights_dtype": "fp32"}

    def key(self, checkpoint=None, data=None, encoder=None):
        inputs = frozen_inputs(self.base, checkpoint or self.checkpoint)
        return cache_key(inputs, self.lock, data or self.data, encoder or self.encoder)

    def test_new_pointer_head_reuses_features(self):
        original = self.key()
        new_checkpoint = self.checkpoint.with_name("new-checkpoint")
        shutil.copytree(self.checkpoint, new_checkpoint)
        (new_checkpoint / "head.pt").write_bytes(b"new head")
        self.assertEqual(original, self.key(checkpoint=new_checkpoint))

    def test_invalidates_for_frozen_files_records_and_encoder(self):
        original = self.key()
        for part in (
            self.base / "model.safetensors", self.base / "config.json", self.base / "tokenizer.json",
            self.checkpoint / "adapter_model.safetensors", self.checkpoint / "adapter_config.json",
        ):
            with self.subTest(part=part.name):
                part.write_bytes(b"new")
                self.assertNotEqual(original, self.key())
                part.write_bytes(b"old")
        moved = deepcopy(self.data)
        moved["train"][0]["valid_moves"].reverse()
        self.assertNotEqual(original, self.key(data=moved))
        for change in ({"device": "cpu"}, {"prompt": "different"}, {"saved_weights_dtype": "bf16"}):
            with self.subTest(change=change):
                self.assertNotEqual(original, self.key(encoder={**self.encoder, **change}))

    def test_tracks_weights_in_linked_directory(self):
        weights = self.base.with_name("linked-weights")
        weights.mkdir()
        weight = weights / "model.safetensors"
        weight.write_bytes(b"old")
        (self.base / "config.json").write_text('{"transformers_weights":"weights/model.safetensors"}', encoding="utf-8")
        try:
            (self.base / "weights").symlink_to(weights, target_is_directory=True)
        except OSError as exc:
            self.skipTest(f"directory symlinks unavailable: {exc}")
        original = self.key()
        weight.write_bytes(b"new")
        self.assertNotEqual(original, self.key())


@unittest.skipUnless(find_spec("torch") and find_spec("safetensors"), "optional native dependencies required")
class FeatureCacheStorageTest(unittest.TestCase):
    def test_round_trip_rejects_wrong_key_or_corruption(self):
        import torch

        counts = [2, 3]
        hidden = 4
        features = [(torch.arange(4, dtype=torch.float32) * count,
                     torch.arange(count * 4, dtype=torch.float32).reshape(count, 4)) for count in counts]
        key = "a" * 64
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "training.features"
            self.assertEqual(read_features(target, key, counts, hidden, "cpu"), (None, "missing"))
            write_features(target, key, features, counts, hidden)
            loaded, error = read_features(target, key, counts, hidden, "cpu")
            self.assertIsNone(error)
            for got, expected in zip(loaded, features, strict=True):
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(got, expected, strict=True)))
            self.assertIsNone(read_features(target, "b" * 64, counts, hidden, "cpu")[0])
            self.assertIsNone(read_features(target, key, [2, 2], hidden, "cpu")[0])
            # Change one data byte without touching the tensor header.
            data = target.read_bytes()
            target.write_bytes(data[:-1] + bytes([data[-1] ^ 1]))
            self.assertIsNone(read_features(target, key, counts, hidden, "cpu")[0])

            replacement = Path.replace
            sharing_violation = PermissionError("temporary Windows sharing violation")
            sharing_violation.winerror = 32

            def replace_after_conflict(source, destination):
                if mocked_replace.call_count == 1:
                    raise sharing_violation
                return replacement(source, destination)

            with patch.object(Path, "replace", autospec=True, side_effect=replace_after_conflict) as mocked_replace:
                write_features(target, key, features, counts, hidden)
            self.assertIsNone(read_features(target, key, counts, hidden, "cpu")[1])


if __name__ == "__main__":
    unittest.main()

"""Exercise telemetry routing without loading a model or starting an experiment."""

from contextlib import contextmanager, redirect_stdout
from importlib.util import find_spec
from io import StringIO
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
import unittest
from unittest.mock import patch

import kev2048_release
from kev2048_release import cli, training_progress


class TrainingEntryTests(unittest.TestCase):
    def test_cli_enters_each_training_context_around_its_runner(self):
        config, out, b18 = Path("config"), Path("output"), Path("b18")
        for command, module, options, arguments in (
            ("correction", "correction", [], (config, out, True)),
            ("round6-order", "round6_order", ["--b18-run", str(b18)], (config, out, b18, True)),
            ("round7-scale", "round7_scale", [], (config, out, True)),
        ):
            with self.subTest(command=command):
                calls = []

                @contextmanager
                def watch(path, name):
                    calls.append(("begin", path, name))
                    yield
                    calls.append(("end", path, name))

                def run(*args):
                    calls.append(("run", args))
                    return {}

                runner = ModuleType("kev2048_release." + module)
                runner.run = run
                argv = ["kev2048", command, "--config", str(config), "--out", str(out), "--resume", *options]
                with (patch.dict(sys.modules, {runner.__name__: runner}), patch.object(sys, "argv", argv),
                      patch.object(training_progress, "watch_training", watch), redirect_stdout(StringIO())):
                    cli.main()
                self.assertEqual(calls, [("begin", out, command), ("run", arguments), ("end", out, command)])

    def test_real_python_module_startup_and_help(self):
        # Run the actual package bootstrap under CPython -m. Stand-ins keep
        # these entry tests independent of torch and physical training data.
        original = Path(kev2048_release.__file__).parent
        with TemporaryDirectory() as tmp:
            package = Path(tmp, "kev2048_release")
            package.mkdir()
            for name in ("__init__.py", "training_entry.py"):
                shutil.copyfile(original / name, package / name)
            (package / "training_progress.py").write_text(
                "from contextlib import contextmanager\n"
                "import json\n"
                "print(json.dumps(['observer imported']))\n"
                "@contextmanager\n"
                "def watch_training(out, command, *, started):\n"
                "    print(json.dumps(['begin', str(out), command]))\n"
                "    yield\n"
                "    print(json.dumps(['end', str(out), command]))\n", encoding="utf-8")
            (package / "head_fit.py").write_text(
                "import json\ndef fit(): print(json.dumps(['fit']))\n"
                "def evaluate(): print(json.dumps(['evaluate']))\n", encoding="utf-8")
            driver = (
                "from importlib.util import find_spec\n"
                "assert find_spec('kev2048_release.head_fit') is not None\n"
                "from .head_fit import fit, evaluate\n"
                "if __name__ == '__main__':\n"
                "    import argparse, json\n"
                "    p = argparse.ArgumentParser()\n"
                "    p.add_argument('--out', required=True)\n"
                "    p.add_argument('--evaluate', action='store_true')\n"
                "    args = p.parse_args()\n"
                "    print(json.dumps(['run', args.out]))\n"
                "    (evaluate if args.evaluate else fit)()\n"
            )
            drivers = ("round8_retest", "round9_kl", "round10_teacher", "round12_onpolicy", "future_driver")
            for name in (*drivers, "cli"):
                (package / (name + ".py")).write_text(driver, encoding="utf-8")

            def invoke(*args):
                return subprocess.run([sys.executable, "-B", *args], cwd=tmp, capture_output=True,
                                      text=True, check=True, timeout=20).stdout

            for name in drivers:
                with self.subTest(name=name):
                    module = "kev2048_release." + name
                    for arguments, output in ((["--out", "a folder"], "a folder"), (["--out=joined"], "joined"),
                                              (["--out", "first", "--o=last"], "last")):
                        with self.subTest(arguments=arguments):
                            stdout = invoke("-m", module, *arguments)
                            journal_dir = str(Path(tmp, output))
                            self.assertEqual([json.loads(row) for row in stdout.splitlines()], [
                                ["run", output], ["observer imported"], ["begin", journal_dir, "standalone-head"],
                                ["fit"], ["end", journal_dir, "standalone-head"]])
                    help_text = invoke("-m", module, "--out", "first", "--help")
                    self.assertIn("usage:", help_text)
                    self.assertNotIn("observer imported", help_text)
            stdout = invoke("-mkev2048_release.future_driver", "--out", "ordinary", "--evaluate")
            self.assertEqual([json.loads(row) for row in stdout.splitlines()], [["run", "ordinary"], ["evaluate"]])
            stdout = invoke("-m", "kev2048_release.cli", "--out", "ordinary")
            self.assertEqual([json.loads(row) for row in stdout.splitlines()], [["run", "ordinary"], ["fit"]])
            invoke("-c", "import kev2048_release.round9_kl, sys; assert 'kev2048_release.training_progress' not in sys.modules")

    @unittest.skipUnless(find_spec("torch"), "Requires torch for a miniature CPU fit")
    def test_new_standalone_driver_records_all_actual_optimizer_steps(self):
        original = Path(kev2048_release.__file__).parent
        with TemporaryDirectory() as tmp:
            package = Path(tmp, "kev2048_release")
            package.mkdir()
            # Extend the test driver's package with the real shared training implementation.
            (package / "__init__.py").write_text(f"__path__.append({str(original)!r})\n"
                + (original / "__init__.py").read_text(encoding="utf-8"), encoding="utf-8")
            (package / "new_driver.py").write_text(
                "import sys\nimport torch\n"
                f"sys.path.insert(0, {str(Path(__file__).parent.absolute())!r})\n"
                "from test_training_progress import Pointer, fixture\n"
                "from .head_fit import fit, FitConfig\n"
                "def train(resolved):\n"
                "    rows, features = fixture(19)\n"
                "    fit(Pointer(), rows, features, rows, features, FitConfig(max_epochs=1, batch_size=2))\n"
                "torch.set_num_threads(1)\n"
                "train({'arm': 'new-candidate'})\n", encoding="utf-8")
            out = Path(tmp, "result")
            out.mkdir()
            subprocess.run([sys.executable, "-B", "-m", "kev2048_release.new_driver", "--out", str(out)],
                           cwd=tmp, check=True, timeout=60, env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"})
            events = [json.loads(line) for line in (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual([row["optimizer_step"] for row in events], list(range(1, 11)))
            self.assertEqual({row["trial"] for row in events}, {"new-candidate"})
            self.assertTrue(all(sample["candidate"]["choice"] in sample["candidate"]["probabilities"]
                                for row in events for sample in row["samples"]))


if __name__ == "__main__":
    unittest.main()

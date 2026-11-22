"""No model load: watcher tests use deterministic, local fake worker processes."""
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from watch_train import run_worker


class TrainingWatchTests(unittest.TestCase):
    def run_fake(self, text):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            out, events = root/"run", root/"safe.jsonl"
            rc = run_worker([sys.executable, "-u", "-c", text, str(out)], out, events, heartbeat=.03)
            rows = [json.loads(r) for r in events.read_text().splitlines()]
            return rc, rows

    def test_stdout_markers_liveness_and_artifact(self):
        rc,rows = self.run_fake("""import pathlib,sys,time
time.sleep(.09)
print('SECRET raw output /some/path')
print('train: cached 100/88')
print('dev: cached 1/2')
print('train: cached 1/2')
time.sleep(.12)
print('train: cached 2/2')
print('train: cached 2/2')
print('dev: cached 2/2')
print('train: cached 3/3')
p=pathlib.Path(sys.argv[1]);p.mkdir();(p/'result.json').write_text('{}')
""")
        self.assertEqual(rc,0)
        self.assertEqual(rows[0]["status"],"loading")
        self.assertTrue(any(r["phase"]=="starting" and r["elapsed_seconds"]>=.03 for r in rows))
        self.assertEqual(rows[-1]["status"],"completed")
        self.assertEqual({r["phase"] for r in rows},{"starting","train_cache","dev_cache","result_saved"})
        steps=[(r["phase"],r["step"],r["total"]) for r in rows if "step" in r]
        self.assertGreaterEqual(steps.count(("train_cache",1,2)),2)
        self.assertEqual(steps.count(("train_cache",2,2)),1)
        self.assertEqual(steps.count(("dev_cache",2,2)),1)
        self.assertNotIn("SECRET", json.dumps(rows))
        self.assertTrue(all(r["timestamp"].endswith("Z") for r in rows))
        self.assertEqual([r["elapsed_seconds"] for r in rows],sorted(r["elapsed_seconds"] for r in rows))

    def test_failure_not_claimed_as_complete(self):
        rc,rows=self.run_fake("raise SystemExit(7)")
        self.assertEqual(rc,7)
        self.assertEqual(rows[-1]["status"],"error")
        self.assertEqual(rows[-1]["phase"],"worker_exit_error")
        rc,rows=self.run_fake("print('done')")
        self.assertEqual(rc,1)
        self.assertEqual(rows[-1]["phase"],"missing_result")

    def test_collisions_do_not_launch_or_overwrite(self):
        exe=[sys.executable,"-c","raise AssertionError('must never start')"]
        with TemporaryDirectory() as tmp:
            root=Path(tmp);out=root/"new";events=root/"e.jsonl"
            with self.assertRaises(ValueError):run_worker(exe,out,out/"nested.jsonl")
            events.write_text("prior")
            with self.assertRaises(FileExistsError):run_worker(exe,out,events)
            self.assertEqual(events.read_text(),"prior")
            out.mkdir()
            with self.assertRaises(FileExistsError):run_worker(exe,out,root/"fresh.jsonl")
            self.assertFalse((root/"fresh.jsonl").exists())

if __name__=="__main__":
    unittest.main()

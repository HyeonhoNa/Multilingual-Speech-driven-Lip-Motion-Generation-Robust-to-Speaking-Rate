# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Terminal output must distinguish stored, fresh, partial, and failed results."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from tta_pipeline import evaluation_worker, launcher
from tta_pipeline.io import read_json, sha256, write_json
from tta_pipeline.score_output import EVENT_PREFIX, ScorePrinter


def score(clip="c1", identity="i1"):
    return dict(clip=clip, identity=identity, lse_d=6.0, lse_c=8.0,
                language="en_us", rate_bin="slow", video="video.mkv",
                status="ok", strict_eligible=True)


class OutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def saved_run(self, rows, **extra):
        write_json(self.root / "run.json", dict(clips=["c1", "c2"],
                   identities=["i1"], full_fixed_cohort=False))
        write_json(self.root / "evaluation/000/results.json", dict(scores=rows, **extra))

    def test_no_average_keeps_rows_and_disk_aggregates(self):
        self.saved_run([score(), score("c2")])
        output = io.StringIO()
        with redirect_stdout(output):
            launcher.report(self.root, no_average=True)
        self.assertEqual(output.getvalue().count("[cached: saved run]"), 2)
        self.assertIn("D=6.000000000", output.getvalue())
        self.assertNotIn("mean", output.getvalue().lower())
        self.assertNotIn('"lse_d"', output.getvalue())
        self.assertEqual(read_json(self.root / "summary.json")["lse_d"], 6.0)
        self.assertTrue((self.root / "language.csv").is_file())

    def test_report_cli_no_average(self):
        self.saved_run([score(), score("c2")])
        output = io.StringIO()
        argv = ["run.py", "report", "--output", str(self.root), "--no-average"]
        with patch.object(sys, "argv", argv), redirect_stdout(output):
            launcher.main()
        self.assertEqual(output.getvalue().count("[cached: saved run]"), 2)
        self.assertNotIn("mean", output.getvalue().lower())

    def test_rescore_refuses_source_output_overlap_before_writing(self):
        self.saved_run([score()])
        before = {str(path.relative_to(self.root)): path.read_bytes()
                  for path in self.root.rglob("*") if path.is_file()}
        argv = ["run.py", "rescore", "--source", str(self.root), "--output", str(self.root)]
        with patch.object(sys, "argv", argv), patch("sys.stderr", io.StringIO()):
            with self.assertRaises(SystemExit):
                launcher.main()
        after = {str(path.relative_to(self.root)): path.read_bytes()
                 for path in self.root.rglob("*") if path.is_file()}
        self.assertEqual(before, after)

    def test_failed_cell_and_partial_mean_remain_visible(self):
        self.saved_run([score()], state="failed", error="face detection failed",
                       failed_cell=dict(clip="c2", identity="i1", error="face detection failed"))
        output = io.StringIO()
        with redirect_stdout(output):
            result = launcher.report(self.root)
        self.assertEqual(result["state"], "partial")
        self.assertIn("[failed] c2 / i1: face detection failed", output.getvalue())
        self.assertIn("Partial mean", output.getvalue())
        self.assertNotIn("Final mean", output.getvalue())
        self.assertIn("Missing cells: 1", output.getvalue())

    def test_final_report_does_not_repeat_streamed_rows(self):
        self.saved_run([score(), score("c2")])
        printer = ScorePrinter()
        output = io.StringIO()
        with redirect_stdout(output):
            printer.score(score())
            launcher.report(self.root, printer=printer)
        self.assertEqual(output.getvalue().count("c1 / i1"), 1)
        self.assertEqual(output.getvalue().count("c2 / i1"), 1)
        self.assertIn("Final mean", output.getvalue())

    def test_worker_failure_prevents_final_mean_even_with_saved_complete_scores(self):
        self.saved_run([score(), score("c2")])
        output = io.StringIO()
        with redirect_stdout(output):
            result = launcher.report(self.root, worker_error="worker exited before validation")
        self.assertEqual(result["state"], "partial")
        self.assertNotIn("Final mean", output.getvalue())
        self.assertIn("worker exited before validation", output.getvalue())

    def test_saved_launcher_failure_stays_visible_in_later_reports(self):
        self.saved_run([score(), score("c2")])
        write_json(self.root / "status.json", dict(state="failed", error="worker failed"))
        output = io.StringIO()
        with redirect_stdout(output):
            result = launcher.report(self.root)
        self.assertEqual(result["state"], "partial")
        self.assertIn("worker failed", output.getvalue())
        self.assertNotIn("Final mean", output.getvalue())

    def test_two_workers_stream_before_first_worker_finishes_and_skip_old_log(self):
        release = self.root / "release"
        old_log = self.root / "logs/evaluate_001.log"
        old_log.parent.mkdir()
        old_log.write_text(EVENT_PREFIX + json.dumps(dict(event="old")) + "\n")
        slow = ("import json,time; from pathlib import Path; "
                "p=Path(%r); deadline=time.monotonic()+3\n"
                "while not p.exists() and time.monotonic()<deadline: time.sleep(.02)\n"
                "print(%r+json.dumps({'event':'slow','released':p.exists()}),flush=True)" %
                (str(release), EVENT_PREFIX))
        fast = "import json; print(%r+json.dumps({'event':'fast'}),flush=True)" % EVENT_PREFIX
        events = []
        def receive(event):
            events.append(event)
            if event["event"] == "fast":
                release.touch()
        launcher.run_workers([(0, [sys.executable, "-u", "-c", slow]),
                              (1, [sys.executable, "-u", "-c", fast])],
                             self.root, "evaluate", on_event=receive)
        self.assertEqual([e["event"] for e in events], ["fast", "slow"])
        self.assertTrue(events[-1]["released"])
        self.assertIn('"event": "old"', old_log.read_text())
        self.assertIn('"event": "fast"', old_log.read_text())

    def test_failed_worker_still_drains_other_worker_score(self):
        events = []
        failing = "import sys; sys.exit(7)"
        completing = "import time; time.sleep(.1); print(%r,flush=True)" % (EVENT_PREFIX + '{"event":"done"}')
        with self.assertRaisesRegex(RuntimeError, "exit 7"):
            launcher.run_workers([(0, [sys.executable, "-c", failing]),
                                  (1, [sys.executable, "-u", "-c", completing])],
                                 self.root, "evaluate", on_event=events.append)
        self.assertEqual(events, [dict(event="done")])

    def test_failed_launch_closes_logs_and_stops_previous_worker(self):
        real_popen = subprocess.Popen
        processes = []
        def launch(command, **kwargs):
            if processes:
                raise OSError("simulated launch failure")
            child = real_popen(command, **kwargs)
            processes.append(child)
            return child
        with patch.object(launcher.subprocess, "Popen", side_effect=launch):
            with self.assertRaisesRegex(OSError, "launch failure"):
                launcher.run_workers([(0, [sys.executable, "-c", "import time; time.sleep(30)"]),
                                      (1, [sys.executable, "-c", "pass"])], self.root, "evaluate")
        self.assertIsNotNone(processes[0].poll())

    def test_interruption_stops_process_and_closes_descriptors(self):
        real_popen = subprocess.Popen
        processes, log_streams = [], []
        def launch(command, **kwargs):
            child = real_popen(command, **kwargs)
            processes.append(child)
            log_streams.append(kwargs["stdout"])
            return child
        def interrupted(event):
            raise KeyboardInterrupt()
        script = ("import time; print(%r,flush=True); time.sleep(30)" %
                  (EVENT_PREFIX + '{"event":"started"}'))
        descriptors = len(list(Path("/proc/self/fd").iterdir()))
        with patch.object(launcher.subprocess, "Popen", side_effect=launch):
            with self.assertRaises(KeyboardInterrupt):
                launcher.run_workers([(0, [sys.executable, "-u", "-c", script])],
                                     self.root, "evaluate", on_event=interrupted)
        self.assertIsNotNone(processes[0].poll())
        self.assertTrue(all(stream.closed for stream in log_streams))
        self.assertEqual(len(list(Path("/proc/self/fd").iterdir())), descriptors)


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.entries = [dict(clip="%s_%s_%s" % (language, rate, n), language="lang%d" % language, rate_bin=rate)
                        for language in range(15) for rate in ("slow", "medium", "fast") for n in range(2)]
        self.ids = ["identity%d" % i for i in range(10)]
        write_json(self.root / "configs/evaluation.json", dict(entries=self.entries, identities=self.ids))
        self.rows = [dict(entry, identity=identity, D=6, C=8)
                     for entry in self.entries for identity in self.ids]

    def tearDown(self):
        self.tmp.cleanup()

    def save(self):
        path = self.root / "reference_results/cell.csv"
        path.parent.mkdir(exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def render(self, no_average=False):
        self.save()
        output = io.StringIO()
        with patch.object(launcher, "ROOT", self.root), redirect_stdout(output):
            result = launcher.reference(no_average=no_average)
        return result, output.getvalue()

    def test_complete_reference_prints_900_cached_scores_and_mean(self):
        result, output = self.render()
        self.assertEqual(output.count("[cached: stored reference]"), 900)
        self.assertIn("Final mean", output)
        self.assertIn("no scoring performed", output)
        self.assertEqual(result["lse_c"], 8)

    def test_reference_no_average(self):
        _, output = self.render(no_average=True)
        self.assertEqual(output.count("[cached: stored reference]"), 900)
        self.assertNotIn("mean", output.lower())

    def test_incomplete_reference_rejected_before_output(self):
        self.rows.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete reference"):
            self.render()

    def test_nonfinite_duplicate_foreign_and_wrong_metadata_rejected(self):
        for change in (dict(D="nan"), dict(identity="foreign"), dict(language="wrong"), dict(rate_bin="wrong")):
            with self.subTest(change=change):
                original = dict(self.rows[0])
                self.rows[0].update(change)
                with self.assertRaises(ValueError):
                    self.render()
                self.rows[0] = original
        self.rows[-1] = dict(self.rows[0])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.render()

    def test_reference_status_is_checked_when_present(self):
        for row in self.rows:
            row["status"] = "ok"
        self.rows[-1]["status"] = "failed"
        with self.assertRaisesRegex(ValueError, "Failed"):
            self.render()

    def test_bad_manifest_cohort_rejected(self):
        write_json(self.root / "configs/evaluation.json", dict(entries=self.entries[:-1], identities=self.ids))
        with self.assertRaisesRegex(ValueError, "90 unique"):
            self.render()


class WorkerTests(unittest.TestCase):
    def test_worker_emits_cached_then_new_score_and_records_failure(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            video = root / "video.mkv"
            video.write_bytes(b"mock media")
            digest = sha256(video)
            entries = [dict(clip=c, language="en_us", rate_bin="slow") for c in ("c1", "c2")]
            write_json(root / "configs/evaluation.json", dict(entries=entries))
            assignment = dict(run_root=str(root), entries=[dict(clip=c, identity="i1", video="video.mkv", video_sha256=digest)
                                                          for c in ("c1", "c2")])
            write_json(root / "assignment.json", assignment)
            cached = dict(score(), video_sha256=digest)
            output = root / "scores"
            write_json(output / "results.json", dict(assignment=assignment, scores=[cached], state="partial"))
            args = types.SimpleNamespace(repo=root, output_dir=output, assignment=root / "assignment.json", gpu=0)
            fake = types.SimpleNamespace(evaluate_video=lambda *a, **kw: score("c2"))
            terminal = io.StringIO()
            with patch.dict(sys.modules, {"tta_pipeline.evaluation": fake}), redirect_stdout(terminal):
                evaluation_worker.run(args)
            events = [json.loads(line[len(EVENT_PREFIX):]) for line in terminal.getvalue().splitlines()]
            self.assertEqual([e["cached"] for e in events], [True, False])
            self.assertEqual(read_json(output / "results.json")["state"], "complete")
            video.write_bytes(b"changed")
            terminal = io.StringIO()
            with patch.dict(sys.modules, {"tta_pipeline.evaluation": fake}), redirect_stdout(terminal):
                with self.assertRaisesRegex(ValueError, "Generated video changed"):
                    evaluation_worker.run(args)
            failed = read_json(output / "results.json")
            self.assertEqual(failed["failed_cell"]["clip"], "c1")
            self.assertIn('"event": "failure"', terminal.getvalue())


if __name__ == "__main__":
    unittest.main()

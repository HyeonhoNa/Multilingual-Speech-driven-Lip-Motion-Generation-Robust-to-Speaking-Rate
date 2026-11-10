"""Guard against silently reporting incomplete or duplicated evaluation as final."""
import csv
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stdout
import io
from unittest.mock import patch

from tta_pipeline import launcher
from tta_pipeline.io import write_json


class AggregateTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.output = Path(self.tmp.name)
        write_json(self.output / "run.json", dict(clips=["c1","c2"],identities=["i1","i2"],full_fixed_cohort=False))

    def tearDown(self):
        self.tmp.cleanup()

    def row(self, clip="c1", identity="i1", d=6.0, c=8.0):
        return dict(clip=clip,identity=identity,language="en_us",rate_bin="slow",lse_d=d,lse_c=c,video="example.mkv",status="ok",strict_eligible=True)

    def evaluate(self, rows):
        write_json(self.output / "evaluation/000/results.json", dict(scores=rows))
        with redirect_stdout(io.StringIO()):
            return launcher.report(self.output)

    def test_partial_is_not_full(self):
        result = self.evaluate([self.row()])
        self.assertEqual(result["state"], "partial")
        self.assertEqual(result["expected_cells"],4)
        self.assertFalse(result["full_fixed_cohort"])
        self.assertIsNone(result["mean_passes"])

    def test_complete_averages_all_four_cells(self):
        result = self.evaluate([self.row("c1","i1",5,9), self.row("c1","i2",6,8),
                                self.row("c2","i1",7,7), self.row("c2","i2",6,8)])
        self.assertEqual(result["state"],"complete")
        self.assertEqual((result["lse_d"],result["lse_c"],result["pass_both"]),(6,8,3))

    def test_duplicate_scores_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate([self.row(),self.row()])

    def test_foreign_identity_rejected(self):
        with self.assertRaises(ValueError):
            self.evaluate([self.row(identity="unknown")])

    def test_missing_or_invalid_result_not_counted(self):
        row = self.row()
        row["error"] = "face detection failed"
        with self.assertRaises(ValueError):
            self.evaluate([row])

    def test_ineligible_official_score_rejected(self):
        row = self.row()
        row["strict_eligible"] = False
        with self.assertRaises(ValueError):
            self.evaluate([row])

    def test_zero_exit_worker_cannot_hide_missing_cells(self):
        write_json(self.output / "evaluation/000/results.json",dict(scores=[self.row()]))
        plan = json.loads((self.output / "run.json").read_text())
        plan["workers"] = [dict(index=0,gpu=0,clips=["c1","c2"])]
        generated = dict(state="complete",entries=[dict(clip=c,identity=i) for c in plan["clips"] for i in plan["identities"]])
        with patch.object(launcher,"generation_manifest",return_value=generated), patch.object(launcher,"run_workers"), patch.object(launcher,"python_for",return_value="python"), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError,"complete requested cohort"):
                launcher.evaluate(plan,self.output)

    @unittest.skipUnless((launcher.ROOT / "reference_results/cell.csv").exists(), "Local reference data is intentionally excluded from Git")
    def test_full_reference_recomputes(self):
        with (launcher.ROOT / "reference_results/cell.csv").open(encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows),900)
        self.assertEqual(len({(r["clip"],r["identity"]) for r in rows}),900)
        self.assertAlmostEqual(sum(float(r["D"]) for r in rows)/900,5.894898862111111, places=10)
        self.assertAlmostEqual(sum(float(r["C"]) for r in rows)/900,8.230908223, places=10)


if __name__ == "__main__":
    unittest.main()

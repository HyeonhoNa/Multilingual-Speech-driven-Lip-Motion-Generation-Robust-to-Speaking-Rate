# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Rescoring preserves old generated inputs and rejects altered provenance."""
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tta_pipeline.generation import CHECKPOINT_SHA256
from tta_pipeline.io import read_json, sha256, write_json
from tta_pipeline.rescore import plan_for_rescore, source_generation


class RescoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.repo, self.source, self.output = (self.root / name for name in ("repo", "source", "evaluation"))
        self.pcm = "b" * 64
        entries = [dict(clip=clip, frames=12, pcm_sha256=self.pcm, language="en_us", rate_bin="slow")
                   for clip in ("c1", "c2", "c3")]
        write_json(self.repo / "configs/evaluation.json", dict(entries=entries, identities=["i1", "i2", "i3"]))
        write_json(self.repo / "configs/models.json", dict(checkpoint_step=9000,
                   files=dict(artalk_step9000=dict(sha256=CHECKPOINT_SHA256))))
        write_json(self.repo / "configs/identities.json", dict(entries=[dict(identity="i1"), dict(identity="i2"), dict(identity="i3")]))
        write_json(self.repo / "configs/code_checksums.json", dict(files=[]))
        hashes = {path.name: sha256(path) for path in (self.repo / "configs").glob("*.json")}
        hashes["code_checksums.json"] = "a" * 64
        self.run = dict(schema=1, checkpoint_step=9000, clips=["c1", "c2"], identities=["i1", "i2"],
                        expected_cells=4, full_fixed_cohort=False, config_sha256=hashes,
                        gpus=[7], workers=[dict(index=0, gpu=7, clips=["c1", "c2"])])
        write_json(self.source / "run.json", self.run)
        rows = []
        for clip in self.run["clips"]:
            for identity in self.run["identities"]:
                video = self.source / "workers/000/videos" / identity / (clip + ".mkv")
                video.parent.mkdir(parents=True, exist_ok=True)
                video.write_bytes((clip + "/" + identity).encode())
                rows.append(dict(clip=clip, identity=identity, frames=12, pcm_sha256=self.pcm,
                                 checkpoint_sha256=CHECKPOINT_SHA256, video=str(video.relative_to(self.source)),
                                 video_sha256=sha256(video), language="en_us"))
        self.generation = dict(state="complete", entries=rows, expected_cells=4, completed_cells=4)
        write_json(self.source / "generation.json", self.generation)
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {}, clear=True).start()

    def args(self, **changes):
        values = dict(source=self.source, clips=None, identities=None, gpus=None, limit=None)
        values.update(changes)
        return SimpleNamespace(**values)

    def plan(self, **changes):
        return plan_for_rescore(self.args(**changes), self.output, self.repo)

    def snapshot(self):
        return {str(path.relative_to(self.source)): path.read_bytes()
                for path in self.source.rglob("*") if path.is_file()}

    def test_old_generation_hashes_are_preserved_without_source_writes_or_copying(self):
        before = self.snapshot()
        plan = self.plan()
        self.assertEqual(plan["gpus"], [0])
        self.assertEqual(plan["source_run_root"], str(self.source))
        self.assertEqual(plan["source_config_sha256"], self.run["config_sha256"])
        self.assertEqual(plan["source_run_sha256"], sha256(self.source / "run.json"))
        self.assertEqual(plan["source_generation_sha256"], sha256(self.source / "generation.json"))
        self.assertNotEqual(plan["config_sha256"]["code_checksums.json"], plan["source_config_sha256"]["code_checksums.json"])
        self.assertFalse(plan["full_fixed_cohort"])
        self.assertEqual(plan["expected_cells"], 4)
        generated = source_generation(plan, self.output, self.repo)
        self.assertEqual(generated, self.generation)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(sorted(path.name for path in self.output.iterdir()), ["run.json"])

    def test_unchanged_resume_reuses_plan_and_does_not_rewrite_it(self):
        plan = self.plan(clips="c2", identities="i2", gpus="2,5")
        path = self.output / "run.json"
        stat = path.stat()
        self.assertEqual(self.plan(), plan)
        self.assertEqual(path.stat().st_mtime_ns, stat.st_mtime_ns)
        generated = source_generation(plan, self.output, self.repo)
        self.assertEqual([(row["clip"], row["identity"]) for row in generated["entries"]], [("c2", "i2")])
        self.assertEqual(generated["completed_cells"], 1)

    def test_source_scope_and_order_determine_selection_and_worker_shards(self):
        plan = self.plan(clips="c2,c1", identities="i2,i1", gpus="4,1")
        self.assertEqual(plan["clips"], ["c1", "c2"])
        self.assertEqual(plan["identities"], ["i1", "i2"])
        self.assertEqual(plan["workers"], [dict(index=0, gpu=4, clips=["c1"]), dict(index=1, gpu=1, clips=["c2"])])
        self.assertNotIn("CUDA_VISIBLE_DEVICES", os.environ)

    def test_full_fixed_cohort_requires_all_900_source_cells(self):
        clips = ["clip%02d" % index for index in range(90)]
        identities = ["identity%02d" % index for index in range(10)]
        entries = [dict(clip=clip, frames=12, pcm_sha256=self.pcm, language="en_us", rate_bin="slow")
                   for clip in clips]
        write_json(self.repo / "configs/evaluation.json", dict(entries=entries, identities=identities))
        write_json(self.repo / "configs/identities.json", dict(entries=[dict(identity=value) for value in identities]))
        run = copy.deepcopy(self.run)
        run.update(clips=clips, identities=identities, expected_cells=900, full_fixed_cohort=True)
        for name in ("evaluation.json", "identities.json"):
            run["config_sha256"][name] = sha256(self.repo / "configs" / name)
        write_json(self.source / "run.json", run)
        rows = []
        for clip in clips:
            for identity in identities:
                video = self.source / (clip + "_" + identity + ".mkv")
                video.write_bytes((clip + identity).encode())
                rows.append(dict(clip=clip, identity=identity, frames=12, pcm_sha256=self.pcm,
                                 checkpoint_sha256=CHECKPOINT_SHA256, video=video.name, video_sha256=sha256(video)))
        write_json(self.source / "generation.json", dict(state="complete", entries=rows, expected_cells=900, completed_cells=900))
        full = self.plan()
        self.assertTrue(full["full_fixed_cohort"])
        self.assertEqual(full["expected_cells"], 900)
        subset = plan_for_rescore(self.args(clips=clips[0]), self.root / "subset", self.repo)
        self.assertFalse(subset["full_fixed_cohort"])
        self.assertEqual(subset["expected_cells"], 10)

    def test_selection_cannot_expand_beyond_source(self):
        for values in (dict(clips="c3"), dict(identities="i3"), dict(clips="c1,c1"),
                       dict(identities="i1,i1"), dict(clips=""), dict(limit=1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                self.plan(**values)
        self.assertFalse((self.output / "run.json").exists())

    def test_resume_rejects_new_scope_or_gpu_assignment(self):
        plan = self.plan()
        for values in (dict(clips="c1"), dict(identities="i1"), dict(gpus="1")):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, "changed"):
                self.plan(**values)
        self.assertEqual(read_json(self.output / "run.json"), plan)

    def test_source_manifest_changes_are_rejected_by_resume_and_before_scoring(self):
        plan = self.plan()
        for name in ("run.json", "generation.json"):
            path = self.source / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, "changed"):
                    self.plan()
                with self.assertRaisesRegex(ValueError, "changed"):
                    source_generation(plan, self.output, self.repo)
            path.write_bytes(original)

    def test_current_metadata_config_changes_are_rejected(self):
        for name in ("models.json", "evaluation.json", "identities.json"):
            path = self.repo / "configs" / name
            original = path.read_bytes()
            path.write_bytes(original + b"\n")
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self.plan()
            path.write_bytes(original)

    def test_scoring_code_hash_changes_require_new_output(self):
        plan = self.plan()
        path = self.repo / "configs/code_checksums.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            self.plan()
        with self.assertRaisesRegex(ValueError, "changed"):
            source_generation(plan, self.output, self.repo)

    def test_missing_duplicate_foreign_or_incomplete_cells_are_rejected(self):
        variants = []
        missing = copy.deepcopy(self.generation)
        missing["entries"].pop()
        variants.append(missing)
        duplicate = copy.deepcopy(self.generation)
        duplicate["entries"][-1] = copy.deepcopy(duplicate["entries"][0])
        variants.append(duplicate)
        foreign = copy.deepcopy(self.generation)
        foreign["entries"][0]["clip"] = "c3"
        variants.append(foreign)
        for field, value in (("state", "partial"), ("state", "failed"), ("completed_cells", 3),
                             ("expected_cells", 3), ("entries", None)):
            changed = copy.deepcopy(self.generation)
            changed[field] = value
            variants.append(changed)
        for index, value in enumerate(variants):
            write_json(self.source / "generation.json", value)
            with self.subTest(index=index), self.assertRaises(ValueError):
                self.plan(clips="c1", identities="i1")
        self.assertFalse((self.output / "run.json").exists())

    def test_invalid_cell_receipts_are_rejected(self):
        for field, value in (("checkpoint_sha256", "f" * 64), ("frames", 13), ("frames", True),
                             ("pcm_sha256", "f" * 64), ("video_sha256", None),
                             ("language", "unknown"), ("identity", None), ("video", None)):
            changed = copy.deepcopy(self.generation)
            changed["entries"][0][field] = value
            write_json(self.source / "generation.json", changed)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                self.plan()

    def test_invalid_source_run_is_rejected(self):
        for field, value in (("checkpoint_step", 8999), ("expected_cells", 3),
                             ("clips", ["c1", "c1"]), ("identities", ["i3"]),
                             ("config_sha256", {}), ("source_run_root", "/another/run")):
            changed = copy.deepcopy(self.run)
            changed[field] = value
            write_json(self.source / "run.json", changed)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.plan()

    def test_missing_source_manifest_is_rejected(self):
        for name in ("run.json", "generation.json"):
            path = self.source / name
            original = path.read_bytes()
            path.unlink()
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Missing source manifest"):
                self.plan()
            path.write_bytes(original)

    def test_missing_source_video_is_rejected_even_if_not_selected(self):
        (self.source / self.generation["entries"][-1]["video"]).unlink()
        with self.assertRaisesRegex(ValueError, "Missing source video"):
            self.plan(clips="c1", identities="i1")

    def test_escaping_absolute_symlink_and_duplicate_video_paths_are_rejected(self):
        outside = self.root / "outside.mkv"
        outside.write_bytes(b"video")
        (self.source / "escape.mkv").symlink_to(outside)
        for relative in ("../outside.mkv", str(outside), "escape.mkv", self.generation["entries"][1]["video"]):
            changed = copy.deepcopy(self.generation)
            changed["entries"][0]["video"] = relative
            write_json(self.source / "generation.json", changed)
            with self.subTest(relative=relative), self.assertRaises(ValueError):
                self.plan()

    def test_nested_or_same_output_is_rejected_without_source_changes(self):
        before = self.snapshot()
        for output in (self.source, self.source / "evaluation", self.root, self.repo):
            with self.subTest(output=output), self.assertRaises(ValueError):
                plan_for_rescore(self.args(), output, self.repo)
        self.assertEqual(self.snapshot(), before)

    def test_existing_generation_output_cannot_be_reused(self):
        write_json(self.output / "generation.json", {})
        with self.assertRaisesRegex(ValueError, "new evaluation output"):
            self.plan()
        (self.output / "generation.json").unlink()
        write_json(self.output / "run.json", self.run)
        with self.assertRaisesRegex(ValueError, "another run"):
            self.plan()

    def test_invalid_gpu_values_and_visible_device_remapping_are_rejected(self):
        for value in ("", "-1", "0,0", "gpu0"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "GPU indices"):
                self.plan(gpus=value)
        with patch.dict(os.environ, CUDA_VISIBLE_DEVICES="1"):
            with self.assertRaisesRegex(ValueError, "Unset CUDA_VISIBLE_DEVICES"):
                self.plan()


if __name__ == "__main__":
    unittest.main()

# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Bootstrap safety checks; no network, model imports, or local asset dependency."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/fetch_upstreams.py"
SPEC = importlib.util.spec_from_file_location("tta_fetch_upstreams", SCRIPT)
bootstrap = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bootstrap
SPEC.loader.exec_module(bootstrap)


class BootstrapTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = {"a.py": b"first source\n", "b.py": b"second source\n"}
        self.sources = [bootstrap.Source(
            "third_party/example/" + name, bootstrap.digest(data), bootstrap.digest(data),
            "example", name) for name, data in self.data.items()]

    def download(self, manifest, key, name, expected, source_cache):
        return self.data[name]

    def fetch(self):
        with redirect_stdout(io.StringIO()):
            bootstrap.fetch(self.root, {}, self.sources)

    def test_success_and_second_run_need_no_downloads(self):
        with patch.object(bootstrap, "read_source", side_effect=self.download) as read:
            self.fetch()
            self.assertEqual(read.call_count, 2)
        with patch.object(bootstrap, "read_source") as read:
            self.fetch()
            read.assert_not_called()
        self.assertEqual(bootstrap.inspect(self.root, self.sources), ([], []))

    def test_changed_user_source_aborts_before_network_or_writes(self):
        path = self.root / self.sources[0].path
        path.parent.mkdir(parents=True)
        path.write_bytes(b"user's work")
        with patch.object(bootstrap, "read_source") as read:
            with self.assertRaisesRegex(bootstrap.BootstrapError, "Refusing to overwrite"):
                self.fetch()
            read.assert_not_called()
        self.assertEqual(path.read_bytes(), b"user's work")
        self.assertFalse((self.root / self.sources[1].path).exists())

    def test_invalid_download_leaves_all_destinations_untouched(self):
        def corrupt(manifest, key, name, expected, source_cache):
            return self.data[name] if name == "a.py" else b"invalid upstream response"
        with patch.object(bootstrap, "read_source", side_effect=corrupt):
            with self.assertRaisesRegex(bootstrap.BootstrapError, "SHA-256 mismatch"):
                self.fetch()
        for source in self.sources:
            self.assertFalse((self.root / source.path).exists())
        self.assertEqual(list((self.root / "third_party").iterdir()), [])

    def test_existing_source_changed_during_download_is_preserved(self):
        existing = self.root / self.sources[0].path
        existing.parent.mkdir(parents=True)
        existing.write_bytes(self.data["a.py"])

        def concurrent_edit(manifest, key, name, expected, source_cache):
            existing.write_bytes(b"concurrent user edit")
            return self.data[name]

        with patch.object(bootstrap, "read_source", side_effect=concurrent_edit):
            with self.assertRaisesRegex(bootstrap.BootstrapError, "changed while downloading"):
                self.fetch()
        self.assertEqual(existing.read_bytes(), b"concurrent user edit")
        self.assertFalse((self.root / self.sources[1].path).exists())

    def test_install_failure_rolls_back_only_our_new_files(self):
        stage = self.root / "stage"
        for source in self.sources:
            path = stage / source.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.data[source.upstream_path])
        real_link = bootstrap.os.link

        def conflict(origin, destination):
            if destination.name == "b.py":
                destination.write_bytes(b"concurrent file")
                raise FileExistsError(str(destination))
            real_link(origin, destination)

        with patch.object(bootstrap.os, "link", side_effect=conflict):
            with self.assertRaises(FileExistsError):
                bootstrap.install_missing(self.root, stage, self.sources)
        self.assertFalse((self.root / self.sources[0].path).exists())
        self.assertEqual((self.root / self.sources[1].path).read_bytes(), b"concurrent file")

    def test_rollback_preserves_in_place_user_edit_to_new_file(self):
        stage = self.root / "stage"
        for source in self.sources:
            path = stage / source.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(self.data[source.upstream_path])
        real_link = bootstrap.os.link
        edited = self.root / self.sources[0].path

        def edit_during_install(origin, destination):
            if destination.name == "b.py":
                edited.write_bytes(b"user edit after creation")
                raise OSError("simulated installation failure")
            real_link(origin, destination)

        with patch.object(bootstrap.os, "link", side_effect=edit_during_install):
            with self.assertRaises(OSError):
                bootstrap.install_missing(self.root, stage, self.sources)
        self.assertEqual(edited.read_bytes(), b"user edit after creation")

    def test_source_symlink_is_rejected(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.root / "third_party").symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(bootstrap.BootstrapError, "symbolic link"):
            bootstrap.inspect(self.root, self.sources)
        self.assertEqual(list(outside.iterdir()), [])

    def test_public_inventory_contains_only_source_metadata(self):
        manifest, sources = bootstrap.load_sources()
        inventory = json.loads((bootstrap.REPO / manifest["inventory"]).read_text())
        self.assertEqual(set(inventory), {"schema_version", "vendor_files", "official_syncnet"})
        for row in inventory["vendor_files"]:
            self.assertEqual(set(row), {"path", "sha256", "original_sha256", "modified"})
        official = inventory["official_syncnet"]
        self.assertEqual(set(official), {"commits", "sources"})
        for row in official["sources"]:
            self.assertEqual(set(row), {"path", "sha256"})
        self.assertEqual(set(inventory["official_syncnet"]["commits"]), {"syncnet", "wav2lip"})

    def test_manifest_has_exact_three_patches_and_seven_allowlisted_repos(self):
        manifest, sources = bootstrap.load_sources()
        self.assertEqual(len(sources), 287)
        self.assertEqual(len(bootstrap.PATCHES), 3)
        self.assertEqual(set(manifest["repositories"]), set(bootstrap.ALLOWED_REPOSITORIES))
        self.assertEqual(sum(source.generated for source in sources), 15)


if __name__ == "__main__":
    unittest.main()

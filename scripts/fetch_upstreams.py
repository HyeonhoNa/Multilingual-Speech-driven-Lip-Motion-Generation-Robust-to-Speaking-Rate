#!/usr/bin/env python3
"""Restore the pinned runtime source subset; never fetch models or datasets.

Uses only Python's standard library. Existing matching files are reused. Modified
files are reported and never overwritten. Every download and generated metadata
file is checked before any missing destination file is installed. No downloaded
code, repository setup script, shell command, or model download script is run.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
import urllib.error
import urllib.request

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "scripts/upstream_sources.json"
# This allowlist is deliberately independent of the JSON manifest.
ALLOWED_REPOSITORIES = {
    "artalk": "xg-chu/ARTalk",
    "flexavatar": "tobias-kirschstein/flexavatar",
    "modnet": "tobias-kirschstein/MODNet",
    "stylegan": "tobias-kirschstein/stylegan-generators",
    "dinov2": "facebookresearch/dinov2",
    "syncnet": "joonson/syncnet_python",
    "wav2lip": "Rudrabha/Wav2Lip",
}
PATCHES = {
    "third_party/ARTalk/app/models.py": [
        ("import math\n", "import math\nimport os\n"),
        ('Wav2Vec2Config.from_pretrained("facebook/wav2vec2-xls-r-300m")',
         'Wav2Vec2Config.from_pretrained(os.environ["TTA_WAV2VEC_CONFIG"], local_files_only=True)'),
    ],
    "third_party/FlexAvatar/src/flexavatar/env.py": [
        ("from pathlib import Path\n", "from pathlib import Path\nimport os\n"),
        ('ASSETS_PATH = f"{REPO_ROOT}/assets"',
         'ASSETS_PATH = os.environ.get("TTA_FLEX_ASSETS", f"{REPO_ROOT}/assets")'),
    ],
    "third_party/FlexAvatar/src/flexavatar/model/dinov2.py": [
        ("from math import sqrt\n", "from math import sqrt\nimport os\n"),
        ("        self._dino = torch.hub.load('facebookresearch/dinov2:85a24602099d397264d5b30461ad7f3bfd726ca1', dino_name,\n"
         "                                    trust_repo=True, skip_validation=True)",
         "        self._dino = torch.hub.load(os.environ['TTA_DINO_SOURCE'], dino_name,\n"
         "                                    source='local', pretrained=False)\n"
         "        self._dino.load_state_dict(torch.load(os.environ['TTA_DINO_WEIGHT'],\n"
         "                                              map_location='cpu', weights_only=True), strict=True)"),
    ],
}


class BootstrapError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise BootstrapError(message)


def relative(value):
    path = PurePosixPath(value)
    require(isinstance(value, str) and value and not path.is_absolute()
            and ".." not in path.parts and str(path) == value
            and "\\" not in value, f"Unsafe relative path: {value!r}")
    return path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify(data, expected, label):
    require(digest(data) == expected, f"SHA-256 mismatch: {label}")
    return data


@dataclass(frozen=True)
class Source:
    path: str
    sha256: str
    original_sha256: str
    repository: str
    upstream_path: str
    generated: bool = False


def load_sources():
    manifest = json.loads(MANIFEST.read_text())
    require(manifest["schema_version"] == 1, "Unsupported source manifest version")
    repos = manifest["repositories"]
    require(set(repos) == set(ALLOWED_REPOSITORIES), "Unexpected repository set")
    for key, repo in repos.items():
        require(repo["repository"] == ALLOWED_REPOSITORIES[key]
                and repo["url"] == f"https://github.com/{ALLOWED_REPOSITORIES[key]}.git",
                f"Repository is not allowlisted: {key}")
        require(re.fullmatch(r"[0-9a-f]{40}", repo["commit"]), f"Unpinned commit: {key}")
        relative(repo["destination"])
        if repo.get("source_prefix"):
            relative(repo["source_prefix"])
    require(manifest["inventory"] == "scripts/upstream_inventory.json", "Unexpected inventory path")
    inventory = json.loads((REPO / manifest["inventory"]).read_text())
    require(inventory["schema_version"] == 1, "Unsupported source inventory version")
    official = inventory["official_syncnet"]
    for key, commit in official["commits"].items():
        require(repos[key]["commit"] == commit, f"Evaluator revision mismatch: {key}")
    sources = []
    for row in inventory["vendor_files"]:
        matches = [key for key, repo in repos.items()
                   if row["path"].startswith(repo["destination"] + "/")]
        require(len(matches) == 1, f"Unknown source destination: {row['path']}")
        key = matches[0]
        upstream_path = row["path"][len(repos[key]["destination"]) + 1:]
        generated = any(part.endswith(".egg-info") for part in relative(upstream_path).parts)
        require(bool(row["modified"]) == (row["path"] in PATCHES),
                f"Unsupported source patch: {row['path']}")
        sources.append(Source(row["path"], row["sha256"], row["original_sha256"],
                              key, upstream_path, generated))
    for row in official["sources"]:
        prefix = repos["syncnet"]["destination"] + "/"
        require(row["path"].startswith(prefix), f"Unexpected evaluator path: {row['path']}")
        name = row["path"][len(prefix):]
        key = "wav2lip" if name in manifest["syncnet_overlay"] else "syncnet"
        upstream_path = "/".join(filter(None, (repos[key].get("source_prefix"), name)))
        sources.append(Source(row["path"], row["sha256"], row["sha256"], key, upstream_path))
    require(len({source.path for source in sources}) == len(sources), "Duplicate source path")
    for source in sources:
        relative(source.path)
        relative(source.upstream_path)
        for expected in (source.sha256, source.original_sha256):
            require(re.fullmatch(r"[0-9a-f]{64}", expected), f"Invalid hash: {source.path}")
    return manifest, sources


def target_path(root, name):
    path = root.joinpath(*relative(name).parts)
    # Refuse links for sources, while leaving separately supplied asset links alone.
    for candidate in (path, *path.parents):
        if candidate == root:
            break
        require(not candidate.is_symlink(), f"Source path is a symbolic link: {candidate}")
        if candidate != path and candidate.exists():
            require(candidate.is_dir(), f"Source parent is not a directory: {candidate}")
    return path


def inspect(root, sources):
    missing, changed = [], []
    for source in sources:
        path = target_path(root, source.path)
        if not path.exists():
            missing.append(source)
        elif not path.is_file() or digest(path.read_bytes()) != source.sha256:
            changed.append(source.path)
    return missing, changed


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise BootstrapError(f"Unexpected source redirect: {request.full_url}")


def read_source(manifest, key, name, expected, source_cache=None):
    relative(name)
    if source_cache:
        local = source_cache / key / name
        if local.is_file():
            return verify(local.read_bytes(), expected, str(local))
    repo = manifest["repositories"][key]
    url = f"https://raw.githubusercontent.com/{repo['repository']}/{repo['commit']}/{name}"
    request = urllib.request.Request(url, headers={"User-Agent": "tta-lip-motion-source-bootstrap/1"})
    try:
        with urllib.request.build_opener(NoRedirect).open(request, timeout=30) as response:
            data = response.read(8 * 1024 * 1024 + 1)
        require(len(data) <= 8 * 1024 * 1024, f"Source exceeds 8 MiB limit: {url}")
    except (urllib.error.URLError, TimeoutError) as error:
        raise BootstrapError(f"Cannot fetch {url}: {error}") from error
    return verify(data, expected, url)


def metadata_files(manifest, sources, key, source_cache=None):
    """Rebuild the recorded setuptools metadata without executing upstream code.

    PKG-INFO headers are declarative package metadata in the manifest. README
    bytes are fetched from the pinned official commit. SOURCES.txt is derived
    from the source inventory plus the recorded build/test filenames.
    """
    config = manifest["generated_egg_info"][key]
    package = config["package"]
    header = "\n".join(config["pkg_info_headers"]).encode() + b"\n"
    if config.get("readme"):
        readme = config["readme"]
        data = read_source(manifest, key, readme["path"], readme["sha256"], source_cache)
        header += b"\n" + data.rstrip(b"\n") + b"\n"
    paths = [source.upstream_path for source in sources
             if source.repository == key and
             (source.upstream_path.startswith(f"src/{package}/") or
              source.upstream_path.startswith(f"src/{package}.egg-info/"))]
    paths.extend(config["sources_extra"])
    paths.sort(key=lambda name: (str(PurePosixPath(name).parent), PurePosixPath(name).name))
    return {
        "PKG-INFO": header,
        "SOURCES.txt": "\n".join(paths).encode(),
        "dependency_links.txt": b"\n",
        "requires.txt": config["requires"].encode(),
        "top_level.txt": (package + "\n").encode(),
    }


def patched(source, data):
    verify(data, source.original_sha256, source.upstream_path)
    for before, after in PATCHES.get(source.path, []):
        before, after = before.encode(), after.encode()
        require(data.count(before) == 1, f"Local path patch context changed: {source.path}")
        data = data.replace(before, after, 1)
    return verify(data, source.sha256, source.path)


def install_missing(root, staged, sources):
    """Exclusive hard links publish verified bytes without replacing any file.

    On failure remove only links created by this invocation, checking their inode
    and content first so concurrent user replacements or edits are preserved.
    """
    installed, directories = [], []
    try:
        for source in sources:
            destination = target_path(root, source.path)
            if destination.exists():
                verify(destination.read_bytes(), source.sha256, source.path)
                continue
            absent = []
            parent = destination.parent
            while not parent.exists():
                absent.append(parent)
                parent = parent.parent
            for directory in reversed(absent):
                directory.mkdir(exist_ok=True)
                directories.append(directory)
            origin = staged / source.path
            os.link(origin, destination)
            installed.append((destination, origin.stat(), source.sha256))
        missing, changed = inspect(root, sources)
        require(not missing and not changed, "Source tree changed during installation")
    except BaseException:
        for destination, original, expected in reversed(installed):
            try:
                current = destination.lstat()
                if ((current.st_dev, current.st_ino) == (original.st_dev, original.st_ino)
                        and digest(destination.read_bytes()) == expected):
                    destination.unlink()
            except FileNotFoundError:
                pass
        for directory in reversed(directories):
            try:
                directory.rmdir()
            except OSError:
                pass
        raise
    return len(installed)


def fetch(root, manifest, sources, source_cache=None):
    missing, changed = inspect(root, sources)
    require(not changed, "Refusing to overwrite changed source files:\n  " + "\n  ".join(changed))
    if not missing:
        print(f"All {len(sources)} source files match; no downloads needed.")
        return
    print(f"Restoring {len(missing)} source files; reusing {len(sources) - len(missing)} verified files.", flush=True)
    stage_parent = target_path(root, "third_party")
    stage_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".upstream-staging-", dir=stage_parent) as temp:
        staged = Path(temp)
        generated = {
            key: metadata_files(manifest, sources, key, source_cache)
            for key in sorted({source.repository for source in missing if source.generated})
        }

        def prepare(source):
            if source.generated:
                data = generated[source.repository][PurePosixPath(source.path).name]
            else:
                data = read_source(manifest, source.repository, source.upstream_path,
                                   source.original_sha256, source_cache)
            data = patched(source, data)
            path = staged / source.path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            path.chmod(0o755 if path.suffix == ".sh" else 0o644)

        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(prepare, missing))
        # Recheck the complete pre-existing tree after downloads and before writes.
        _, changed = inspect(root, sources)
        require(not changed, "Source files changed while downloading; nothing installed:\n  " + "\n  ".join(changed))
        installed = install_missing(root, staged, missing)
    missing, changed = inspect(root, sources)
    require(not missing and not changed, "Source verification failed after installation")
    print(f"Installed {installed} files. All {len(sources)} source hashes match.")
    print("Weights, datasets, private compatibility modules, and environment archives must be supplied separately.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="Verify source hashes without network access or writes")
    mode.add_argument("--fetch", action="store_true", help="Restore missing pinned source files (default)")
    parser.add_argument("--root", type=Path, default=REPO, help="Destination repository root")
    parser.add_argument("--source-cache", type=Path,
                        help="Optional local upstream checkouts at DIRECTORY/<repository key>/; verify every reused byte")
    args = parser.parse_args(argv)
    try:
        manifest, sources = load_sources()
        root = args.root.resolve()
        if args.check:
            missing, changed = inspect(root, sources)
            for source in missing:
                print("MISSING " + source.path)
            for name in changed:
                print("CHANGED " + name)
            if missing or changed:
                print(f"Source check failed: {len(missing)} missing, {len(changed)} changed.", file=sys.stderr)
                return 1
            print(f"All {len(sources)} source hashes match (including three local path patches).")
        else:
            fetch(root, manifest, sources, args.source_cache)
    except (BootstrapError, OSError, ValueError, KeyError) as error:
        print(f"Source bootstrap failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

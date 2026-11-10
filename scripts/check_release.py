#!/usr/bin/env python3
# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Inspect staged Git objects; local ignored TTA assets remain untouched."""
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {".gitignore", "README.md", "THIRD_PARTY.md", "CITATIONS.bib", "LICENSE", "LICENSING.md", "DEPENDENCY_EXCEPTION.md"}
TOP_DIRS = {"tta_pipeline", "scripts", "tests", "docs", "LICENSES"}
PRIVATE_MODULES = {"tta_pipeline/_runtime/rate.py", "tta_pipeline/_runtime/deepspeech.py"}
BINARY = {".pt", ".pth", ".ckpt", ".model", ".pb", ".safetensors", ".npy", ".npz",
          ".pkl", ".pickle", ".wav", ".mp3", ".flac", ".mp4", ".avi", ".mkv",
          ".jpg", ".jpeg", ".png", ".so", ".o", ".zip", ".gz", ".tar"}
PRIVATE_PATH = re.compile(rb"/(?:source|data[0-9]*)/[A-Za-z][^/\s]+/(?:starlab|\.codex)|/root/\.(?:ssh|aws|codex|cache)")

def main():
    index = subprocess.check_output(["git", "ls-files", "--stage", "-z"], cwd=ROOT)
    errors, total, names = [], 0, []
    for record in index.split(b"\0"):
        if not record:
            continue
        metadata, rawname = record.split(b"\t", 1)
        mode, oid, stage = metadata.decode().split()
        name = rawname.decode("utf-8")
        path = PurePosixPath(name)
        names.append(name)
        if name not in TOP_FILES and (not path.parts or path.parts[0] not in TOP_DIRS) and name != "run.py":
            errors.append(name + ": internal or non-release path")
        if name in PRIVATE_MODULES or path.suffix.lower() in BINARY:
            errors.append(name + ": excluded compatibility module or binary/data file")
        if mode not in {"100644", "100755"} or stage != "0":
            errors.append(name + ": symlink/submodule/unmerged entry")
            continue
        content = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=ROOT)
        total += len(content)
        if b"\0" in content or len(content) > 2_000_000:
            errors.append(name + ": binary or unexpectedly large object")
        if PRIVATE_PATH.search(content):
            errors.append(name + ": private deployment path")
    required = {"LICENSE", "LICENSING.md", "DEPENDENCY_EXCEPTION.md", "THIRD_PARTY.md",
                "LICENSES/ARTalk.txt", "LICENSES/Apache-2.0.txt",
                "LICENSES/FlexAvatar-CC-BY-NC-4.0.txt", "LICENSES/README.md"}
    for missing in sorted(required - set(names)):
        errors.append(missing + ": required license/scope notice missing from Git index")
    if not names:
        errors.append("No files staged/tracked. Run git add on the intended source files first.")
    if errors:
        print("\n".join(errors), file=sys.stderr)
        raise SystemExit(1)
    print(json.dumps({"release_check": "passed", "git_files": len(names), "git_bytes": total,
                      "third_party_trees_in_git": False, "weights_or_data_in_git": False,
                      "local_files_deleted": False}, indent=2))
    return 0

if __name__ == "__main__":
    main()

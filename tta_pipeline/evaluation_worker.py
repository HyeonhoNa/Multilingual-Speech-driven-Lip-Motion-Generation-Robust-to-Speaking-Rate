# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Score an assigned list of generated videos, preserving completed cells."""
import argparse
import datetime
import os
import math
import signal
from pathlib import Path

from .io import inside, read_json, sha256, write_json


def run(args):
    repo, output = args.repo.resolve(), args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignment = read_json(args.assignment)
    assigned_keys = [(r["clip"], r["identity"]) for r in assignment["entries"]]
    if not assigned_keys or len(set(assigned_keys)) != len(assigned_keys):
        raise ValueError("Empty or duplicate evaluation assignment")
    manifest = read_json(repo / "configs/evaluation.json")
    entries = {entry["clip"]: entry for entry in manifest["entries"]}
    from .evaluation import evaluate_video

    path = output / "results.json"
    result = dict(schema=1, assignment=assignment, state="running", scores=[], error=None)
    if path.exists():
        result = read_json(path)
        if result["assignment"] != assignment:
            raise ValueError("Evaluation assignment changed. Use a new output directory.")
    completed = {(row["clip"], row["identity"]): row for row in result["scores"]}
    if len(completed) != len(result["scores"]) or set(completed) - set(assigned_keys):
        raise ValueError("Unexpected or duplicate saved evaluation scores")
    for row in completed.values():
        if row.get("error") or row.get("status") != "ok" or row.get("strict_eligible") is not True or not all(math.isfinite(float(row[k])) for k in ("lse_d", "lse_c")):
            raise ValueError("Invalid saved official score")
    result.update(state="running", error=None)
    write_json(path, result)
    try:
        for item in assignment["entries"]:
            key = item["clip"], item["identity"]
            video = inside(assignment["run_root"], item["video"])
            digest = sha256(video)
            if digest != item["video_sha256"]:
                raise ValueError(f"Generated video changed: {video}")
            if key in completed:
                if completed[key]["video_sha256"] != digest:
                    raise ValueError(f"Previously scored video changed: {video}")
                continue
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            attempt = output / "cells" / item["clip"] / item["identity"] / stamp
            score = evaluate_video(video, entries[item["clip"]], attempt,
                                   gpu=args.gpu, repo_root=repo)
            score.update(clip=item["clip"], identity=item["identity"],
                         language=entries[item["clip"]]["language"],
                         rate_bin=entries[item["clip"]]["rate_bin"], video=item["video"],
                         video_sha256=digest)
            if score.get("error") or score.get("status") != "ok" or score.get("strict_eligible") is not True or not all(score.get(k) is not None and math.isfinite(float(score[k])) for k in ("lse_d","lse_c")):
                raise RuntimeError(f"Official scoring failed for {key}")
            result["scores"].append(score)
            completed[key] = score
            result["completed_cells"] = len(completed)
            write_json(path, result)
            print(f"Scored {len(completed)}/{len(assignment['entries'])}: {key[0]} / {key[1]}", flush=True)
        result["state"] = "complete"
        write_json(path, result)
    except BaseException as exc:
        result.update(state="failed", error=str(exc))
        write_json(path, result)
        raise


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt("Evaluation stopped")
    signal.signal(signal.SIGTERM, interrupted)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--gpu", type=int, required=True)
    run(parser.parse_args())

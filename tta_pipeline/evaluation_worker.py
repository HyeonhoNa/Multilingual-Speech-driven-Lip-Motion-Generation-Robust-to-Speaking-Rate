# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Score an assigned list of generated videos, preserving completed cells."""
import argparse
import datetime
import signal
from pathlib import Path

from .io import inside, read_json, sha256, write_json
from .score_output import emit_failure, emit_score, valid_score


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
        if not valid_score(row):
            raise ValueError("Invalid saved official score")
    result.update(state="running", error=None, failed_cell=None)
    write_json(path, result)
    current = None
    try:
        for item in assignment["entries"]:
            current = item
            key = item["clip"], item["identity"]
            video = inside(assignment["run_root"], item["video"])
            digest = sha256(video)
            if digest != item["video_sha256"]:
                raise ValueError(f"Generated video changed: {video}")
            if key in completed:
                if completed[key]["video_sha256"] != digest:
                    raise ValueError(f"Previously scored video changed: {video}")
                emit_score(completed[key], cached=True)
                continue
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            attempt = output / "cells" / item["clip"] / item["identity"] / stamp
            score = evaluate_video(video, entries[item["clip"]], attempt,
                                   gpu=args.gpu, repo_root=repo)
            score.update(clip=item["clip"], identity=item["identity"],
                         language=entries[item["clip"]]["language"],
                         rate_bin=entries[item["clip"]]["rate_bin"], video=item["video"],
                         video_sha256=digest)
            if not valid_score(score):
                raise RuntimeError("Official scoring failed for %s: %s" %
                                   (key, score.get("error") or "invalid or ineligible score"))
            result["scores"].append(score)
            completed[key] = score
            result["completed_cells"] = len(completed)
            write_json(path, result)
            emit_score(score)
        result["state"] = "complete"
        write_json(path, result)
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        failure = dict(clip=current["clip"], identity=current["identity"], error=error) if current else None
        result.update(state="failed", error=error, failed_cell=failure)
        write_json(path, result)
        if failure:
            emit_failure(failure)
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

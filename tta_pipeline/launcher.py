# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Standard-library launcher for the frozen TTA reproduction package."""
import argparse
import collections
import csv
import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .io import inside, read_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def require_local_installation(repo):
    required = ["assets/checksums.json", "configs/models.json", "configs/evaluation.json",
                "configs/identities.json", "tta_pipeline/_runtime/rate.py",
                "tta_pipeline/_runtime/deepspeech.py"]
    missing = [name for name in required if not (repo / name).is_file()]
    if missing:
        raise SystemExit("This source checkout needs the separately supplied TTA assets/modules. "
                         "See docs/ASSETS.md. Missing: " + ", ".join(missing))


def verify(repo, quick=False):
    require_local_installation(repo)
    inventory = read_json(repo / "assets/checksums.json")["files"]
    optional = repo / "configs/code_checksums.json"
    if optional.exists():
        inventory += read_json(optional)["files"]
    runtime = repo / "assets/runtime_checksums.json"
    if runtime.exists():
        inventory += read_json(runtime)["files"]
    for row in inventory:
        path = inside(repo, row["path"])
        if not path.is_file() or path.stat().st_size != row["bytes"]:
            raise ValueError("Missing or changed file: " + str(path))
        if not quick and sha256(path) != row["sha256"]:
            raise ValueError("Checksum mismatch: " + str(path))
    config = read_json(repo / "configs/evaluation.json")
    entries, ids = config["entries"], config["identities"]
    counts = collections.Counter((e["language"], e["rate_bin"]) for e in entries)
    if len(entries) != 90 or len({e["clip"] for e in entries}) != 90 or len(set(ids)) != 10:
        raise ValueError("The fixed cohort must contain 90 unique clips and 10 identities")
    if len(counts) != 45 or set(counts.values()) != {2}:
        raise ValueError("Expected 15 languages x 3 speed bins x 2 recordings")
    print("Verified %d files; 15 languages / 90 recordings / 10 identities." % len(inventory), flush=True)


def csv_values(value):
    return [s.strip() for s in value.split(",") if s.strip()]


def plan_for(args, output):
    config = read_json(ROOT / "configs/evaluation.json")
    all_clips = [e["clip"] for e in config["entries"]]
    all_ids = config["identities"]
    old = read_json(output / "run.json") if (output / "run.json").exists() else None
    clips = csv_values(args.clips) if args.clips else (old["clips"] if old else all_clips)
    ids = csv_values(args.identities) if args.identities else (old["identities"] if old else all_ids)
    if args.limit:
        if args.clips:
            raise ValueError("Use --limit or --clips, not both")
        clips = all_clips[:args.limit]
    if not clips or not ids or len(set(clips)) != len(clips) or len(set(ids)) != len(ids):
        raise ValueError("Clip and identity lists must be nonempty and unique")
    if set(clips) - set(all_clips) or set(ids) - set(all_ids):
        raise ValueError("Only identifiers in configs/evaluation.json may be selected")
    clips = [x for x in all_clips if x in clips]
    ids = [x for x in all_ids if x in ids]
    gpus = [int(s) for s in csv_values(args.gpus)] if args.gpus else (old["gpus"] if old else [0])
    if not gpus or min(gpus) < 0 or len(set(gpus)) != len(gpus):
        raise ValueError("Supply distinct nonnegative GPU indices")
    # These indices are physical GPU indices, matching the official scorer's CLI.
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Unset CUDA_VISIBLE_DEVICES; select GPUs using --gpus")
    hashes = {p.name: sha256(p) for p in sorted((ROOT / "configs").glob("*.json"))}
    runtime = ROOT / "assets/runtime_checksums.json"
    if runtime.exists():
        hashes["runtime_checksums.json"] = sha256(runtime)
    plan = dict(schema=1, checkpoint_step=9000, clips=clips, identities=ids,
                gpus=gpus, config_sha256=hashes, expected_cells=len(clips) * len(ids),
                full_fixed_cohort=len(clips) == 90 and len(ids) == 10)
    if old:
        if any(old.get(k) != v for k, v in plan.items()):
            raise ValueError("Run configuration changed. Keep the saved settings or use a new --output")
        return old
    plan["created_utc"] = now()
    plan["workers"] = [dict(index=i, gpu=gpu, clips=clips[i::len(gpus)])
                       for i, gpu in enumerate(gpus) if clips[i::len(gpus)]]
    write_json(output / "run.json", plan)
    return plan


def environment(interpreter=None):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", HF_HOME=str(ROOT / ".cache/huggingface"),
               TORCH_HOME=str(ROOT / ".cache/torch"),
               PATH=str(ROOT / "media/bin") + os.pathsep + env.get("PATH", ""))
    if interpreter and Path(interpreter) == ROOT / "envs/generation/bin/python":
        prefix = ROOT / "envs/generation"
        env["CUDA_HOME"] = str(prefix)
        env["PATH"] = str(ROOT / "media/bin") + os.pathsep + str(prefix / "bin") + os.pathsep + env["PATH"]
        paths = [str(ROOT / "media/cuda/lib"), str(prefix / "lib")]
        if env.get("LD_LIBRARY_PATH"):
            paths.append(env["LD_LIBRARY_PATH"])
        env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    return env


def python_for(name):
    path = ROOT / "envs" / name / "bin/python"
    if not path.is_file():
        raise FileNotFoundError("Run bash scripts/setup_envs.sh first: " + str(path))
    archive = next(row for row in read_json(ROOT / "environments/archives.json") if row["name"] == name)
    marker = path.parents[1] / ".tta-environment.json"
    expected = dict(prefix=str(path.parents[1]), archive_sha256=archive["sha256"])
    if not marker.is_file() or read_json(marker) != expected:
        raise ValueError("Environment was moved or not installed here. Run bash scripts/setup_envs.sh --recreate")
    return str(path)


def run_workers(jobs, output, stage):
    children = []
    try:
        for worker, command in jobs:
            logfile = output / "logs" / (stage + "_%03d.log" % worker)
            logfile.parent.mkdir(parents=True, exist_ok=True)
            stream = logfile.open("a", encoding="utf-8")
            process = subprocess.Popen(command, cwd=str(ROOT), env=environment(command[0]),
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            children.append((process, stream, logfile))
        failures = []
        for process, stream, logfile in children:
            rc = process.wait()
            stream.close()
            if rc:
                failures.append("%s (exit %d)" % (logfile, rc))
        if failures:
            raise RuntimeError("Worker failed; completed work is preserved: " + "; ".join(failures))
    finally:
        for process, stream, logfile in children:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for process, stream, logfile in children:
            if process.poll() is None:
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
            if not stream.closed:
                stream.close()


def prepare(plan, output):
    jobs = []
    for w in plan["workers"]:
        folder = output / "workers" / ("%03d" % w["index"])
        jobs.append((w["index"], [python_for("generation"), "-m", "tta_pipeline.prepare",
                     "--repo", str(ROOT), "--output-dir", str(folder),
                     "--clips", ",".join(w["clips"])]))
    run_workers(jobs, output, "prepare")


def generation_manifest(plan, output):
    entries = []
    for w in plan["workers"]:
        prefix = Path("workers") / ("%03d" % w["index"])
        path = output / prefix / "generation.json"
        if not path.exists():
            continue
        result = read_json(path)
        for raw in result.get("entries", []):
            entry = dict(raw)
            for key in ("video", "motion"):
                if key in entry:
                    entry[key] = str(prefix / entry[key])
            entries.append(entry)
    keys = [(r["clip"], r["identity"]) for r in entries]
    expected = {(c, i) for c in plan["clips"] for i in plan["identities"]}
    if len(set(keys)) != len(keys) or set(keys) - expected:
        raise ValueError("Unexpected or duplicate generated cells")
    result = dict(state="complete" if set(keys) == expected else "partial", entries=entries,
                  expected_cells=len(expected), completed_cells=len(entries))
    write_json(output / "generation.json", result)
    return result


def generate(plan, output):
    jobs = []
    for w in plan["workers"]:
        folder = output / "workers" / ("%03d" % w["index"])
        jobs.append((w["index"], [python_for("generation"), "-m", "tta_pipeline.generation",
                     "--repo", str(ROOT), "--output-dir", str(folder), "--gpu", str(w["gpu"]),
                     "--clips", ",".join(w["clips"]), "--identities", ",".join(plan["identities"])]))
    try:
        run_workers(jobs, output, "generate")
    finally:
        result = generation_manifest(plan, output)
    if result["state"] != "complete":
        raise RuntimeError("Generation ended without the full requested cohort")


def evaluate(plan, output):
    generated = generation_manifest(plan, output)
    if generated["state"] != "complete":
        raise RuntimeError("Complete generation first; no silent omission of missing videos")
    jobs = []
    for w in plan["workers"]:
        folder = output / "evaluation" / ("%03d" % w["index"])
        assignment = dict(run_root=str(output), entries=[r for r in generated["entries"] if r["clip"] in w["clips"]])
        assignment_path = folder / "assignment.json"
        if assignment_path.exists() and read_json(assignment_path) != assignment:
            raise ValueError("Evaluation inputs changed: " + str(assignment_path))
        write_json(assignment_path, assignment)
        jobs.append((w["index"], [python_for("evaluation"), "-m", "tta_pipeline.evaluation_worker",
                     "--repo", str(ROOT), "--output-dir", str(folder), "--gpu", str(w["gpu"]),
                     "--assignment", str(assignment_path)]))
    try:
        run_workers(jobs, output, "evaluate")
    finally:
        summary = report(output)
    if summary["state"] != "complete":
        raise RuntimeError("Evaluation ended without the complete requested cohort")


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(output):
    plan = read_json(output / "run.json")
    rows = []
    for path in sorted((output / "evaluation").glob("*/results.json")):
        rows.extend(read_json(path).get("scores", []))
    expected = {(c, i) for c in plan["clips"] for i in plan["identities"]}
    keys = [(r["clip"], r["identity"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) - expected:
        raise ValueError("Duplicate or unexpected official results")
    for row in rows:
        if row.get("error") or row.get("status") != "ok" or row.get("strict_eligible") is not True or not all(math.isfinite(float(row[k])) for k in ("lse_d", "lse_c")):
            raise ValueError("Failed/nonfinite scores cannot enter the aggregate")
        row["pass_both"] = row["lse_d"] <= 6.5 and row["lse_c"] >= 8.0
    def stats(group):
        return dict(cells=len(group), lse_d=sum(r["lse_d"] for r in group)/len(group),
                    lse_c=sum(r["lse_c"] for r in group)/len(group),
                    pass_both=sum(r["pass_both"] for r in group)) if group else dict(cells=0, lse_d=None,lse_c=None,pass_both=0)
    summary = dict(state="complete" if len(rows) == len(expected) else "partial",
                   full_fixed_cohort=plan["full_fixed_cohort"], expected_cells=len(expected),
                   checkpoint_step=9000, thresholds=dict(lse_d_max=6.5,lse_c_min=8.0),
                   updated_utc=now(), **stats(rows))
    current_mean_passes = bool(rows and summary["lse_d"] <= 6.5 and summary["lse_c"] >= 8)
    summary["mean_passes"] = current_mean_passes if summary["state"] == "complete" else None
    if summary["state"] != "complete":
        summary["partial_mean_passes"] = current_mean_passes
    reference_path = ROOT / "reference_results/cell.csv"
    if reference_path.exists() and rows:
        with reference_path.open(encoding="utf-8-sig") as f:
            gold = {(r["clip"],r["identity"]):r for r in csv.DictReader(f)}
        matched = [r for r in rows if (r["clip"],r["identity"]) in gold]
        if matched:
            first = next(iter(gold.values()))
            dk, ck = ("lse_d", "lse_c") if "lse_d" in first else ("D", "C")
            if dk in first and ck in first:
                summary["reference_comparison"] = dict(matched_cells=len(matched),
                    mean_delta_d=sum(r["lse_d"]-float(gold[(r["clip"],r["identity"])][dk]) for r in matched)/len(matched),
                    mean_delta_c=sum(r["lse_c"]-float(gold[(r["clip"],r["identity"])][ck]) for r in matched)/len(matched))
    write_json(output / "results.json", dict(scores=rows))
    write_json(output / "summary.json", summary)
    write_csv(output / "cell.csv", rows, ["clip","identity","language","rate_bin","lse_d","lse_c","pass_both","video"])
    for label, key in (("language","language"),("identity","identity"),("utterance","clip")):
        groups = collections.defaultdict(list)
        for row in rows:
            groups[row[key]].append(row)
        grouped = [dict(**{key:k},**stats(v)) for k,v in groups.items()]
        grouped.sort(key=lambda r:r["lse_c"],reverse=True)
        write_csv(output / (label+".csv"), grouped, [key,"cells","lse_d","lse_c","pass_both"])
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Frozen ARTalk9000 / AW100 / FlexAvatar official evaluation")
    parser.add_argument("command", choices=["verify","prepare","generate","evaluate","run","status","report"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tta900")
    parser.add_argument("--gpus", help="Physical GPU indices, e.g. 0,1; default 0 or saved run settings")
    parser.add_argument("--clips", help="Comma-separated exact manifest clip identifiers")
    parser.add_argument("--identities", help="Comma-separated exact identity identifiers; default all ten")
    parser.add_argument("--limit", type=int, help="Smoke test: first N clips, clearly marked partial cohort")
    parser.add_argument("--quick", action="store_true", help="verify only: check sizes without reading file hashes")
    parser.add_argument("--background", action="store_true", help="run only: detach, log and preserve PID")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 90:
        parser.error("--limit must be between 1 and 90")
    output = args.output.resolve()
    if args.command == "verify":
        verify(ROOT, args.quick)
        return
    if args.command in ("status", "report"):
        if not (output / "run.json").exists():
            parser.error("No run found: " + str(output))
        if args.command == "report":
            report(output)
        else:
            status = read_json(output / "status.json") if (output / "status.json").exists() else {}
            status["generated_cells"] = generation_manifest(read_json(output / "run.json"), output)["completed_cells"]
            status["evaluated_cells"] = sum(len(read_json(p).get("scores",[])) for p in (output / "evaluation").glob("*/results.json"))
            print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    require_local_installation(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    if args.background:
        if args.command != "run":
            parser.error("--background is supported only for run")
        command = [sys.executable, str(ROOT / "run.py")] + [a for a in sys.argv[1:] if a != "--background"]
        with (output / "launcher.log").open("a") as log:
            child = subprocess.Popen(command, cwd=os.getcwd(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        write_json(output / "background.json", dict(pid=child.pid, launched_utc=now(), command=command))
        print("Started PID %d; log: %s" % (child.pid, output / "launcher.log"))
        return
    lock = (output / ".lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("This output directory is already being processed")
    signal.signal(signal.SIGTERM, lambda signum, frame: (_ for _ in ()).throw(KeyboardInterrupt()))
    status = dict(state="running", command=args.command, started_utc=now(), pid=os.getpid())
    write_json(output / "status.json", status)
    try:
        plan = plan_for(args, output)
        verify(ROOT)
        stages = ["prepare","generate","evaluate"] if args.command == "run" else [args.command]
        for stage in stages:
            status.update(stage=stage, updated_utc=now())
            write_json(output / "status.json", status)
            print("Starting " + stage + " at " + now(), flush=True)
            globals()[stage](plan, output)
        status.update(state="complete", finished_utc=now())
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), finished_utc=now())
        raise
    finally:
        write_json(output / "status.json", status)
        lock.close()

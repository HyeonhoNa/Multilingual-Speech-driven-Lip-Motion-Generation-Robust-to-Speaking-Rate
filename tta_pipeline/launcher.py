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
from .score_output import EVENT_PREFIX, ScorePrinter, print_summary, valid_score

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


class EventTail:
    """Read only newly appended worker output, with bounded per-worker buffers."""
    limit = 64 * 1024

    def __init__(self, path):
        self.stream = path.open("rb")
        self.stream.seek(0, os.SEEK_END)
        self.pending = b""
        self.discarding = False

    def read(self, callback):
        data = self.stream.read(self.limit)
        for index, fragment in enumerate(data.split(b"\n")):
            if index:
                if self.pending and not self.discarding and callback:
                    line = self.pending.decode("utf-8", errors="replace")
                    if line.startswith(EVENT_PREFIX):
                        callback(json.loads(line[len(EVENT_PREFIX):]))
                self.pending = b""
                self.discarding = False
            if not self.discarding:
                self.pending += fragment
                if len(self.pending) > self.limit:
                    self.pending = b""
                    self.discarding = True
        return bool(data)


def stop_group(process, sig):
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_workers(jobs, output, stage, on_event=None):
    children = []
    try:
        for worker, command in jobs:
            logfile = output / "logs" / (stage + "_%03d.log" % worker)
            logfile.parent.mkdir(parents=True, exist_ok=True)
            stream = logfile.open("ab")
            tail = None
            try:
                tail = EventTail(logfile)
                process = subprocess.Popen(command, cwd=str(ROOT), env=environment(command[0]),
                                           stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            except BaseException:
                stream.close()
                if tail:
                    tail.stream.close()
                raise
            children.append((process, stream, logfile, tail))
        failures = []
        active = list(children)
        while active:
            progressed = False
            for child in active[:]:
                process, stream, logfile, tail = child
                progressed = tail.read(on_event) or progressed
                rc = process.poll()
                # Read through EOF after exit, including scores written just before exit.
                if rc is not None and tail.stream.tell() == os.fstat(tail.stream.fileno()).st_size:
                    stream.close()
                    tail.stream.close()
                    active.remove(child)
                    if rc:
                        failures.append("%s (exit %d)" % (logfile, rc))
            if active and not progressed:
                time.sleep(0.1)
        if failures:
            raise RuntimeError("Worker failed; completed work is preserved: " + "; ".join(failures))
    finally:
        for process, stream, logfile, tail in children:
            if process.poll() is None:
                stop_group(process, signal.SIGTERM)
        deadline = time.monotonic() + 20
        for process, stream, logfile, tail in children:
            try:
                if process.poll() is None:
                    try:
                        process.wait(timeout=max(0, deadline - time.monotonic()))
                    except subprocess.TimeoutExpired:
                        stop_group(process, signal.SIGKILL)
                        process.wait()
            finally:
                stream.close()
                tail.stream.close()


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


def evaluate(plan, output, no_average=False):
    if plan.get("source_run_root"):
        from .rescore import source_generation
        generated = source_generation(plan, output, ROOT)
    else:
        generated = generation_manifest(plan, output)
    if generated["state"] != "complete":
        raise RuntimeError("Complete generation first; no silent omission of missing videos")
    jobs = []
    for w in plan["workers"]:
        folder = output / "evaluation" / ("%03d" % w["index"])
        assignment = dict(run_root=plan.get("source_run_root", str(output)),
                          entries=[r for r in generated["entries"] if r["clip"] in w["clips"]])
        assignment_path = folder / "assignment.json"
        if assignment_path.exists() and read_json(assignment_path) != assignment:
            raise ValueError("Evaluation inputs changed: " + str(assignment_path))
        write_json(assignment_path, assignment)
        jobs.append((w["index"], [python_for("evaluation"), "-m", "tta_pipeline.evaluation_worker",
                     "--repo", str(ROOT), "--output-dir", str(folder), "--gpu", str(w["gpu"]),
                     "--assignment", str(assignment_path)]))
    printer = ScorePrinter({(c, i) for c in plan["clips"] for i in plan["identities"]})
    try:
        run_workers(jobs, output, "evaluate", on_event=printer.event)
    except BaseException as exc:
        report(output, no_average=no_average, printer=printer,
               worker_error=str(exc) or type(exc).__name__)
        raise
    summary = report(output, no_average=no_average, printer=printer)
    if summary["state"] != "complete":
        raise RuntimeError("Evaluation ended without the complete requested cohort")


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def report(output, no_average=False, printer=None, worker_error=None):
    plan = read_json(output / "run.json")
    rows = []
    failures = []
    for path in sorted((output / "evaluation").glob("*/results.json")):
        result = read_json(path)
        rows.extend(result.get("scores", []))
        if result.get("error") or result.get("state") == "failed":
            failure = dict(result.get("failed_cell") or {})
            failure.setdefault("error", result.get("error") or "Evaluation worker failed")
            failure["worker"] = path.parent.name
            failures.append(failure)
    status_path = output / "status.json"
    if not worker_error and status_path.exists():
        status = read_json(status_path)
        if status.get("state") in ("failed", "interrupted"):
            worker_error = status.get("error") or "Previous run did not finish successfully"
    if worker_error and not failures:
        failures.append(dict(error=worker_error, worker="launcher"))
    expected = {(c, i) for c in plan["clips"] for i in plan["identities"]}
    keys = [(r["clip"], r["identity"]) for r in rows]
    if len(set(keys)) != len(keys) or set(keys) - expected:
        raise ValueError("Duplicate or unexpected official results")
    for row in rows:
        if not valid_score(row):
            raise ValueError("Failed/nonfinite scores cannot enter the aggregate")
        row["lse_d"], row["lse_c"] = float(row["lse_d"]), float(row["lse_c"])
        row["pass_both"] = row["lse_d"] <= 6.5 and row["lse_c"] >= 8.0
        if plan.get("source_run_root") and row.get("video"):
            row["video"] = str(inside(plan["source_run_root"], row["video"]))
    def stats(group):
        return dict(cells=len(group), lse_d=sum(r["lse_d"] for r in group)/len(group),
                    lse_c=sum(r["lse_c"] for r in group)/len(group),
                    pass_both=sum(r["pass_both"] for r in group)) if group else dict(cells=0, lse_d=None,lse_c=None,pass_both=0)
    summary = dict(state="complete" if len(rows) == len(expected) and not failures else "partial",
                   full_fixed_cohort=plan["full_fixed_cohort"], expected_cells=len(expected),
                   checkpoint_step=9000, thresholds=dict(lse_d_max=6.5,lse_c_min=8.0),
                   updated_utc=now(), failures=failures, **stats(rows))
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
    printer = printer or ScorePrinter(expected)
    for row in rows:
        printer.score(row, cached=True, source="saved run")
    for failure in failures:
        printer.failure(failure)
    print_summary(summary, no_average=no_average)
    return summary


def fixed_cohort(repo):
    config = read_json(repo / "configs/evaluation.json")
    entries, identities = config["entries"], config["identities"]
    by_clip = {entry["clip"]: entry for entry in entries}
    languages = {entry["language"] for entry in entries}
    counts = collections.Counter((entry["language"], entry["rate_bin"]) for entry in entries)
    if len(entries) != 90 or len(by_clip) != 90 or len(identities) != 10 or len(set(identities)) != 10:
        raise ValueError("The fixed cohort must contain 90 unique clips and 10 unique identities")
    if (len(languages) != 15 or len(counts) != 45 or set(counts.values()) != {2}
            or {entry["rate_bin"] for entry in entries} != {"slow", "medium", "fast"}
            or ("languages" in config and set(config["languages"]) != languages)):
        raise ValueError("Expected 15 languages x 3 speed bins x 2 recordings")
    return config, by_clip


def reference(no_average=False):
    config, entries = fixed_cohort(ROOT)
    expected = {(clip, identity) for clip in entries for identity in config["identities"]}
    path = ROOT / "reference_results/cell.csv"
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if len(set(columns)) != len(columns):
            raise ValueError("Duplicate reference CSV columns")
        dk, ck = ("D", "C") if "D" in columns and "C" in columns else ("lse_d", "lse_c")
        if not {"clip", "identity", "language", "rate_bin", dk, ck}.issubset(columns):
            raise ValueError("Reference CSV needs clip, identity, language, rate_bin, and D/C columns")
        raw_rows = list(reader)
    rows, seen = [], set()
    for raw in raw_rows:
        key = raw["clip"], raw["identity"]
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate reference cell: %s / %s" % key)
        entry = entries[raw["clip"]]
        if any(raw[k] != entry[k] for k in ("language", "rate_bin")):
            raise ValueError("Reference language/rate metadata differs from the frozen cohort")
        if (raw.get("error") or ("status" in raw and raw["status"] != "ok")
                or ("strict_eligible" in raw and str(raw["strict_eligible"]).lower() not in ("true", "1"))):
            raise ValueError("Failed or ineligible reference score")
        try:
            d, c = float(raw[dk]), float(raw[ck])
        except (TypeError, ValueError, OverflowError):
            raise ValueError("Reference D/C scores must be finite numbers")
        if not math.isfinite(d) or not math.isfinite(c):
            raise ValueError("Reference D/C scores must be finite numbers")
        rows.append(dict(raw, lse_d=d, lse_c=c, status="ok", strict_eligible=True))
        seen.add(key)
    if seen != expected:
        raise ValueError("Incomplete reference cohort: expected all 900 cells, found %d" % len(seen))
    printer = ScorePrinter(expected)
    for row in rows:
        printer.score(row, cached=True, source="stored reference")
    summary = dict(state="complete", full_fixed_cohort=True, cells=len(rows), expected_cells=900,
                   lse_d=sum(r["lse_d"] for r in rows) / len(rows),
                   lse_c=sum(r["lse_c"] for r in rows) / len(rows))
    print_summary(summary, no_average=no_average, source="stored reference; no scoring performed")
    return summary


def list_selections():
    config, entries = fixed_cohort(ROOT)
    print("Available clips (clip ID, language, rate bin):", flush=True)
    for entry in entries.values():
        print("%s\t%s\t%s" % (entry["clip"], entry["language"], entry["rate_bin"]))
    print("\nIdentity IDs:")
    for identity in config["identities"]:
        print(identity)
    print("\nSelection example: --clips %s --identities %s" %
          (config["entries"][0]["clip"], config["identities"][0]), flush=True)


def main():
    parser = argparse.ArgumentParser(description="Frozen ARTalk9000 / AW100 / FlexAvatar official evaluation")
    parser.add_argument("command", choices=["verify","prepare","generate","evaluate","run","rescore","status","report","reference","list"])
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/tta900")
    parser.add_argument("--source", type=Path, help="rescore only: existing run containing generated videos")
    parser.add_argument("--gpus", help="Physical GPU indices, e.g. 0,1; default 0 or saved run settings")
    parser.add_argument("--clips", help="Comma-separated exact manifest clip identifiers")
    parser.add_argument("--identities", help="Comma-separated exact identity identifiers; default all ten")
    parser.add_argument("--limit", type=int, help="Smoke test: first N clips, clearly marked partial cohort")
    parser.add_argument("--quick", action="store_true", help="verify only: check sizes without reading file hashes")
    parser.add_argument("--no-average", action="store_true", help="Print individual scores without terminal means; aggregate files are still written")
    parser.add_argument("--background", action="store_true", help="run only: detach, log and preserve PID")
    args = parser.parse_args()
    if args.limit is not None and not 1 <= args.limit <= 90:
        parser.error("--limit must be between 1 and 90")
    output = args.output.resolve()
    if args.command in ("reference", "list"):
        if args.clips or args.identities or args.limit:
            parser.error("reference and list show the full fixed cohort; selections apply to run/rescore")
        if args.command == "reference":
            reference(no_average=args.no_average)
        else:
            list_selections()
        return
    if args.source and args.command != "rescore":
        parser.error("--source is supported only for rescore")
    if args.command == "rescore" and not args.source:
        parser.error("rescore requires --source pointing to the existing generated run")
    if args.command == "rescore":
        source = args.source.resolve()
        if source == output or source in output.parents or output in source.parents:
            parser.error("rescore --source and --output must be separate, non-nested directories")
        if output == ROOT or output in ROOT.parents:
            parser.error("rescore needs a separate output directory, not the repository root")
        if ((output / "generation.json").exists()
                or ((output / "run.json").exists()
                    and not read_json(output / "run.json").get("source_run_root"))):
            parser.error("rescore needs a new output directory or an existing rescore output")
    if args.command == "verify":
        verify(ROOT, args.quick)
        return
    if args.command in ("status", "report"):
        if not (output / "run.json").exists():
            parser.error("No run found: " + str(output))
        if args.command == "report":
            report(output, no_average=args.no_average)
        else:
            status = read_json(output / "status.json") if (output / "status.json").exists() else {}
            plan = read_json(output / "run.json")
            if plan.get("source_run_root"):
                from .rescore import source_generation
                status["generated_cells"] = len(source_generation(plan, output, ROOT)["entries"])
            else:
                status["generated_cells"] = generation_manifest(plan, output)["completed_cells"]
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
        if args.command == "rescore":
            from .rescore import plan_for_rescore
            plan = plan_for_rescore(args, output, ROOT)
        else:
            plan = plan_for(args, output)
        verify(ROOT)
        stages = (["prepare","generate","evaluate"] if args.command == "run" else
                  ["evaluate"] if args.command == "rescore" else [args.command])
        for stage in stages:
            status.update(stage=stage, updated_utc=now())
            write_json(output / "status.json", status)
            print("Starting " + stage + " at " + now(), flush=True)
            if stage == "evaluate":
                evaluate(plan, output, no_average=args.no_average)
            else:
                globals()[stage](plan, output)
        status.update(state="complete", finished_utc=now())
    except BaseException as exc:
        status.update(state="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc), finished_utc=now())
        raise
    finally:
        write_json(output / "status.json", status)
        lock.close()

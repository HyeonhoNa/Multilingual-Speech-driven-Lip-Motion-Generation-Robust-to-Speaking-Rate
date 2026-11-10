# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Evaluate a frozen generated run without rewriting or copying its artifacts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re

from .generation import CHECKPOINT_SHA256
from .io import inside, read_json, sha256, write_json


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _names(values, label):
    _require(isinstance(values, list) and bool(values), label + " must be a nonempty list")
    _require(all(isinstance(value, str) and value and Path(value).name == value
                 and value not in (".", "..") for value in values), "Unsafe " + label)
    _require(len(values) == len(set(values)), "Duplicate " + label)
    return values


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value) is not None


def _snapshot(root, name):
    path = inside(root, name)
    _require(path.is_file(), "Missing source manifest: " + str(path))
    payload = path.read_bytes()
    result = json.loads(payload)
    _require(isinstance(result, dict), "Expected a JSON object: " + str(path))
    return result, hashlib.sha256(payload).hexdigest()


def _separate(source, output, repo):
    _require(source != output and source not in output.parents and output not in source.parents,
             "Rescore --source and --output must be separate, non-nested directories")
    _require(output != repo and output not in repo.parents,
             "Use a separate evaluation output directory, not the repository root")


def _current_hashes(repo):
    hashes = {path.name: sha256(inside(repo, str(path.relative_to(repo))))
              for path in sorted((repo / "configs").glob("*.json"))}
    runtime = repo / "assets/runtime_checksums.json"
    if runtime.exists():
        hashes["runtime_checksums.json"] = sha256(inside(repo, "assets/runtime_checksums.json"))
    return hashes


def _source(source, repo):
    run, run_hash = _snapshot(source, "run.json")
    generated, generation_hash = _snapshot(source, "generation.json")
    _require(not run.get("source_run_root"), "Select the original generated run as --source")
    _require(type(run.get("checkpoint_step")) is int and run["checkpoint_step"] == 9000,
             "Source run must use checkpoint step 9000")
    clips = _names(run.get("clips"), "source clips")
    identities = _names(run.get("identities"), "source identities")
    old_hashes = run.get("config_sha256")
    _require(isinstance(old_hashes, dict) and all(_digest(value) for value in old_hashes.values()),
             "Source run needs valid saved configuration hashes")
    hashes = _current_hashes(repo)
    for name in ("models.json", "evaluation.json", "identities.json"):
        _require(name in old_hashes and old_hashes[name] == hashes.get(name),
                 "Source configuration differs from current " + name)
    # Generation code may deliberately differ. Preserve its saved hashes in the
    # new run alongside the hashes of the code/configuration used for scoring.
    config = read_json(inside(repo, "configs/evaluation.json"))
    models = read_json(inside(repo, "configs/models.json"))
    _require(type(models.get("checkpoint_step")) is int and models["checkpoint_step"] == 9000,
             "Current models configuration must use checkpoint step 9000")
    checkpoint = models.get("files", {}).get("artalk_step9000", {}).get("sha256")
    _require(checkpoint == CHECKPOINT_SHA256, "Checkpoint SHA256 differs from the frozen step 9000 model")
    config_entries = config.get("entries")
    _require(isinstance(config_entries, list) and all(isinstance(row, dict) for row in config_entries),
             "Invalid current evaluation entries")
    all_clips = _names([row.get("clip") for row in config_entries], "configured clips")
    all_ids = _names(config.get("identities"), "configured identities")
    _require(not set(clips) - set(all_clips) and not set(identities) - set(all_ids),
             "Source scope contains identifiers outside the frozen evaluation cohort")
    metadata = {row["clip"]: row for row in config_entries}
    expected = {(clip, identity) for clip in clips for identity in identities}
    _require(type(run.get("expected_cells")) is int and run["expected_cells"] == len(expected),
             "Source run expected cell count differs from its scope")
    _require(generated.get("state") == "complete", "Source generation must be complete")
    for field in ("expected_cells", "completed_cells"):
        _require(type(generated.get(field)) is int and generated[field] == len(expected),
                 "Source generation " + field + " differs from its complete scope")
    rows = generated.get("entries")
    _require(isinstance(rows, list), "Source generation entries must be a list")
    seen, videos = set(), set()
    for row in rows:
        _require(isinstance(row, dict), "Invalid source generation entry")
        clip, identity = row.get("clip"), row.get("identity")
        _require(isinstance(clip, str) and isinstance(identity, str), "Missing source cell identifiers")
        key = clip, identity
        _require(key in expected and key not in seen, "Unexpected or duplicate source generated cell: %s / %s" % key)
        seen.add(key)
        _require(row.get("checkpoint_sha256") == checkpoint, "Source cell checkpoint SHA256 differs: %s / %s" % key)
        entry = metadata[clip]
        _require(type(row.get("frames")) is int and row["frames"] > 5
                 and row["frames"] == entry.get("frames"), "Source cell frame count differs: " + clip)
        _require(_digest(row.get("pcm_sha256")) and row["pcm_sha256"] == entry.get("pcm_sha256"),
                 "Source cell canonical audio differs: " + clip)
        for field in ("language", "rate_bin"):
            if field in row:
                _require(row[field] == entry.get(field), "Source cell " + field + " differs: " + clip)
        audio = entry.get("score_audio", entry.get("audio"))
        if "audio_sha256" in row and isinstance(audio, dict) and audio.get("sha256"):
            _require(row["audio_sha256"] == audio["sha256"], "Source cell audio SHA256 differs: " + clip)
        relative = row.get("video")
        _require(isinstance(relative, str) and relative and not Path(relative).is_absolute(),
                 "Source video must have a relative path")
        video = inside(source, relative)
        _require(video.is_file(), "Missing source video: " + str(video))
        _require(video not in videos, "Duplicate source video path: " + str(video))
        videos.add(video)
        _require(_digest(row.get("video_sha256")), "Missing or invalid source video SHA256: " + str(video))
    _require(seen == expected, "Source generation is missing requested cells")
    # Workers verify every selected video's digest immediately before scoring,
    # including videos whose saved scores are resumed. No source file is written.
    _require(sha256(inside(source, "run.json")) == run_hash
             and sha256(inside(source, "generation.json")) == generation_hash,
             "Source manifests changed during validation")
    return run, generated, config, hashes, run_hash, generation_hash


def _selection(value, available, label):
    values = _names(value, label)
    _require(not set(values) - set(available), "Selected " + label + " are absent from the source run")
    return [item for item in available if item in values]


def _plan(source, data, clips, identities, gpus):
    run, generated, config, hashes, run_hash, generation_hash = data
    clips = _selection(clips, run["clips"], "clips")
    identities = _selection(identities, run["identities"], "identities")
    _require(isinstance(gpus, list) and gpus and all(type(gpu) is int and gpu >= 0 for gpu in gpus)
             and len(gpus) == len(set(gpus)), "Supply distinct nonnegative GPU indices")
    all_clips = [entry["clip"] for entry in config["entries"]]
    return dict(schema=1, mode="rescore", checkpoint_step=9000,
                checkpoint_sha256=CHECKPOINT_SHA256, source_run_root=str(source),
                source_run_sha256=run_hash, source_generation_sha256=generation_hash,
                source_config_sha256=run["config_sha256"], config_sha256=hashes,
                clips=clips, identities=identities, gpus=gpus,
                expected_cells=len(clips) * len(identities),
                full_fixed_cohort=len(clips) == 90 and len(identities) == 10
                and set(clips) == set(all_clips) and set(identities) == set(config["identities"]),
                workers=[dict(index=index, gpu=gpu, clips=clips[index::len(gpus)])
                         for index, gpu in enumerate(gpus) if clips[index::len(gpus)]])


def plan_for_rescore(args, output, repo):
    """Create/resume evaluation-only output, preserving the source provenance."""
    _require(getattr(args, "source", None) is not None, "rescore requires --source")
    source, output, repo = Path(args.source).resolve(), Path(output).resolve(), Path(repo).resolve()
    _separate(source, output, repo)
    _require(getattr(args, "limit", None) is None, "rescore supports --clips and --identities; --limit is not supported")
    _require(not os.environ.get("CUDA_VISIBLE_DEVICES"), "Unset CUDA_VISIBLE_DEVICES; select GPUs using --gpus")
    _require(not (output / "generation.json").exists(), "Use a new evaluation output, not an existing generated run")
    path = inside(output, "run.json")
    old = read_json(path) if path.exists() else None
    _require(old is None or isinstance(old, dict) and old.get("mode") == "rescore",
             "Output belongs to another run; use a new evaluation output")
    data = _source(source, repo)
    run = data[0]
    selections = {}
    for name in ("clips", "identities"):
        value = getattr(args, name, None)
        selections[name] = ([item.strip() for item in value.split(",") if item.strip()]
                            if value is not None else old.get(name) if old else run[name])
    value = getattr(args, "gpus", None)
    try:
        gpus = ([int(item.strip()) for item in value.split(",") if item.strip()]
                if value is not None else old.get("gpus") if old else [0])
    except ValueError:
        raise ValueError("Supply distinct nonnegative GPU indices")
    plan = _plan(source, data, selections["clips"], selections["identities"], gpus)
    if old is not None:
        _require(all(old.get(key) == value for key, value in plan.items()),
                 "Rescore configuration or source changed. Use the saved settings or a new --output")
        return old
    plan["created_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    write_json(path, plan)
    return plan


def source_generation(plan, output, repo):
    """Revalidate frozen inputs and return the selected, source-relative manifest."""
    _require(isinstance(plan, dict) and plan.get("mode") == "rescore" and plan.get("source_run_root"),
             "Expected a saved rescore plan")
    source, output, repo = Path(plan["source_run_root"]).resolve(), Path(output).resolve(), Path(repo).resolve()
    _separate(source, output, repo)
    data = _source(source, repo)
    current = _plan(source, data, plan.get("clips"), plan.get("identities"), plan.get("gpus"))
    _require(all(plan.get(key) == value for key, value in current.items()),
             "Rescore configuration or source changed; frozen inputs must remain unchanged")
    rows = [dict(row) for row in data[1]["entries"]
            if row["clip"] in plan["clips"] and row["identity"] in plan["identities"]]
    return dict(state="complete", entries=rows, expected_cells=plan["expected_cells"], completed_cells=len(rows))

"""Unmodified, pinned official CLI scoring with explicit media-extent audits.

This is an external orchestrator, not an implementation of the LSE formula.
Callers own GPU scheduling. Only this helper's newly allocated /tmp directory
is cleaned; inputs, pristine code, cached weights and historical runs are kept.

Ported from lse_gap_official_score_v1.py by changing only path/provenance bindings.
The metadata-only extent gate allows the official scorer's min(video, audio)
extent to exclude conversion-added timeline frames, never native content.
Neither historical receipts nor the pinned official CLI are modified.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import tempfile
import time
import wave

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO / 'vendor/official_syncnet'
MANIFEST = REPO / 'assets/official_syncnet_manifest.json'
MANIFEST_SHA = 'fdab2eb843c286a890be8bcede2c01ce3eb85f379cd38edbee1b4f4335046f23'
PYTHON = REPO / 'envs/evaluation/bin/python'
FFMPEG = REPO / 'media/bin/ffmpeg'
FFPROBE = REPO / 'media/bin/ffprobe'
PREFIX = 'starlab-lse-gap-official-v1-'
SAFE_NAME = re.compile(r'[A-Za-z0-9_-]{1,120}')
NUMBER = r'[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?'
PAIR = re.compile(r'\s*(' + NUMBER + r')\s+(' + NUMBER + r')\s*')


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def pin(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return dict(path=str(path), resolved_path=str(path.resolve()),
                size=path.stat().st_size, sha256=h.hexdigest())


def _save(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def parse_scores(text):
    """Parse CLI output, never recompute or select the highest-scoring track."""
    pairs = []
    for line in text.splitlines():
        match = PAIR.fullmatch(line)
        if match:
            d, c = map(float, match.groups())
            require(math.isfinite(d) and math.isfinite(c), 'Nonfinite official score')
            pairs.append(dict(lse_d=d, lse_c=c))
        elif re.fullmatch(r'\s*(?:nan|[-+]?inf)\s+\S+\s*', line, re.I):
            raise RuntimeError('Nonfinite official score')
    require(bool(pairs), 'Official CLI produced no score')
    return pairs


def verify_pristine():
    """Verify unchanged official source, weights and media binaries in this repo."""
    require(pin(MANIFEST)['sha256'] == MANIFEST_SHA, 'Pinned official manifest changed')
    plan = json.loads(MANIFEST.read_text())
    for record in plan['sources'] + plan['assets']:
        current = pin(REPO / record['path'])
        require(current['sha256'] == record['sha256'] and current['size'] == record['size'],
                'Pinned official asset changed: ' + record['path'])
    wanted = {REPO / p['path'] for p in plan['sources']
              if Path(p['path']).suffix in {'.py', '.sh'}}
    found = {p for p in RUNTIME.rglob('*') if p.suffix in {'.py', '.sh'}}
    require(found == wanted, 'Unexpected or missing official executable source')
    media = json.loads((REPO / 'media/manifest.json').read_text())
    for record in media['files']:
        require(pin(REPO / record['path'])['sha256'] == record['sha256'],
                'Pinned FFmpeg runtime changed: ' + record['path'])
    return dict(manifest=pin(MANIFEST), commits=plan['commits'], sources=plan['sources'],
        weights=plan['assets'], source_verified=True,
        weight_origin_verified=all(a['fresh_official_download'] for a in plan['assets']),
        weight_origin_note='Cached weight hashes are pinned; fresh upstream origin was NOT independently verified.',
        scoring_helper=pin(__file__), interpreter=pin(PYTHON),
        ffmpeg=pin(REPO / 'media/libexec/ffmpeg'), ffprobe=pin(REPO / 'media/libexec/ffprobe'),
        expected_environment=plan['environment'])


def _run(command, log, env, timeout):
    """Own a child process group; reap ffmpeg descendants even on interruption."""
    with Path(log).open('x', encoding='utf-8') as stream:
        stream.write('COMMAND ' + json.dumps([str(x) for x in command]) + '\n')
        stream.flush()
        process = subprocess.Popen([str(x) for x in command], cwd=RUNTIME, env=env,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        finally:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            if process.poll() is None:
                process.wait(timeout=5)
    require(code == 0, 'Official command failed: ' + str(log))
    return Path(log).read_text(errors='replace')


def _probe(path, env):
    return json.loads(subprocess.check_output([str(FFPROBE), '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(path)], env=env, timeout=90))


def _environment(env, expected):
    code = ('import importlib.metadata as m,json,sys; '
            'print(json.dumps(dict(python=sys.version,executable=sys.executable,'
            'packages={n:m.version(n) for n in json.loads(sys.argv[1])})))')
    value = json.loads(subprocess.check_output([str(PYTHON), '-B', '-c', code,
        json.dumps(list(expected['packages']))], env=env, cwd=RUNTIME, timeout=60))
    require(value['packages'] == expected['packages'], 'Official package versions differ from pinned environment')
    value['ffmpeg'] = subprocess.check_output([str(FFMPEG), '-version'], env=env,
                                             text=True, timeout=30).splitlines()[0]
    require(value['ffmpeg'] == expected['ffmpeg'], 'Official ffmpeg version changed')
    return value


def _video_stream(meta):
    streams = [s for s in meta['streams'] if s['codec_type'] == 'video']
    require(len(streams) == 1, 'Expected exactly one video stream')
    return streams[0]


def _wav(path):
    with wave.open(str(path), 'rb') as stream:
        require(stream.getnchannels() == 1 and stream.getframerate() == 16000
                and stream.getsampwidth() == 2 and stream.getcomptype() == 'NONE',
                'Official audio is not mono16k PCM16')
        pcm = stream.readframes(stream.getnframes())
        return dict(samples=stream.getnframes(), pcm_sha256=hashlib.sha256(pcm).hexdigest(), file=pin(path))


def audit_extent(*, native_frames, converted_actual_frames, converted_header_frames,
                 jpeg_names, track_frames, scene_ranges, native_audio_samples,
                 converted_audio_samples, scored_audio_samples, scored_video_frames):
    """Audit the official extent without trimming, padding, or rescoring media.

    The extracted track and cropped video must cover the complete converted
    timeline. Official evaluation can use fewer frames because its own audio
    length floor excludes a fractional conversion-added tail. This is valid
    only when its effective extent still covers every native frame.
    """
    for value in (native_frames, converted_actual_frames, converted_header_frames,
                  native_audio_samples, converted_audio_samples, scored_audio_samples, scored_video_frames):
        require(type(value) is int and value > 0, 'Positive integer media extents required')
    count = len(jpeg_names)
    require(native_frames > 5 and native_audio_samples == native_frames * 640,
            'Native PCM does not cover exactly the native frames')
    require(converted_actual_frames == native_frames, 'Official conversion dropped or added decoded content frames')
    require(converted_header_frames >= native_frames and count == converted_header_frames,
            'Extracted frame list differs from the official converted timeline')
    require(list(jpeg_names) == ['%06d.jpg' % i for i in range(1, count + 1)],
            'Missing, duplicated or misnumbered extracted JPEG')
    require(len(track_frames) == 1 and list(track_frames[0]) == list(range(count)),
            'Expected one face track covering the entire extracted official timeline')
    cursor = 0
    require(bool(scene_ranges), 'Missing official scene ranges')
    for start, end in scene_ranges:
        require(type(start) is int and type(end) is int and start == cursor and end > start,
                'Official scenes have missing/overlapping content')
        cursor = end
    require(cursor == count, 'Scene duration and extracted official timeline differ')
    require(converted_audio_samples >= native_audio_samples and scored_audio_samples >= native_audio_samples,
            'Official audio processing lost native audio extent')
    require(scored_video_frames == count,
            'Scored video does not cover the complete official track')
    effective = min(scored_video_frames, scored_audio_samples // 640)
    require(native_frames <= effective <= count,
            'Official effective scoring extent does not cover native content within the full track')
    return dict(single_complete=True, canonical_frames=native_frames,
        converted_actual_frames=converted_actual_frames, converted_header_frames=converted_header_frames,
        extracted_jpegs=count, extracted_extra_frames=count-native_frames,
        timing_expansion_preserved=True, silent_trim_or_padding=False,
        track_frames=track_frames, scene_ranges=scene_ranges,
        native_audio_samples=native_audio_samples, converted_audio_samples=converted_audio_samples,
        scored_audio_samples=scored_audio_samples, scored_video_frames=scored_video_frames,
        effective_evaluation_frames=effective, official_embedding_windows=effective-5,
        effective_extent_rule='min(scored_video_frames, scored_audio_samples // 640)',
        full_official_track_frames=count, native_extent_covered=True,
        evaluated_extra_timeline_frames=effective-native_frames,
        official_timeline_frames_outside_effective_extent=count-effective,
        scored_audio_subframe_samples=scored_audio_samples % 640,
        extent_gate_version='native_coverage_v1_metadata_only')


def _artifact_data(work, ref, env):
    # Pickles are trusted outputs created by our official subprocess. Load them
    # in its own environment (NumPy2/SceneDetect), not the fitting environment.
    code = """import json,pickle,sys
from pathlib import Path
p=Path(sys.argv[1]); ref=sys.argv[2]
with (p/'pywork'/ref/'tracks.pckl').open('rb') as f: tracks=pickle.load(f)
with (p/'pywork'/ref/'scene.pckl').open('rb') as f: scenes=pickle.load(f)
print(json.dumps({'track_frames':[[int(i) for i in t['track']['frame']] for t in tracks],
 'scene_ranges':[[int(a.get_frames()),int(b.get_frames())] for a,b in scenes]}))
"""
    return json.loads(subprocess.check_output([str(PYTHON), '-B', '-c', code, str(work), ref],
                      env=env, cwd=RUNTIME, timeout=60))


def _inventory(work):
    require(work.parent == Path('/tmp') and work.name.startswith(PREFIX) and work.resolve() == work,
            'Cleanup inventory is outside the uniquely owned scratch directory')
    records = []
    for path in sorted(work.rglob('*')):
        require(not path.is_symlink(), 'Unexpected symlink in owned scratch')
        if path.is_file():
            record = pin(path)
            record['allocated_bytes'] = path.stat().st_blocks * 512
            records.append(record)
    return records


def score_video(video, entry, condition, root, env=None, *, timeout_s=900):
    """Return a durable score receipt; failures retain raw scores but are ineligible.

    entry requires clip, frames and canonical pcm_sha256. GPU ownership belongs
    to the caller. Repeated (condition,clip) outputs are rejected, not replaced.
    """
    root, video = Path(root), Path(video)
    root, video = root.resolve(), video.resolve()
    require(root != REPO and not root.is_relative_to(RUNTIME),
            'A separate explicit output directory is required')
    require(SAFE_NAME.fullmatch(str(entry['clip'])) and SAFE_NAME.fullmatch(str(condition)), 'Unsafe output key')
    require(type(entry['frames']) is int and entry['frames'] > 5, 'Native frame count required')
    require(re.fullmatch(r'[a-f0-9]{64}', entry['pcm_sha256']), 'Canonical PCM SHA256 required')
    require(video.is_file(), 'Expected an existing generated video')
    ref = 'lsegap__' + condition + '__' + entry['clip']
    logs = root / 'lse_logs' / ref
    logs.mkdir(parents=True, exist_ok=False)
    row = dict(clip=entry['clip'], condition=condition, frames=entry['frames'], video=str(video),
        language=entry.get('language', entry.get('config', 'unknown')), status='failed', strict_eligible=False,
        lse_d=None, lse_c=None, raw_track_scores=[], final_test=True, error=None, offset=None,
        offset_note='The unmodified Wav2Lip official CLI prints D/C only; no offset is recomputed.',
        official_math='unchanged pinned run_pipeline.py + Wav2Lip calculate_scores_real_videos.py',
        extent_gate_version='native_coverage_v1_metadata_only')
    started = time.monotonic()
    work = None
    before = pin(video)
    row['input'] = before
    try:
        provenance = verify_pristine()
        row['provenance'] = provenance
        run_env = dict(os.environ if env is None else env)
        run_env.pop('PYTHONPATH', None)
        run_env.pop('PYTHONHOME', None)
        run_env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
        run_env['PATH'] = str(FFMPEG.parent) + os.pathsep + run_env.get('PATH', '')
        row['runtime_environment'] = _environment(run_env, provenance['expected_environment'])
        native = _probe(video, run_env)
        vs = _video_stream(native)
        aud = [s for s in native['streams'] if s['codec_type'] == 'audio']
        require(int(vs['nb_read_frames']) == entry['frames'] and vs['r_frame_rate'] == '25/1'
                and (vs['width'], vs['height']) == (512, 512), 'Native video extent/fps/size mismatch')
        require(len(aud) == 1 and aud[0]['codec_name'] == 'pcm_s16le'
                and int(aud[0]['sample_rate']) == 16000 and aud[0]['channels'] == 1,
                'Native audio must be mono16k PCM16')
        pcm = subprocess.check_output([str(FFMPEG), '-v', 'error', '-i', str(video), '-map', '0:a:0',
            '-f', 's16le', '-acodec', 'pcm_s16le', '-'], env=run_env, timeout=90)
        require(len(pcm) == entry['frames'] * 640 * 2 and hashlib.sha256(pcm).hexdigest() == entry['pcm_sha256'],
                'Native audio differs from pinned canonical PCM')
        row['video_receipt'] = dict(input=before, probe=native, pcm_sha256=entry['pcm_sha256'],
                                    samples=len(pcm)//2)
        del pcm
        with tempfile.TemporaryDirectory(prefix=PREFIX, dir='/tmp') as name:
            work = Path(name)
            try:
                common = ['--videofile', str(video), '--reference', ref, '--data_dir', str(work)]
                _run([PYTHON, '-X', 'utf8', '-B', 'run_pipeline.py'] + common,
                     logs/'pipeline.log', run_env, timeout_s)
                output = _run([PYTHON, '-X', 'utf8', '-B', 'calculate_scores_real_videos.py'] + common,
                              logs/'score.log', run_env, timeout_s)
                row['raw_track_scores'] = parse_scores(output)
                require(len(row['raw_track_scores']) == 1, 'Expected one official score, no best-track selection')
                artifacts = _artifact_data(work, ref, run_env)
                converted = _probe(work/'pyavi'/ref/'video.avi', run_env)
                cropped = _probe(work/'pycrop'/ref/'00000.avi', run_env)
                converted_audio = _wav(work/'pyavi'/ref/'audio.wav')
                scored_audio = _wav(work/'pytmp'/ref/'audio.wav')
                jpegs = sorted((work/'pyframes'/ref).glob('*.jpg'))
                jpeg_records = [pin(p) for p in jpegs]
                _save(logs/'media_probe.json', dict(native=native, converted=converted, cropped=cropped,
                    converted_audio=converted_audio, scored_audio=scored_audio, artifacts=artifacts,
                    extracted_jpegs=jpeg_records,
                    identical_adjacent_jpeg_indices=[i for i in range(1, len(jpeg_records))
                        if jpeg_records[i]['sha256'] == jpeg_records[i-1]['sha256']]))
                cv, cr = _video_stream(converted), _video_stream(cropped)
                require(cv['r_frame_rate'] == cr['r_frame_rate'] == '25/1', 'Official video fps mismatch')
                row['track_audit'] = audit_extent(native_frames=entry['frames'],
                    converted_actual_frames=int(cv['nb_read_frames']), converted_header_frames=int(cv['nb_frames']),
                    jpeg_names=[p.name for p in jpegs], track_frames=artifacts['track_frames'],
                    scene_ranges=artifacts['scene_ranges'], native_audio_samples=entry['frames']*640,
                    converted_audio_samples=converted_audio['samples'], scored_audio_samples=scored_audio['samples'],
                    scored_video_frames=int(cr['nb_read_frames']))
                require(pin(video) == before, 'Input video changed during scoring')
                require(verify_pristine() == provenance, 'Pinned scoring dependencies changed during scoring')
                row.update(row['raw_track_scores'][0])
                row.update(status='ok', strict_eligible=True)
            finally:
                # Preserve all available trusted diagnostic artifacts even if a
                # later validity gate fails; then remove only this tempdir.
                preserved = []
                for basename in ('tracks.pckl', 'scene.pckl', 'faces.pckl'):
                    source = work/'pywork'/ref/basename
                    if source.is_file():
                        require(not (logs/basename).exists(), 'Artifact overwrite forbidden')
                        shutil.copy2(source, logs/basename)
                        preserved.append(pin(logs/basename))
                records = _inventory(work)
                row['scratch'] = dict(path=str(work), files=len(records),
                    bytes=sum(r['allocated_bytes'] for r in records), deleted_files=records,
                    preserved=preserved, cleanup_completed=False,
                    recovery='Re-run unchanged official preprocessing from the retained input video; logs/tracks/probes are kept.')
    except Exception as error:
        row.update(status='failed', strict_eligible=False, error=type(error).__name__ + ': ' + str(error))
    finally:
        row['input_unchanged'] = video.is_file() and pin(video) == before
        if not row['input_unchanged']:
            row.update(status='failed', strict_eligible=False, error='Input changed during scoring')
        if work is not None and 'scratch' in row:
            row['scratch']['cleanup_completed'] = not work.exists()
        row['seconds'] = time.monotonic() - started
        _save(logs/'receipt.json', row)
    return row


def evaluate_video(video, entry, output_dir, gpu=0, repo_root=None, timeout_s=900):
    """Evaluate one full generated AVI using the exact final-result official CLI.

    No fitting is performed. Identity-specific D/C is never pooled with another
    identity inside SyncNet. The caller owns worker/GPU assignment and aggregation.
    """
    if repo_root is not None:
        require(Path(repo_root).resolve() == REPO, 'Evaluator must be imported from requested repo')
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = str(gpu)
    env.setdefault('OMP_NUM_THREADS', '1')
    env.setdefault('MKL_NUM_THREADS', '1')
    row = score_video(Path(video).resolve(), entry, 'official',
                      Path(output_dir).resolve(), env, timeout_s=timeout_s)
    row['identity'] = entry.get('identity')
    return row


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=REPO)
    parser.add_argument('--video', type=Path, required=True)
    parser.add_argument('--clip', required=True)
    parser.add_argument('--identity', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    entries = json.loads((args.repo / 'configs/evaluation.json').read_text())['entries']
    matches = [e for e in entries if e['clip'] == args.clip]
    require(len(matches) == 1, 'Clip must match one fixed evaluation entry')
    entry = dict(matches[0], identity=args.identity)
    row = evaluate_video(args.video, entry, args.output_dir, args.gpu, args.repo)
    print(json.dumps(row, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if row['status'] == 'ok' else 1)


if __name__ == '__main__':
    main()

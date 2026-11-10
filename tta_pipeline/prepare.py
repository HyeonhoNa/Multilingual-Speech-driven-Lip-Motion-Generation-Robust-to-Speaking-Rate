# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""Fresh DeepSpeech -> frozen TOG SRNet/window controls for selected audio."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from ._runtime.io import identifier, model_paths, read, resolve, save, select_entries, sha256


def deepspeech_worker(repo, output, entries):
    import numpy as np
    from ._runtime.deepspeech import DeepSpeechCPU
    model = None
    graph = model_paths(repo)['tog_deepspeech_graph']
    for entry in entries:
        clip = identifier(entry['clip'])
        audio = resolve(repo, entry['original_audio'])
        target = output / 'prepared/deepspeech' / (clip + '.npy')
        receipt = target.with_suffix('.json')
        source_hash = sha256(audio)
        if target.exists() and receipt.exists():
            old = read(receipt)
            if old['source_sha256'] == source_hash and old['feature_sha256'] == sha256(target):
                continue
            raise ValueError('Changed prepared DeepSpeech asset: ' + clip)
        if model is None:
            model = DeepSpeechCPU(graph, threads=2)
        begin = time.monotonic()
        features, info = model.extract(audio)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.save(stream, features, allow_pickle=False)
        os.replace(temporary, target)
        save(receipt, dict(clip=clip, source_sha256=source_hash, feature_sha256=sha256(target),
                          elapsed_seconds=time.monotonic()-begin, **info))
        print('DeepSpeech: ' + clip, flush=True)


def prepare(repo, output, entries, clips):
    output.mkdir(parents=True, exist_ok=True)
    scope = dict(models_sha256=sha256(repo/'configs/models.json'),
                 evaluation_sha256=sha256(repo/'configs/evaluation.json'),
                 clips=[entry['clip'] for entry in entries])
    scope_path = output/'prepared/scope.json'
    if scope_path.exists() and read(scope_path) != scope:
        raise ValueError('Preparation scope changed; use a fresh output directory')
    save(scope_path, scope)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=str(repo) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    command = [str(repo/'envs/deepspeech/bin/python'), '-m', 'tta_pipeline.prepare', '--repo', str(repo),
               '--output-dir', str(output), '--deepspeech-worker']
    if clips:
        command += ['--clips', clips]
    subprocess.run(command, check=True, env=env, cwd=repo)
    import numpy as np
    import torch
    from ._runtime import rate
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    paths = model_paths(repo)
    device = torch.device('cpu')
    sr, predictor = rate.load_models(paths['tog_srnet'], paths['tog_window_generator'], device)
    sr.requires_grad_(False)
    predictor.requires_grad_(False)
    sm, ss = rate.load_stats(paths['tog_srnet_mean_std'])
    gm, gs = rate.load_stats(paths['tog_generator_mean_std'])
    rows = []
    for entry in entries:
        clip = identifier(entry['clip'])
        ds = output/'prepared/deepspeech'/(clip+'.npy')
        target = output/'prepared/rates'/(clip+'.npz')
        receipt = target.with_suffix('.json')
        source_hash = sha256(resolve(repo, entry['original_audio']))
        if target.exists() and receipt.exists():
            old = read(receipt)
            if old['source_sha256'] != source_hash or old['rate_sha256'] != sha256(target):
                raise ValueError('Changed prepared rate asset: ' + clip)
            rows.append(old)
            continue
        begin = time.monotonic()
        features = np.asarray(np.load(ds, allow_pickle=False), dtype=np.float32)
        pps60 = rate.batched_pps(sr, rate.edge_windows((features-sm)/(ss+1e-6), 60), device, 64)
        windows60 = rate.batched_windows(predictor, rate.edge_windows((features-gm)/(gs+1e-6), 60),
                                         rate.edge_windows(pps60[:, None], 60), device, 64)
        pps = rate.resample_sequence(pps60, entry['frames'], 60, 25)
        windows = rate.resample_sequence(windows60, entry['frames'], 60, 25)
        ratio = np.clip(windows/40, .5, 1.5).astype(np.float32)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.tmp')
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, pps=pps, window_size=windows, window_ratio=ratio)
        os.replace(temporary, target)
        row = dict(clip=clip, source_sha256=source_hash, rate_sha256=sha256(target),
                   rate=str(target.relative_to(output)), frames=entry['frames'],
                   elapsed_seconds=time.monotonic()-begin)
        save(receipt, row)
        rows.append(row)
        print('SRNet/window: ' + clip, flush=True)
    save(output/'prepared/manifest.json', dict(state='complete', entries=rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--clips')
    parser.add_argument('--deepspeech-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    repo, output = args.repo.resolve(), args.output_dir.resolve()
    entries = select_entries(repo, args.clips)
    if args.deepspeech_worker:
        deepspeech_worker(repo, output, entries)
    else:
        prepare(repo, output, entries, args.clips)


if __name__ == '__main__':
    main()

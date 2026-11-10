"""Project AW100 + ARTalk step9000 + FLEX-1 inference orchestration.

Connects the project rate/pooling, chunk rollout, and renderer adapters derived
from experiments/neural_renderer_lse. ARTalk and FlexAvatar implementations
are imported from separately installed third_party sources; checkpoints and
source assets are loaded from explicit local paths. This file handles project
parameter selection, deterministic settings, validation, and output receipts.
"""
from __future__ import annotations
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time
import wave

from ._runtime.io import identifier, model_paths, read, resolve, save, select_entries, sha256

CHECKPOINT_SHA256 = '1dda0c9f2cb71152819050a200f94412bc39b78a999eb9023d66a082e165e5dc'
BASE_SHA256 = 'd89615a32e2a6b4e35cf5494c1e400b2acedc43812f63423b9f5d705046d67c2'
TARGET_SHA256 = '019f5a7d64799f6c4b196aef5466356f5d8fa1dc60eff5b0771a02eb8f7ec24a'


def configure(repo, output, gpu, paths):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    wav_config = paths['wav2vec2_config']
    mapping = dict(TTA_WAV2VEC_CONFIG=wav_config.parent if wav_config.is_file() else wav_config,
        TTA_DINO_SOURCE=repo/'third_party/dinov2', TTA_DINO_WEIGHT=paths['dino_weights'],
        TTA_FLEX_ASSETS=paths['flex_template'].parent,
        FLEXAVATAR_MODELS_PATH=paths['flex_model_config'].parent.parent,
        FLEXAVATAR_INPUTS_PATH=repo/'data/identities', FLEXAVATAR_PIXEL3DMM_PROCESSING_PATH=repo/'data/identities',
        FLEXAVATAR_AVATAR_CODE_PATH=output/'avatars', FLEXAVATAR_RENDERINGS_PATH=output,
        FLEXAVATAR_EVALUATIONS_PATH=output/'evaluations',
        TORCH_HOME=output/'cache/torch', HF_HOME=output/'cache/huggingface', XDG_CACHE_HOME=output/'cache',
        MPLCONFIGDIR=output/'cache/matplotlib', TORCH_EXTENSIONS_DIR=output/'cache/torch_extensions',
        CUDA_HOME=repo/'envs/generation',
        HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
    os.environ.update({key: str(value) for key, value in mapping.items()})
    os.environ['PATH'] = os.pathsep.join((str(repo/'media/bin'), str(repo/'envs/generation/bin'), os.environ['PATH']))
    for relative in ('third_party/ARTalk', 'third_party/FlexAvatar/src',
                     'third_party/FlexAvatar_MODNet/src', 'third_party/FlexAvatar_stylegan_generators/src'):
        sys.path.insert(0, str(repo/relative))


def tensor_hash(values):
    digest = hashlib.sha256()
    for name, value in sorted(values.items()):
        cpu = value.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(cpu.dtype).encode())
        digest.update(str(tuple(cpu.shape)).encode())
        import torch
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_model(paths):
    import torch
    from app import BitwiseARModel
    if sha256(paths['artalk_step9000']) != CHECKPOINT_SHA256 or sha256(paths['artalk_base']) != BASE_SHA256:
        raise ValueError('Final checkpoint or released ARTalk SHA256 differs')
    torch.set_num_threads(2)
    torch.manual_seed(20261007)
    torch.cuda.manual_seed_all(20261007)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = read(paths['artalk_config'])
    config['AR_CONFIG']['AUDIO_ENCODER'] = 'wav2vec'
    model = BitwiseARModel(config).eval().requires_grad_(False)
    model.load_state_dict(torch.load(paths['artalk_base'], map_location='cpu', weights_only=True), strict=True)
    model = model.cuda()
    style = torch.load(paths['artalk_style'], map_location='cpu', weights_only=True)[None].cuda()
    with torch.no_grad():
        style_cond = model.style_cond_embed(model.style_encoder(style))[:, None]*1.1-model.null_style_cond*.1
    payload = torch.load(paths['artalk_step9000'], map_location='cpu', weights_only=True, mmap=True)
    targets = payload['trainable']
    if tensor_hash(targets) != TARGET_SHA256:
        raise ValueError('Final trained parameter hash differs')
    parameters = dict(model.named_parameters())
    expected = {f'attn_blocks.{i}.ada_lin.1.{kind}' for i in range(12) for kind in ('weight','bias')}
    expected |= {'cond_logits_head.ada_lin.1.weight', 'cond_logits_head.ada_lin.1.bias'}
    expected |= {f'attn_blocks.{i}.{suffix}.{kind}' for i in range(4,12)
                 for suffix in ('attn.query','attn.key','attn.value','attn.proj','ffn.0','ffn.2')
                 for kind in ('weight','bias') if f'attn_blocks.{i}.{suffix}.{kind}' in parameters}
    expected |= {f'attn_blocks.{i}.attn.scale_mul_1H11' for i in range(4,12)
                 if f'attn_blocks.{i}.attn.scale_mul_1H11' in parameters}
    if set(targets) != expected or len(targets) != 122:
        raise ValueError('Checkpoint target topology differs')
    with torch.no_grad():
        for name, value in targets.items():
            if value.shape != parameters[name].shape or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid trained parameter: ' + name)
            parameters[name].copy_(value)
    if tensor_hash({name: parameters[name] for name in targets}) != TARGET_SHA256:
        raise ValueError('Loaded model parameter hash differs')
    # FullAdaLNLinear used in training is exactly F.linear; native nn.Linear
    # uses identical weights/bias and forward expression, with no initial buffers.
    del payload
    return model.eval().requires_grad_(False), style_cond


def generate_motion(model, style, repo, output, entry):
    import numpy as np
    import torch
    from ._runtime.pooling import mixed_multiscale_audio_condition
    from ._runtime.rollout import rollout_precomputed
    from ._runtime.motion import postprocess_motion
    audio_path = resolve(repo, entry['original_audio'])
    with wave.open(str(audio_path), 'rb') as stream:
        if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate()) != (1,2,16000):
            raise ValueError('Expected native mono PCM16 16kHz audio')
        samples = stream.getnframes()
        raw_pcm = stream.readframes(samples)
    audio = torch.from_numpy(np.frombuffer(raw_pcm, dtype='<i2').copy()).float().cuda()/32768.
    frames = entry['frames']
    padded_frames = ((frames+99)//100)*100
    audio = torch.nn.functional.pad(audio, (0, padded_frames*640-len(audio)))
    rate_path = output/'prepared/rates'/(entry['clip']+'.npz')
    receipt = read(rate_path.with_suffix('.json'))
    if receipt['source_sha256'] != sha256(audio_path) or receipt['rate_sha256'] != sha256(rate_path):
        raise ValueError('Stale adaptive-window preparation')
    ratio = torch.from_numpy(np.load(rate_path, allow_pickle=False)['window_ratio']).cuda()
    if len(ratio) != frames:
        raise ValueError('Adaptive-window frame count differs')
    held = torch.nn.functional.pad(ratio[None], (0,padded_frames-frames), mode='replicate')[0]
    with torch.no_grad():
        features = [model.audio_encoder(chunk[None]).detach() for chunk in audio.split(64000)]
        conditions = [mixed_multiscale_audio_condition(feature,
            frame_width_ratios=held[n*100:(n+1)*100], strength=1., patch_lengths=model.patch_nums,
            ratio_frame_count=100).detach() for n,feature in enumerate(features)]
        raw = rollout_precomputed(model, conditions, style, straight_through=False).raw_motion[0,:frames]
        motion = postprocess_motion(raw)
    if tuple(motion.shape) != (frames,106) or not bool(torch.isfinite(motion).all()):
        raise ValueError('Invalid full native FLAME motion')
    return motion


def export_video(repo, renderer, motion, audio, target):
    import numpy as np
    import torch
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.stem+'.partial.mkv')
    command = [str(repo/'media/bin/ffmpeg'), '-y', '-nostdin', '-v', 'error', '-threads', '2',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '512x512', '-r', '25', '-i', 'pipe:0',
        '-i', str(audio), '-map', '0:v:0', '-map', '1:a:0', '-t', str(len(motion)/25),
        '-c:v', 'ffv1', '-level', '3', '-pix_fmt', 'bgr0', '-threads', '2', '-c:a', 'copy', str(temporary)]
    child = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        with torch.no_grad():
            for rgb in renderer.iter_frames(motion):
                array = (rgb.clamp(0,1)*255+.5).byte().permute(1,2,0).cpu().numpy()
                child.stdin.write(np.ascontiguousarray(array).tobytes())
        child.stdin.close()
        if child.wait(timeout=300):
            raise RuntimeError('Video encoding failed')
    finally:
        if child.poll() is None:
            child.kill()
            child.wait()
    os.replace(temporary, target)


def generate(repo, output, gpu, clips=None, identities=None):
    paths = model_paths(repo)
    configure(repo, output, gpu, paths)
    import numpy as np
    import torch
    from ._runtime.renderer import Renderer
    entries = select_entries(repo, clips)
    id_entries = read(repo/'configs/identities.json')['entries']
    if identities:
        wanted = set(identities.split(','))
        if wanted - {row['identity'] for row in id_entries}:
            raise ValueError('Unknown identity')
        id_entries = [row for row in id_entries if row['identity'] in wanted]
    if not id_entries:
        raise ValueError('Empty identity list')
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output/'generation.json'
    result = read(manifest_path) if manifest_path.exists() else dict(state='running', entries=[])
    scope = dict(clips=[e['clip'] for e in entries], identities=[e['identity'] for e in id_entries],
                 checkpoint_sha256=CHECKPOINT_SHA256, evaluation_sha256=sha256(repo/'configs/evaluation.json'),
                 identities_sha256=sha256(repo/'configs/identities.json'))
    if result.get('scope', scope) != scope:
        raise ValueError('Run scope changed; use a fresh output directory')
    result['scope'] = scope
    completed = set()
    for row in result['entries']:
        if sha256(output/row['video']) != row['video_sha256'] or sha256(output/row['motion']) != row['motion_sha256']:
            raise ValueError('Completed output was changed')
        completed.add((row['clip'],row['identity']))
    if len(completed) == len(entries)*len(id_entries):
        return result
    model, style = load_model(paths)
    renderers = {}
    for row in id_entries:
        identity = identifier(row['identity'])
        renderer = Renderer(repo, row, paths, output/'avatars'/identity)
        if renderers:
            reference = next(iter(renderers.values())).model
            if any(not torch.equal(t, reference.state_dict()[name]) for name,t in renderer.model.state_dict().items()):
                raise ValueError('Renderer checkpoint mismatch')
            renderer.model = reference
        renderers[identity] = renderer
        save(output/'avatars'/identity/'metadata.json', renderer.metadata)
    for entry in entries:
        clip = identifier(entry['clip'])
        if all((clip,identity) in completed for identity in renderers):
            continue
        began = time.monotonic()
        motion_path = output/'motions'/(clip+'.npy')
        motion_receipt = motion_path.with_suffix('.json')
        motion_path.parent.mkdir(parents=True, exist_ok=True)
        if motion_path.exists() and motion_receipt.exists():
            receipt = read(motion_receipt)
            if receipt['checkpoint_sha256'] != CHECKPOINT_SHA256 or receipt['motion_sha256'] != sha256(motion_path):
                raise ValueError('Saved motion changed')
            motion = torch.from_numpy(np.load(motion_path, allow_pickle=False)).cuda()
        else:
            motion = generate_motion(model, style, repo, output, entry)
            temporary = motion_path.with_suffix('.tmp')
            with temporary.open('wb') as stream:
                np.save(stream, motion.cpu().numpy(), allow_pickle=False)
            os.replace(temporary, motion_path)
            save(motion_receipt, dict(clip=clip, checkpoint_sha256=CHECKPOINT_SHA256,
                                     motion_sha256=sha256(motion_path)))
        motion_seconds = time.monotonic()-began
        for identity, renderer in renderers.items():
            if (clip,identity) in completed:
                continue
            print('Rendering: '+clip+' / '+identity, flush=True)
            began = time.monotonic()
            video = output/'videos'/identity/(clip+'.mkv')
            audio = resolve(repo, entry.get('score_audio',entry['audio']))
            export_video(repo, renderer, motion, audio, video)
            row = dict(clip=clip, identity=identity, language=entry['language'], frames=entry['frames'],
                video=str(video.relative_to(output)), motion=str(motion_path.relative_to(output)),
                video_sha256=sha256(video), motion_sha256=sha256(motion_path),
                audio_sha256=sha256(audio), pcm_sha256=entry.get('pcm_sha256'),
                render_seconds=time.monotonic()-began, motion_seconds=motion_seconds,
                checkpoint_sha256=CHECKPOINT_SHA256, fitting_steps=0, AW_strength=1.)
            result['entries'].append(row)
            completed.add((clip,identity))
            result['completed'] = len(completed)
            save(manifest_path, result)
    result['state'] = 'complete'
    save(manifest_path, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--clips')
    parser.add_argument('--identities')
    args = parser.parse_args()
    generate(args.repo.resolve(), args.output_dir.resolve(), args.gpu, args.clips, args.identities)


if __name__ == '__main__':
    main()

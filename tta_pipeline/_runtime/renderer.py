"""Project FLEX-1 integration with a pinned source camera and matte guard.

Adapted from the project experiments public10_common30_renderer_v1.py,
public10_matte_guard_v1.py, and flex_artalk_render_v1.py. This wrapper uses
externally installed FlexAvatar, gaussian_splatting, visage, and MODNet APIs.
PinnedMODNet retains the visage MODNetMatter initialization convention while
loading an explicit local checkpoint; its empty-matte fallback is project code.
Model architectures, rendering kernels, and weights remain external assets.
Only filesystem routing and historical experiment receipt checks differ from
the local experiment integration.
"""
from __future__ import annotations
import pickle
from pathlib import Path
import numpy as np
import torch
from torch import nn

from .io import resolve
from .motion import motion106_to_expression135


class _NumpyCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'numpy._core' or module.startswith('numpy._core.'):
            module = 'numpy.core' + module[len('numpy._core'):]
        return super().find_class(module, name)


class _TrackingPickle:
    __name__ = 'pickle'
    Unpickler = _NumpyCompatUnpickler
    load = staticmethod(pickle.load)
    loads = staticmethod(pickle.loads)
    dump = staticmethod(pickle.dump)
    dumps = staticmethod(pickle.dumps)


class Renderer(nn.Module):
    def __init__(self, repo, entry, paths, output):
        super().__init__()
        from flexavatar.config.dataset_config import SampleMetadata
        from flexavatar.data_adapter import example_data
        from flexavatar.data_adapter.in_the_wild_data_adapter import InTheWildDataAdapter
        from flexavatar.model.flexavatar_preprocessor import FlexAvatarPreprocessor
        from flexavatar.model_manager.flexavatar_model_manager import FlexAvatarModelManager
        from gaussian_splatting.arguments import PipelineParams2
        from gaussian_splatting.gaussian_renderer import render_distwar
        from gaussian_splatting.scene.cameras import pose_to_rendercam
        from visage.matting.modnet import MODNetMatter, MODNet, transforms

        source = resolve(repo, entry['source_image'])
        crop = resolve(repo, entry['crop'])
        tracking_path = resolve(repo, entry['tracking'])
        tracking = torch.load(tracking_path, map_location='cpu', weights_only=False,
                              pickle_module=_TrackingPickle)
        identity = entry['identity']
        guard_records = []

        class SourceAdapter(InTheWildDataAdapter):
            def get_image_path(self, image_name):
                if image_name != identity:
                    raise ValueError('Unexpected source identity')
                return str(source)

            def _load_tracking(self, timestep):
                if timestep != 0:
                    raise ValueError('Only the pinned source frame is permitted')
                return tracking

            def _get_crop_params_path(self):
                return crop

        class EagerManager(FlexAvatarModelManager):
            def load_model_config(self, *args, **kwargs):
                config = super().load_model_config(*args, **kwargs)
                config.compile = False
                return config

        class PinnedMODNet(MODNetMatter):
            def __init__(self):
                self._transform = transforms.Compose([transforms.Normalize((.5, .5, .5), (.5, .5, .5))])
                self._modnet = nn.DataParallel(MODNet(backbone_pretrained=False)).cuda()
                weights = torch.load(paths['modnet_weights'], map_location='cpu', weights_only=True)
                self._modnet.load_state_dict(weights)
                self._modnet.eval()

            def parse(self, *args, **kwargs):
                alpha = super().parse(*args, **kwargs)
                mean = alpha.mean(dim=(-2, -1))
                failed = mean < .05
                guard_records.extend(dict(mean_alpha=float(m), fallback=bool(f)) for m, f in zip(mean, failed))
                return torch.where(failed[:, None, None], torch.ones_like(alpha), alpha) if bool(failed.any()) else alpha

        manager = EagerManager('FLEX-1')
        dataset_config = manager.load_dataset_config()
        adapter = SourceAdapter(identity, expression_code_config=dataset_config.expression_code_config)
        cuda_index = torch.cuda.current_device()
        original_matter = example_data.MODNetMatter
        example_data.MODNetMatter = PinnedMODNet
        try:
            with torch.random.fork_rng(devices=[cuda_index]), torch.cuda.device(cuda_index), torch.no_grad():
                torch.random.default_generator.manual_seed(entry['fixed_settings']['seed'])
                torch.cuda.manual_seed(entry['fixed_settings']['seed'])
                batch = example_data.create_example_batch(adapter, identity, n_frames=1).to('cuda')
                preprocessor = FlexAvatarPreprocessor(dataset_config)
                batch = preprocessor.process(batch)
                self.model = manager.load_checkpoint(900).cuda().eval().requires_grad_(False)
                value = self.model.create_gaussian_models(
                    batch.input_images, batch.features, batch.input_cam2worlds, batch.input_intrinsics,
                    dataset_ids=batch.dataset_ids, only_internal_representations=True)
                avatar = value.internal_representations.detach().clone()
        finally:
            example_data.MODNetMatter = original_matter
        if not bool(torch.isfinite(avatar).all()) or len(guard_records) != 1:
            raise ValueError('Invalid source encoding or matte')
        self.register_buffer('avatar', avatar, persistent=False)
        self.batch = batch
        pose, intrinsics = adapter.load_camera_params(SampleMetadata(identity, None, 0, '0'))
        self.camera = pose_to_rendercam(pose, intrinsics, 512, 512)
        self.pipeline = PipelineParams2()
        self.register_buffer('background', torch.ones(3, device='cuda'), persistent=False)
        self._rasterize = render_distwar
        self.metadata = dict(identity=identity, source_matte_guard=guard_records,
                             render_cam2world=np.asarray(pose).tolist(),
                             render_intrinsics=np.asarray(intrinsics).tolist(),
                             fixed_settings=entry['fixed_settings'], fitting_steps=0)
        output.mkdir(parents=True, exist_ok=True)
        np.save(output/'avatar.npy', avatar.cpu().numpy(), allow_pickle=False)
        self.eval().requires_grad_(False)

    def iter_frames(self, motion):
        motion = motion.to(device=self.avatar.device, dtype=torch.float32)
        codes = motion106_to_expression135(motion)
        batch = self.batch
        for code in codes:
            value = self.model.create_gaussian_models(
                batch.input_images, batch.features, batch.input_cam2worlds, batch.input_intrinsics,
                expression_codes=code[None, None], dataset_ids=batch.dataset_ids,
                cached_internal_representations=self.avatar)
            rgb = self._rasterize(self.camera, value.gaussian_models[0][0], self.pipeline,
                                  self.background)['render']
            if tuple(rgb.shape) != (3, 512, 512) or not bool(torch.isfinite(rgb).all()):
                raise ValueError('Invalid FlexAvatar RGB')
            yield rgb.clamp(0, 1)

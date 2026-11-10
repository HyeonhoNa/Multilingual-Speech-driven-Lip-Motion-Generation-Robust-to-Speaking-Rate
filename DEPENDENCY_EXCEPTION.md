# Additional permission under GNU GPL version 3, section 7

The copyright holders of the project-authored portions covered by
[LICENSING.md](LICENSING.md) grant permission, under their copyrights in those
portions, to link or combine them with the components expressly enumerated below,
including separately obtained copies and the upstream excerpts/path-patch
material expressly identified in [LICENSING.md](LICENSING.md), and to convey
the resulting combination,
while those components remain governed by their respective terms instead of
GPLv3.

This permission applies only to these enumerated components, including
modifications of them made and used with the necessary third-party permissions.
All GPLv3 obligations applicable to the project-authored portions, including
provision of their Corresponding Source when conveying non-source forms,
remain in effect. For combinations under this permission, the GPL-side source
requirement does not require supplying the third-party-owned source portions
or model files of those separately licensed components. Source obligations
for project-authored GPL-covered modifications, including modifications within
a dependency, remain in effect. Any source or notice obligations imposed by
the components' own licenses also remain in effect.

**This permission grants no rights in third-party code, model weights, datasets
or other assets.** It does not waive their terms or establish any right to use,
modify or redistribute them. Distribution of a combination containing
third-party material requires independent permission for that material. In
particular, noncommercial conditions are not removed, and unverified licensing
is not treated as permission. This permission cannot exempt another author's
GPL-covered contribution unless that author also grants the permission.

This additional permission may be removed as provided in GPLv3 section 7.
It is a project-specific permission, not a claim to be an unmodified standard
SPDX linking-exception text. It does not permit relicensing the covered project
code under proprietary terms merely by invoking the exception.

## Enumerated components

This list identifies the allowed GPL-side combinations; it is not a grant of
any component's license. Precise provenance and applicable terms are recorded
in [THIRD_PARTY.md](THIRD_PARTY.md). The list does not automatically expand when
a new dependency is added to that document or to a package environment.

| Component | Identifying source |
|---|---|
| ARTalk | [xg-chu/ARTalk](https://github.com/xg-chu/ARTalk) |
| FlexAvatar / FLEX-1 | [tobias-kirschstein/flexavatar](https://github.com/tobias-kirschstein/flexavatar) |
| Gaussian Splatting and the distwar rasterizer variants used by FlexAvatar | [gaussian-splatting](https://github.com/tobias-kirschstein/gaussian-splatting), [diff-gaussian-rasterization-distwar](https://github.com/tobias-kirschstein/diff-gaussian-rasterization-distwar) |
| StyleGAN generator implementations used by FlexAvatar | [stylegan-generators](https://github.com/tobias-kirschstein/stylegan-generators), including its NVIDIA-origin code |
| visage / face-parser | [tobias-kirschstein/face-parser](https://github.com/tobias-kirschstein/face-parser); permissions must be obtained independently |
| MODNet | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet) and the [FlexAvatar maintainer's fork](https://github.com/tobias-kirschstein/MODNet) |
| DINOv2 | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) |
| wav2vec 2.0 / XLS-R through Transformers | [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m), [huggingface/transformers](https://github.com/huggingface/transformers) |
| TOG speaking-rate/window predictor and its local compatibility implementation | [Speed-Aware Audio-Driven Speech Animation using Adaptive Windows](https://doi.org/10.1145/3691341); source/weight permissions must be obtained independently, not supplied or licensed by this project |
| DeepSpeech and VOCA-derived compatible preprocessing | [mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech), [TimoBolkart/voca](https://github.com/TimoBolkart/voca); exact graph and VOCA permissions remain separate |
| FLAME model and evaluation code | [FLAME](https://flame.is.tue.mpg.de/); edition-specific permission remains separate |
| SyncNet, its S3FD detector and Wav2Lip evaluation scripts | [joonson/syncnet_python](https://github.com/joonson/syncnet_python), [Wav2Lip evaluation](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) |

Associated checkpoints, styles, configuration and model assets remain subject
to their own terms. Recordings and portrait datasets are not licensed by this
exception. A program's ability to load a file is not permission to distribute
that file.

# Third-party origins and citations

This document accompanies **Multilingual Speech-driven Lip Motion Generation Robust to Speaking Rate**. It records upstream attribution and observed license notices; it does not grant rights to upstream software, checkpoints, datasets, or identities. The publication policy is to obtain upstream source trees and assets separately, without committing third-party source trees, weights, data, compiled libraries, or packed environments. Small modified integrations derived from licensed code retain explicit origin and license notices; they are not presented as original upstream implementations. Bibliographic entries are in [CITATIONS.bib](CITATIONS.bib).

## Core dependencies

| Component and role | Author-maintained source | Observed terms and asset scope | Citation key |
|---|---|---|---|
| ARTalk: autoregressive FLAME motion generator | [xg-chu/ARTalk](https://github.com/xg-chu/ARTalk) | [MIT source license](https://github.com/xg-chu/ARTalk/blob/92f6c81b75939f55e42d12f96a141e023d10c22a/LICENSE). Obtain released weights/style assets through the author's instructions. This code license does not replace the terms of FLAME, XLS-R, or other included assets. | `chu2025artalk` |
| FlexAvatar / FLEX-1: portrait-to-avatar rendering | [tobias-kirschstein/flexavatar](https://github.com/tobias-kirschstein/flexavatar) | The pinned [LICENSE](https://github.com/tobias-kirschstein/flexavatar/blob/6e009870cad61150163da30b7ee4fedca316d773/LICENSE) is **CC BY-NC 4.0**, while its `pyproject.toml` says Apache 2.0. Record this discrepancy; do not treat the package metadata as a replacement license. Obtain FLEX-1 checkpoints and dependent assets from the upstream instructions. No separate checkpoint redistribution grant is asserted here. | `kirschstein2026flexavatar` |
| TOG adaptive windows: speaking-rate estimator and window predictor | [Jung et al., DOI 10.1145/3691341](https://doi.org/10.1145/3691341); [KAIST publication record](https://pure.kaist.ac.kr/en/publications/speed-aware-audio-driven-speech-animation-using-adaptive-windows/) | The supplied local `TOG24_sj` tree has no root Git metadata or root license file. No official public repository/download URL or checkpoint redistribution permission was established. Supply the required checkpoints/statistics separately from an authorized source. | `jung2024speedaware` |
| wav2vec 2.0 / XLS-R 300M: multilingual audio features inside ARTalk | [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) | The model card declares Apache 2.0. The runtime uses the encoder stored in the selected ARTalk checkpoint; this is separate from DeepSpeech. | `babu2021xlsr` |
| SyncNet: official audiovisual synchrony evaluator | [joonson/syncnet_python](https://github.com/joonson/syncnet_python) | [MIT source license](https://github.com/joonson/syncnet_python/blob/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623/LICENSE.md). Obtain SyncNet and S3FD model assets using upstream links and retain their applicable terms. Recorded local weight hashes are not evidence of a fresh download from the author. | `chung2016outoftime` |
| Wav2Lip: LSE scoring scripts added to the SyncNet evaluator | [Rudrabha/Wav2Lip](https://github.com/Rudrabha/Wav2Lip/tree/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation) | The pinned [README](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/README.md) restricts use to personal/research/noncommercial purposes. The SyncNet MIT notice does not replace these separate conditions. This project uses evaluation code, not Wav2Lip synthesis. | `prajwal2020wav2lip` |
| DINOv2 ViT-B/14 with registers: FlexAvatar image features | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | The upstream README states Apache 2.0 for classic DINOv2 code and model weights. This statement does not cover the repository's separately licensed XRay-DINO or Cell-DINO additions. | `oquab2023dinov2`, `darcet2023registers` |
| MODNet: portrait matte | [ZHKKKe/MODNet](https://github.com/ZHKKKe/MODNet); [FlexAvatar maintainer's fork](https://github.com/tobias-kirschstein/MODNet) | The [fork README](https://github.com/tobias-kirschstein/MODNet/blob/b2b64fdb828dafe120e491ddcad94636cf9dfa7f/README.md) explicitly applies Apache 2.0 to code, models, and demos, excluding GIFs under `doc/gif`. | `ke2022modnet` |
| Mozilla DeepSpeech: frozen features for rate estimation | [mozilla/DeepSpeech](https://github.com/mozilla/DeepSpeech) | [MPL 2.0 source license](https://github.com/mozilla/DeepSpeech/blob/master/LICENSE). The exact release and independent model terms for the local TOG `output_graph.pb` have not been established; do not infer them from the software license. The cited Deep Speech paper describes the research lineage, not the identity of that graph. | `hannun2014deepspeech`, `mozillaDeepSpeech` |
| VOCA: origin of the local speech preprocessing implementation | [Official repository](https://github.com/TimoBolkart/voca); [project](https://voca.is.tue.mpg.de/) | The supplied TOG `src/data_loader/deepspeech.py` retains the MPG/VOCA copyright header. The [official license](https://voca.is.tue.mpg.de/license.html) specifies noncommercial research/education/art use and restricts distribution. This local compatibility implementation is excluded from Git; no redistribution permission is asserted. | `cudeiro2019voca` |
| FLAME: face parameterization and model assets | [Official project](https://flame.is.tue.mpg.de/); [model terms](https://flame.is.tue.mpg.de/modellicense.html) | Terms depend on the exact model edition. The official page distinguishes academic FLAME versions from **FLAME 2023 Open**. The local `FLAME_with_eye.pt` has not been established as the Open edition; acquire the appropriate model under its own terms. ARTalk's MIT license is not a FLAME model license. | `li2017flame` |

## Asset acquisition references

The pinned [ARTalk resource script](https://github.com/xg-chu/ARTalk/blob/92f6c81b75939f55e42d12f96a141e023d10c22a/build_resources.sh) names the author's [base checkpoint](https://huggingface.co/xg-chu/ARTalk/resolve/main/ARTalk_wav2vec.pt), [configuration](https://huggingface.co/xg-chu/ARTalk/resolve/main/config.json), and [natural style motion](https://huggingface.co/xg-chu/ARTalk/resolve/main/style_motion/natural_0.pt). The FlexAvatar README supplies the [FLEX-1 checkpoint](https://kaldir.vc.cit.tum.de/flexavatar/models/FLEX-1/checkpoints/ckpt-900k.pt). These are author-hosted acquisition references, not mirrors or redistribution grants; mutable URLs must be checked against the project's expected asset hashes. Acquire FLAME through its own registration/license process.

For the project-specific 9,000-step update, TOG checkpoints/statistics, the exact frozen DeepSpeech graph, and local compatibility modules, obtain the precise hash-matching artifacts from their respective authors or an authorized provider. No public download is claimed for those artifacts, and an installed DeepSpeech package version does not identify the release of the frozen TensorFlow graph.

## Supporting renderer code

FlexAvatar imports the maintainer's [Gaussian Splatting fork](https://github.com/tobias-kirschstein/gaussian-splatting), [rasterizer forks](https://github.com/tobias-kirschstein/diff-gaussian-rasterization-distwar), and [stylegan-generators](https://github.com/tobias-kirschstein/stylegan-generators). Gaussian Splatting components carry their own Inria/MPII research license notices; follow the license in each fetched repository. Cite `kerbl2023gaussians` for the underlying rendering method.

The StyleGAN wrapper's package metadata says Apache 2.0, but copied NVIDIA implementation files contain `LicenseRef-NvidiaProprietary` notices. Do not infer a uniform Apache license for those files. Consult the file-level notices and [NVIDIA StyleGAN3 license](https://github.com/NVlabs/stylegan3/blob/main/LICENSE.txt); cite `karras2021aliasfree` where that implementation is used. Additional packages installed into an environment retain their own notices; the dependency records under `environments/` identify the versions used here.

## Data

| Dataset | Official source and observed terms | Local use | Citation key |
|---|---|---|---|
| FLEURS | [Google dataset card](https://huggingface.co/datasets/google/fleurs), which declares CC BY 4.0 | Selected test audio and transcripts. Per-record provenance in `configs/evaluation.json` records revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`, source paths, and hashes. Recordings are acquired separately. Retained transcript metadata remains FLEURS-derived material. | `conneau2022fleurs` |
| Labeled Faces in the Wild (LFW) | [UMass project](https://vis-www.cs.umass.edu/lfw/); [original report](https://people.cs.umass.edu/~elm/papers/lfw.pdf) | Portrait inputs and their derived tracking/crop assets are supplied separately. Photograph rights are not established by the license of a downloader such as scikit-learn. This repository asserts no new redistribution rights or subject permissions. | `huang2007lfw` |

## Source-level attribution map

- `tta_pipeline/generation.py` integrates ARTalk inference, the local trained parameter update, and the adaptive pooling interface. ARTalk-derived generation/rollout changes retain their MIT attribution notices. The multilingual audio encoder originates in XLS-R. The project-specific 9,000-step fine-tuned checkpoint is also a separate local prerequisite; it is not available from the upstream ARTalk repository.
- Local-only `tta_pipeline/_runtime/rate.py` and `tta_pipeline/_runtime/deepspeech.py` contain compatibility implementations associated with the supplied TOG/DeepSpeech/VOCA processing path. They are excluded from Git because source redistribution permissions have not been established. The runtime needs these authorized local modules as well as the TOG checkpoints/statistics; **TOG weights alone are insufficient**. No clean-room or independent-implementation claim is made.
- `tta_pipeline/_runtime/renderer.py` integrates FlexAvatar, DINOv2, MODNet, and the Gaussian Splatting renderer. FLAME coefficient and rotation conversion occurs in `tta_pipeline/_runtime/motion.py`.
- `tta_pipeline/evaluation.py` orchestrates the separately obtained SyncNet CLI and Wav2Lip LSE scoring additions. Their individual notices and citations both apply.
- `configs/evaluation.json`, `configs/identities.json`, and `assets/provenance/` retain the local experiment's asset provenance. Historical provenance describes the local experiment and does not authorize publication of its referenced assets.

## Revisions recovered from the original checkouts

These are the original Git HEADs, not claims that every local file was unmodified. Runtime-specific changes and file hashes must be checked separately. In particular, the historical SyncNet checkout contains local edits, while the evaluation bundle has its own file-hash manifest.

| Repository | Recorded revision |
|---|---|
| `https://github.com/xg-chu/ARTalk.git` | `92f6c81b75939f55e42d12f96a141e023d10c22a` |
| `https://github.com/tobias-kirschstein/flexavatar.git` | `6e009870cad61150163da30b7ee4fedca316d773` |
| `https://github.com/tobias-kirschstein/MODNet.git` | `b2b64fdb828dafe120e491ddcad94636cf9dfa7f` |
| `https://github.com/tobias-kirschstein/stylegan-generators.git` | `6b1dbff11a4c5e09b70aa5d82317531688f04e37` |
| `https://github.com/tobias-kirschstein/gaussian-splatting.git` | `7b475f61eccbd23d1b0b85617dc105f7a6aee254` |
| `https://github.com/tobias-kirschstein/diff-gaussian-rasterization-distwar.git` (standard) | `380bf963065ceba23fe83d303eea44bb41bc7d6c` |
| `https://github.com/tobias-kirschstein/diff-gaussian-rasterization-distwar.git` (features) | `d4f1ead85768689e50b45b19f15dc9f5230a1902` |
| `https://github.com/joonson/syncnet_python.git` | `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623` |
| `https://github.com/Rudrabha/Wav2Lip.git` | `bac9a81e63ecc153202353372e5724b83d9e6322` |

The original DINOv2 source came from a `torch.hub` cache without Git metadata; its original commit was not recovered. TOG source similarly has no recoverable Git revision. Do not substitute the current upstream default branch and describe it as the historical revision. Model/checkpoint hashes are recorded separately in `configs/models.json` and the local asset manifests.

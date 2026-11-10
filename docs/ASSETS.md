# External models and data

Git contains no model weights, audio, transcripts, photographs or prepared identities. The complete local TTA deployment retains these in ignored directories.

## Models

| Required asset | Author source / acquisition |
|---|---|
| ARTalk base, config, natural style | [ARTalk](https://github.com/xg-chu/ARTalk), its `build_resources.sh`, and [author weights](https://huggingface.co/xg-chu/ARTalk) |
| Project step-9,000 checkpoint | Separately supplied by the project owner; not available from upstream ARTalk |
| FlexAvatar FLEX-1 | [Official setup](https://github.com/tobias-kirschstein/flexavatar); [author checkpoint](https://kaldir.vc.cit.tum.de/flexavatar/models/FLEX-1/checkpoints/ckpt-900k.pt) |
| TOG SRNet/window predictor and statistics | Authorized project/author supply; no verified public source or weight-distribution URL established |
| Exact DeepSpeech `output_graph.pb` | Supplied separately. [Mozilla software source](https://github.com/mozilla/DeepSpeech) does not establish this graph's provenance; arbitrary newer graphs are not interchangeable |
| DINOv2 / MODNet | [DINOv2](https://github.com/facebookresearch/dinov2), [MODNet](https://github.com/ZHKKKe/MODNet) and FlexAvatar's setup |
| FLAME | [Registration](https://flame.is.tue.mpg.de/) and [edition-specific terms](https://flame.is.tue.mpg.de/modellicense.html) |
| SyncNet and S3FD weights | [Official SyncNet setup](https://github.com/joonson/syncnet_python) |

Mutable upstream download URLs can provide a different artifact. Exact deployment files are pinned in the private `configs/models.json` and `assets/checksums.json`. Step-9,000 SHA256: `1dda0c9f2cb71152819050a200f94412bc39b78a999eb9023d66a082e165e5dc`.

Local `tta_pipeline/_runtime/rate.py` supplies the TOG architecture interface; `_runtime/deepspeech.py` supplies compatible TOG/VOCA preprocessing. They are retained for TTA but not published. **TOG .pth weights alone cannot replace the architectures.** The frozen DeepSpeech graph contains its network but still needs matching preprocessing.

## Input data

| Dataset | Download source |
|---|---|
| FLEURS test recordings | [Google FLEURS](https://huggingface.co/datasets/google/fleurs). The local experiment records revision `168de341b3db6859a9bac1c50a2ef5e3b47647e0`. |
| LFW photographs | [UMass LFW](https://vis-www.cs.umass.edu/lfw/); [original archive mirror](https://ndownloader.figshare.com/files/5976018), [pairs mirror](https://ndownloader.figshare.com/files/5976006), as referenced by the [scikit-learn loader](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/datasets/_lfw.py). |

The fixed 90-recording/10-ID selection, exact tracking/crops and audio processing are retained privately in `configs/evaluation.json`, `configs/identities.json`, `data/` and provenance records. Downloading other samples or re-estimating tracking is not automatically the same experiment.

## Complete local structure

```text
assets/weights/
assets/{checksums,runtime_checksums,official_syncnet_manifest}.json
configs/{models,evaluation,identities}.json
data/                               # 90 recordings, 10 prepared identities
third_party/                        # separately downloaded author sources
vendor/official_syncnet/             # evaluator sources and weights
tta_pipeline/_runtime/{rate,deepspeech}.py
environments/                       # private archives and inventories
envs/                               # installed project-local environments
media/                              # pinned FFmpeg and CUDA runtime
```

These files remain on disk when excluded from Git. Back up the complete deployment separately. Do not run `git clean -fdx` there: it would delete ignored models, data and environments.

# Environments

The local deployment was verified on Linux x86-64, NVIDIA RTX A5000 24 GiB, compatible driver and host C++ compiler. Environments are stored inside the project.

| Stage | Python | Important tested packages |
|---|---|---|
| Generation/adaptive windows | 3.9.23 | PyTorch 2.7.1+cu118, NumPy 1.26.4, CUDA 11.8 compiler |
| Legacy DeepSpeech preprocessing | 3.8.20 | DeepSpeech 0.9.3, NumPy 1.18.5; frozen graph supplied separately |
| Official LSE | 3.10.20 | PyTorch 2.5.1, NumPy 2.2.6, SceneDetect 0.6.7.1, OpenCV 4.13.0.92 |

## Existing complete TTA installation

Source-only publication leaves installed environments intact. Run `python3 run.py verify`, then the generation/evaluation command in README.

To move an authorized complete deployment, copy private models/data/manifests, media binaries and environment archives alongside the source. At the new location:

```bash
bash scripts/setup_envs.sh
# If unpacked envs/ was moved too:
bash scripts/setup_envs.sh --recreate
```

This offline installer requires the private `environments/archives.json` and matching archives, deliberately absent from Git. It verifies hashes and reinstalls only project-local environment directories.

## Independent upstream installation

1. Fetch source with `python3 scripts/fetch_upstreams.py --fetch`.
2. Follow [ARTalk](https://github.com/xg-chu/ARTalk) and [FlexAvatar](https://github.com/tobias-kirschstein/flexavatar) installation, including native Gaussian rasterizers. The source bootstrap does not install them.
3. Follow [SyncNet](https://github.com/joonson/syncnet_python) and [Wav2Lip evaluation](https://github.com/Rudrabha/Wav2Lip/tree/master/evaluation) for scoring dependencies.
4. Acquire the models, authorized compatibility modules and exact configuration in [ASSETS.md](ASSETS.md).
5. Validate generated media and official scores. Codec, image-processing and library changes can affect numerical results.

The offline installer is the verified relocation route; an independently rebuilt upstream environment is a separate, unverified setup.

# Multilingual Speech-driven Lip Motion Generation Robust to Speaking Rate

Integration and evaluation code combining TOG adaptive-window controls, wav2vec 2.0 features, the project's fine-tuned ARTalk generator and FlexAvatar rendering. Final videos are scored with the pinned SyncNet/Wav2Lip official LSE evaluation path.

## Source repository and local TTA deployment

**The local TTA installation remains complete and runnable.** Git publishes integration code, acquisition tools and attribution. Third-party sources, all checkpoints, recordings, photographs, tracking assets, environments and internal manifests remain installed locally and are excluded from Git. Untracking does not delete local files.

A source clone alone does not include the trained deployment. The project-specific step-9,000 checkpoint is not the upstream ARTalk checkpoint. See [required assets](docs/ASSETS.md), [environment setup](docs/ENVIRONMENT.md) and [attribution](THIRD_PARTY.md).

## License

Project-authored portions are licensed under the [GNU GPL version 3 only](LICENSE), with the specific GPLv3 §7 additional permission in [DEPENDENCY_EXCEPTION.md](DEPENDENCY_EXCEPTION.md). See [LICENSING.md](LICENSING.md) for the scope of this grant. ARTalk-derived portions retain their MIT notices.

External code and snippets, model weights (including the project-specific step-9,000 fine-tuned checkpoint), and data are outside this project's GPL grant and retain their applicable terms. The complete pipeline is **not automatically licensed for commercial use**: upstream noncommercial restrictions and other conditions remain in force. See [THIRD_PARTY.md](THIRD_PARTY.md) for origins and observed terms, and [ASSETS.md](docs/ASSETS.md) for separately supplied assets.

## Run the complete local installation

From the prepared local project directory:

```bash
python3 run.py verify
python3 run.py run --gpus 0 --output outputs/tta900 --background
python3 run.py status --output outputs/tta900
```

The frozen local evaluation uses **15 languages, 90 total recordings and 10 identities: 900 combinations**. Each language has six recordings: two each in slow/medium/fast categories. TTA refers to the Korean testing organization, not test-time adaptation.

Use `--gpus 0,1` when two GPUs are available. Repeat the same command to resume. Scope/hashes are checked; failed or missing evaluations are not silently omitted. Use a new output directory after changing code, configuration or GPU allocation.

| Output under `outputs/tta900/` | Content |
|---|---|
| `workers/*/videos/` | Videos with audio, lossless MKV evaluation inputs |
| `workers/*/motions/` | FLAME motion arrays |
| `cell.csv` | Official D/C per audio–identity pair |
| `language.csv`, `identity.csv`, `utterance.csv` | Grouped scores |
| `summary.json` | Completion count, averages and thresholds |
| `logs/`, `evaluation/` | Progress and evaluator receipts |

Generation uses adaptive pooling strength 1.0 without per-recording fitting. Wav2Lip/LatentSync image replacement is not part of generation; Wav2Lip contributes scoring scripts here. The original local runbook is retained at `local_docs/TTA_RUNBOOK.md`.

## Acquire author source separately

```bash
python3 scripts/fetch_upstreams.py --fetch
python3 scripts/fetch_upstreams.py --check
```

The standard-library tool downloads the required source subset directly from pinned official repositories into Git-ignored `third_party/` and `vendor/`. It checks hashes, applies three documented local-path patches and retains license notices. Valid files are reused; edited files are never overwritten. It does not download weights/data/private modules, run upstream setup scripts, or install native renderer dependencies.

Models, data and exact deployment configuration must be obtained separately as described in [ASSETS.md](docs/ASSETS.md). TOG network/VOCA preprocessing compatibility modules remain installed locally but are excluded from Git; TOG state-dictionary weights alone cannot implement the network architecture.

## Attribution and release checks

See [THIRD_PARTY.md](THIRD_PARTY.md) and [CITATIONS.bib](CITATIONS.bib) for component origins, papers and observed terms. Modified integrations retain code-level notices. The ARTalk-derived rollout includes its MIT notice and [license copy](LICENSES/ARTalk.txt). No blanket license overrides upstream code, model or data terms.

```bash
python3 -m unittest discover -s tests -v
python3 scripts/check_release.py
```

The release check inspects staged Git objects to reject copied source trees, model/data files, local compatibility modules and internal manifests. The full local deployment is verified separately with `python3 run.py verify`.

# License scope

## Project-authored material

Unless an original third-party notice or an exception below applies, the
project-authored portions of the tracked source code, scripts, tests and
documentation in this repository are licensed under the **GNU General Public
License, version 3 only (GPL-3.0-only)**. The full, unmodified license text is in
[LICENSE](LICENSE). The project also grants the limited additional permission
under GPLv3 section 7 in [DEPENDENCY_EXCEPTION.md](DEPENDENCY_EXCEPTION.md).
This is version 3 only, not an automatic grant under future GPL versions.

These grants apply only to rights held by the licensors in their own
contributions. They do not claim ownership of upstream code, change upstream
copyright notices, or license external assets. Merely being present in a local
installation does not put a file under this project's GPL grant.

## Third-party portions and dependencies

| Material | Applicable scope |
|---|---|
| ARTalk-derived routines in `tta_pipeline/_runtime/rollout.py` and ARTalk source excerpts in `scripts/fetch_upstreams.py` | Original portions retain Xuangeng Chu's MIT notice and [license](LICENSES/ARTalk.txt). Project-authored changes are covered by the grant above; the original MIT portions do not become exclusively GPL. |
| MODNet initialization in `tta_pipeline/_runtime/renderer.py` | Reworked from the pinned Apache-2.0 MODNet demo identified in that file. The original portions retain [Apache-2.0](LICENSES/Apache-2.0.txt); project modifications and their purpose are identified in the source. |
| FlexAvatar source excerpts/path patches in `scripts/fetch_upstreams.py` | Preserve the original FlexAvatar [CC BY-NC 4.0 terms](LICENSES/FlexAvatar-CC-BY-NC-4.0.txt) for upstream material; the GPL grant does not replace those terms. |
| Separately obtained ARTalk, FlexAvatar, MODNet, DINOv2, renderer libraries, visage and evaluation tools | Their original source licenses and file-level notices apply. See [THIRD_PARTY.md](THIRD_PARTY.md) and the pinned upstream manifests. |
| License texts, third-party notices and upstream metadata reproduced here | Their original terms and notices are preserved; they are not relicensed as project-authored GPL material. |

MIT- or Apache-licensed source can be used in a GPLv3-covered work with its
original notices retained. This does not allow noncommercial or unidentified
third-party terms to be overwritten with GPL. The section 7 permission resolves
only the GPL-side conditions for the specific combinations it describes.

FlexAvatar's pinned root `LICENSE` states **CC BY-NC 4.0**, despite the upstream
package metadata saying Apache 2.0. The metadata is not treated as a broader
license grant. The visage dependency has no confirmed project-wide license in
the inspected distribution; the MODNet license does not license visage itself.

## Models, data and local-only material

All checkpoints and weights (including the project's fine-tuned step-9,000
checkpoint), TOG statistics, DeepSpeech graphs, FLAME models, recordings,
transcripts, photographs, prepared identities, tracking data and environment
archives are **outside this repository's GPL grant**. They are excluded from
Git and require their own applicable terms or permission. Fine-tuning a model
does not by itself give permission to relicense the complete checkpoint.

The local-only `tta_pipeline/_runtime/rate.py` and
`tta_pipeline/_runtime/deepspeech.py` are also excluded from this grant and from
Git. AW original code and weights are confidential transferred technology and
are intentionally withheld from public distribution. Separate VOCA provenance
and permissions remain applicable. A missing or unverified license is not a
public-domain grant.

See [docs/ASSETS.md](docs/ASSETS.md) for acquisition information. This license
change does not delete or replace the complete local TTA installation.

## Use and redistribution

GPLv3 permits commercial use of covered project code subject to its terms.
It does **not** make the complete dependency stack commercially licensed.
FlexAvatar's noncommercial terms and other external restrictions still apply;
separate permission or replacement components may be needed for commercial
products or technology transfer.

When distributing project code, retain `LICENSE`, this scope document,
`DEPENDENCY_EXCEPTION.md` when relying on its permission, the applicable
`LICENSES/` copies, source notices and `THIRD_PARTY.md`. A distribution containing
third-party code or assets must independently satisfy their terms as well.
The source bootstrap and a citation to a paper are not substitutes for those
permissions. Generated videos are not automatically GPL merely because a GPL
program produced them; any rights associated with included assets still apply.

# LSE-D and LSE-C: exact evaluator definition

Project-authored documentation: **GPL-3.0-only**, with the scope and additional
permission in [LICENSING.md](../LICENSING.md) and
[DEPENDENCY_EXCEPTION.md](../DEPENDENCY_EXCEPTION.md). Upstream code, papers and
model weights retain their own terms.

**LSE-D는 시간 오프셋별 평균 특징 거리의 최솟값이며, 작을수록 좋습니다.
LSE-C는 그 평균 거리들의 중앙값에서 LSE-D를 뺀 값이며, 클수록 좋습니다.**
아래 식은 이 프로젝트가 실제로 실행하는 고정 버전의 공식 평가 코드에
대응합니다. 특징을 L2 정규화하거나 코사인 유사도로 바꾸지 않습니다.

## Exact formula for one face track

Let $v_t,a_t\in\mathbb{R}^{1024}$ be the visual and audio embeddings returned by
the pinned SyncNet model, for window starts $t=0,\ldots,T-1$. Extend audio
embeddings with zeros: $\widetilde a_u=a_u$ when $0\leq u<T$, otherwise
$\widetilde a_u=0$. With $S=\mathtt{vshift}=15$, define

$$
d_t(\delta)=\left\|v_t-\widetilde a_{t+\delta}
                  +10^{-6}\mathbf{1}\right\|_2,
\qquad \delta\in\{-S,\ldots,S\},
$$

$$
\overline d(\delta)=\frac{1}{T}\sum_{t=0}^{T-1}d_t(\delta),
\qquad
\mathrm{LSE\!\text{-}\!D}=\min_{\delta}\overline d(\delta),
$$

$$
\mathrm{LSE\!\text{-}\!C}
=\operatorname{median}_{\delta}\overline d(\delta)
 -\min_{\delta}\overline d(\delta).
$$

The $10^{-6}\mathbf{1}$ term records PyTorch's exact `pairwise_distance`
defaults, `p=2, eps=1e-6`; the usual mathematical shorthand is Euclidean
distance $\|v_t-\widetilde a_{t+\delta}\|_2$. Embeddings are the model's raw
fully connected outputs, without unit-length normalization. Distances are
unsquared, and confidence is a distance margin, not a probability.
The default epsilon is documented in the
[PyTorch 2.5.1 source](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/nn/modules/distance.py#L10-L58).

The order matters: **mean across windows, then minimum across offsets**.
It is not a mean of independently selected per-window minima. Confidence uses
the median of the **31 offset means**, not the median over all frame distances.
Zero-padded feature comparisons remain in the mean with denominator $T$;
the implementation does not discard boundary windows for each shift.

In array notation, `mdist[k]` corresponds to $\delta=k-S$. The selected index
$k^*$ is the first minimum in ascending index order, and the reported offset is

$$
o=S-k^*=-\delta^* \quad\text{frames},\qquad o/25\quad\text{seconds}.
$$

Thus an audio embedding at $t+3$ matched to visual window $t$ gives reported
offset $-3$. The search covers $\pm15$ frames ($\pm0.6$ seconds) in 40 ms steps.
The standalone helper's default is 10, but both Wav2Lip score CLIs explicitly
pass their default of 15. The wrapper retains D/C only; its `offset` field is
null and it does not estimate a new offset.

Code anchors: [distance and zero padding, lines 19–31](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/SyncNetInstance_calc_scores.py#L19-L31),
[reductions and offset, lines 131–150](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/SyncNetInstance_calc_scores.py#L131-L150),
[raw model outputs](https://github.com/joonson/syncnet_python/blob/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623/SyncNetModel.py#L44-L101).
The nine-sample median filter on diagnostic frame confidence does not enter
either returned scalar.

## Preprocessing and scoring conditions

The actual execution path is
[`evaluation.py`](../tta_pipeline/evaluation.py#L303) → pinned
`run_pipeline.py` → Wav2Lip `calculate_scores_real_videos.py` →
`SyncNetInstance_calc_scores.py`. `SyncNetInstance.py` contains the underlying
SyncNet distance/confidence logic, but is not the class imported by this
score CLI; its third return value is a distance array, whereas the Wav2Lip
variant returns the minimum scalar.

| Stage | Conditions used by the pinned implementation |
|---|---|
| Media conversion | FFmpeg converts to 25 fps, extracts JPEG frames, and extracts mono 16 kHz PCM16 audio. |
| Detection and tracks | S3FD confidence threshold 0.9, detection scale 0.25; scene splitting with `ContentDetector`; association requires IoU > 0.5 and detection-frame gap ≤ 25; accepted tracks contain > 100 detections and maximum mean box dimension > 100 pixels. Missing boxes inside a track are linearly interpolated. |
| Face crops | Box center/half-size use a 13-sample median filter. `crop_scale=0.40` expands the crop, including downward coverage; outside-image padding is pixel value 110. Crops become 224 × 224 XVID frames. Audio is cropped to the same track interval. |
| Visual input | The score class extracts JPEGs again, reads OpenCV BGR images, resizes to 224 × 224, and converts pixels directly to float32. It applies no RGB conversion, division by 255, or input mean/std normalization. Each embedding uses five consecutive frames, with a one-frame stride. |
| Audio input | `python_speech_features.mfcc(audio, sample_rate)` from version 0.6: 13 coefficients, 25 ms analysis window, 10 ms step, 26 filters, 512-point FFT, pre-emphasis 0.97, lifter 22, rectangular window, frequency range 0–8 kHz, and coefficient zero replaced by log energy. Each model input uses MFCC columns `4*t:4*t+20`. |
| Extent | $L=\min(N_{\rm video},\lfloor N_{\rm audio\ samples}/640\rfloor)$ and $T=L-5$. The loop uses starts $0,\ldots,L-6$ exactly; it does not use the additional last mathematically possible five-frame window. |
| Inference | `syncnet_v2.model`, model evaluation mode, 1024-dimensional output embeddings, batch size 20, CUDA for the Wav2Lip score class. Each offset uses the same $T$ windows. |

These conditions are visible in the pinned
[pipeline defaults](https://github.com/joonson/syncnet_python/blob/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623/run_pipeline.py#L24-L34),
[tracking/cropping](https://github.com/joonson/syncnet_python/blob/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623/run_pipeline.py#L67-L172),
[media conversion](https://github.com/joonson/syncnet_python/blob/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623/run_pipeline.py#L264-L281),
and [feature extraction](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/SyncNetInstance_calc_scores.py#L55-L123).
Library and media-binary versions are pinned locally; rebuilding with different
versions can change numerical scores.

The project wrapper additionally requires a 512 × 512, 25 fps native video,
canonical mono 16 kHz PCM16 audio with exactly 640 samples per native frame,
and one face track covering the entire converted timeline. Its extent audit
requires the official effective extent to cover every native frame. These are
project eligibility gates, not modifications to the metric. See
[`audit_extent`](../tta_pipeline/evaluation.py#L167) and
[`score_video`](../tta_pipeline/evaluation.py#L248).

## Track scores and dataset averages

- [`calculate_scores_real_videos.py`](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/calculate_scores_real_videos.py#L32-L40)
  emits one `D C` line per face track. It does **not** average tracks.
- The project wrapper requires exactly one score and one complete track.
  It neither chooses the best track nor combines multiple tracks.
- [`calculate_scores_LRS.py`](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/scores_LSE/calculate_scores_LRS.py#L32-L50)
  directly scores each input MP4 and reports the unweighted arithmetic mean
  of per-video D and C. For $M$ videos, these are $M^{-1}\sum_i D_i$ and
  $M^{-1}\sum_i C_i$; video lengths do not weight this dataset mean.

## Reference code, papers, and provenance

The executable implementation is the original pinned upstream code, invoked
externally without rewriting its LSE calculation:

- [SyncNet Python](https://github.com/joonson/syncnet_python/tree/907c0b579c2e2d83f0eae1b2ac9e720cde4e5623),
  commit `907c0b579c2e2d83f0eae1b2ac9e720cde4e5623`.
- [Wav2Lip evaluation](https://github.com/Rudrabha/Wav2Lip/tree/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation),
  commit `bac9a81e63ecc153202353372e5724b83d9e6322`.
  Its [evaluation instructions](https://github.com/Rudrabha/Wav2Lip/blob/bac9a81e63ecc153202353372e5724b83d9e6322/evaluation/README.md)
  explicitly use the pretrained SyncNet model and copy the Wav2Lip scoring
  scripts into the SyncNet repository.

The scholarly references have complementary roles:

1. **Joon Son Chung and Andrew Zisserman. _Out of Time: Automated Lip Sync in
   the Wild_.** Workshop on Multi-view Lip-reading, ACCV 2016; proceedings in
   *Computer Vision – ACCV 2016 Workshops*, LNCS 10117, pp. 251–263,
   published 2017. This is the SyncNet audio/visual embedding and synchronization
   reference. [Authors' publication page](https://www.robots.ox.ac.uk/~vgg/publications/2016/Chung16a/);
   [DOI: 10.1007/978-3-319-54427-4_19](https://doi.org/10.1007/978-3-319-54427-4_19).
2. **K R Prajwal, Rudrabha Mukhopadhyay, Vinay Namboodiri, and C V Jawahar.
   _A Lip Sync Expert Is All You Need for Speech to Lip Generation In The Wild_.**
   ACM Multimedia 2020. Section 4.2 introduces the LSE-D/LSE-C evaluation and
   uses the released SyncNet evaluator. Its separately trained expert's cosine
   loss in Section 3.3 is not the LSE scoring formula above.
   [arXiv:2008.10010](https://arxiv.org/abs/2008.10010);
   [DOI: 10.1145/3394171.3413532](https://doi.org/10.1145/3394171.3413532).

The local `assets/official_syncnet_manifest.json` records source and weight
hashes and the environment. [`verify_pristine`](../tta_pipeline/evaluation.py#L81)
checks that manifest, every listed source/asset, and media binaries before
scoring and again afterward. The cached SyncNet weight SHA-256 is
`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`;
the recorded upstream source is the
[Oxford SyncNet model](http://www.robots.ox.ac.uk/~vgg/software/lipsync/data/syncnet_v2.model).
Both weight records have `fresh_official_download=false`: byte identity is
pinned locally, but a fresh upstream download's origin was not independently
verified. This document introduces no scoring implementation or measured
experiment results.

#!/usr/bin/env python3
"""Speaking-rate Gaussian pooling for the project's ARTalk integration.

Derived from the project experiment artalk_rate_window_diagnostic.py.
The Gaussian window and ratio validation are project integration code. The
fixed area-interpolation baseline follows ARTalk/app/models.py using PyTorch
operations; ARTalk model definitions and weights remain external dependencies.
The unused diagnostic copy of ARTalk's full inference loop is omitted here.
"""

from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F


SHIPPED_PATCH_LENGTHS: Tuple[int, ...] = (1, 5, 25, 50, 100)
SHIPPED_CONDITION_LENGTH = sum(SHIPPED_PATCH_LENGTHS)
SHIPPED_WAV2VEC_LENGTH = 199
SHIPPED_WAV2VEC_DIM = 1024
CHUNK_FRAMES = 100
SAMPLE_RATE = 16_000
MOTION_FPS = 25.0

RatioInput = Union[float, int, Sequence[float], torch.Tensor]


def _validate_features(features: torch.Tensor) -> None:
    if features.ndim != 3:
        raise ValueError(
            "Expected wav2vec features with shape [B, T, C], "
            f"got {tuple(features.shape)}."
        )
    if not features.is_floating_point():
        raise TypeError(f"Features must be floating point, got {features.dtype}.")
    if features.shape[1] < 1:
        raise ValueError("The wav2vec time dimension must be nonempty.")


def _validate_patch_lengths(patch_lengths: Sequence[int]) -> Tuple[int, ...]:
    result = tuple(int(length) for length in patch_lengths)
    if not result or any(length < 1 for length in result):
        raise ValueError(f"Patch lengths must be positive, got {result}.")
    return result


def _work_dtype(dtype: torch.dtype) -> torch.dtype:
    # Coordinate exponentials are fragile in fp16/bfloat16.  Computing the
    # weights in fp32 still preserves autograd to the original inputs.
    if dtype in (torch.float16, torch.bfloat16):
        return torch.float32
    return dtype


def _as_ratio_tensor(
    width_ratio: RatioInput,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(width_ratio, torch.Tensor):
        return width_ratio.to(device=device, dtype=dtype)
    return torch.as_tensor(width_ratio, device=device, dtype=dtype)


def _expand_frame_ratios(
    width_ratio: RatioInput,
    *,
    batch_size: int,
    frame_count: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return positive ratios as ``[B, frame_count]`` without detaching them."""

    ratios = _as_ratio_tensor(width_ratio, device=device, dtype=dtype)
    if ratios.ndim == 0:
        ratios = ratios.reshape(1, 1).expand(batch_size, frame_count)
    elif ratios.ndim == 1:
        if ratios.shape[0] == 1:
            ratios = ratios.reshape(1, 1).expand(batch_size, frame_count)
        elif ratios.shape[0] == frame_count:
            ratios = ratios.unsqueeze(0).expand(batch_size, -1)
        else:
            raise ValueError(
                f"One-dimensional ratios must have 1 or {frame_count} values, "
                f"got {ratios.shape[0]}."
            )
    elif ratios.ndim == 2:
        if ratios.shape[1] == 1:
            ratios = ratios.expand(-1, frame_count)
        elif ratios.shape[1] != frame_count:
            raise ValueError(
                f"Ratio frame dimension must be 1 or {frame_count}, "
                f"got {ratios.shape[1]}."
            )
        if ratios.shape[0] == 1:
            ratios = ratios.expand(batch_size, -1)
        elif ratios.shape[0] != batch_size:
            raise ValueError(
                f"Ratio batch dimension must be 1 or {batch_size}, "
                f"got {ratios.shape[0]}."
            )
    else:
        raise ValueError(
            "Ratios must be scalar, [frames], or [B, frames], "
            f"got shape {tuple(ratios.shape)}."
        )

    if not bool(torch.isfinite(ratios).all()):
        raise ValueError("All width ratios must be finite.")
    if not bool((ratios > 0).all()):
        raise ValueError("All width ratios must be strictly positive.")
    return ratios


def fixed_area_pyramid(
    wav2vec_features: torch.Tensor,
    patch_lengths: Sequence[int] = SHIPPED_PATCH_LENGTHS,
) -> Tuple[torch.Tensor, ...]:
    """Run the exact fixed area-interpolation expression shipped by ARTalk."""

    _validate_features(wav2vec_features)
    lengths = _validate_patch_lengths(patch_lengths)
    channels_first = wav2vec_features.permute(0, 2, 1)
    return tuple(
        F.interpolate(channels_first, size=length, mode="area").permute(0, 2, 1)
        for length in lengths
    )


def gaussian_pool_scale(
    wav2vec_features: torch.Tensor,
    frame_width_ratios: RatioInput,
    output_length: int,
    *,
    ratio_frame_count: int = CHUNK_FRAMES,
) -> torch.Tensor:
    """Center-align and Gaussian-pool ``[B,T,C]`` to ``[B,output_length,C]``.

    Input and target locations are bin centers on the normalized chunk
    interval.  A ratio of one uses a Gaussian whose variance matches a uniform
    target bin: ``sigma = (1 / output_length) / sqrt(12)``.  Ratios are defined
    at 25-fps frame centers and area-averaged inside each target-token bin.
    """

    _validate_features(wav2vec_features)
    if output_length < 1:
        raise ValueError(f"output_length must be positive, got {output_length}.")
    if ratio_frame_count < 1:
        raise ValueError(
            f"ratio_frame_count must be positive, got {ratio_frame_count}."
        )

    batch_size, input_length, _ = wav2vec_features.shape
    dtype = _work_dtype(wav2vec_features.dtype)
    ratios = _expand_frame_ratios(
        frame_width_ratios,
        batch_size=batch_size,
        frame_count=ratio_frame_count,
        device=wav2vec_features.device,
        dtype=dtype,
    )
    # Every released scale divides 100, so area interpolation is the exact mean
    # of the per-frame ratios covered by each target token.
    target_ratios = F.interpolate(
        ratios.unsqueeze(1), size=output_length, mode="area"
    ).squeeze(1)

    input_centers = (
        torch.arange(input_length, device=wav2vec_features.device, dtype=dtype) + 0.5
    ) / input_length
    target_centers = (
        torch.arange(output_length, device=wav2vec_features.device, dtype=dtype) + 0.5
    ) / output_length
    distance = target_centers[:, None] - input_centers[None, :]
    sigma = target_ratios / (output_length * math.sqrt(12.0))
    logits = -0.5 * (distance[None, :, :] / sigma[:, :, None]).square()
    weights = logits.softmax(dim=-1)

    pooled = torch.bmm(weights, wav2vec_features.to(dtype=dtype))
    return pooled.to(dtype=wav2vec_features.dtype)


def mixed_multiscale_audio_features(
    wav2vec_features: torch.Tensor,
    frame_width_ratios: RatioInput = 1.0,
    strength: float = 1.0,
    patch_lengths: Sequence[int] = SHIPPED_PATCH_LENGTHS,
    *,
    ratio_frame_count: int = CHUNK_FRAMES,
) -> Tuple[torch.Tensor, ...]:
    """Return fixed/Gaussian mixtures at all ARTalk scales.

    ``strength=0`` returns the fixed tensors directly (no arithmetic), while
    ``strength=1`` selects Gaussian pooling.  Intermediate values linearly mix
    the two.  Width ratios may be scalar (fixed ratio) or ``[frames]`` /
    ``[B, frames]`` (adaptive ratios).
    """

    strength_value = float(strength)
    if not math.isfinite(strength_value) or not 0.0 <= strength_value <= 1.0:
        raise ValueError(f"strength must be finite and in [0, 1], got {strength}.")

    lengths = _validate_patch_lengths(patch_lengths)
    fixed = fixed_area_pyramid(wav2vec_features, lengths)
    if strength_value == 0.0:
        # Returning the exact F.interpolate results is intentional: even a
        # nominal zero multiply/add can change NaN or signed-zero bit patterns.
        return fixed

    gaussian = tuple(
        gaussian_pool_scale(
            wav2vec_features,
            frame_width_ratios,
            length,
            ratio_frame_count=ratio_frame_count,
        )
        for length in lengths
    )
    if strength_value == 1.0:
        return gaussian
    return tuple(
        area + strength_value * (adaptive - area)
        for area, adaptive in zip(fixed, gaussian)
    )


def mixed_multiscale_audio_condition(
    wav2vec_features: torch.Tensor,
    frame_width_ratios: RatioInput = 1.0,
    strength: float = 1.0,
    patch_lengths: Sequence[int] = SHIPPED_PATCH_LENGTHS,
    *,
    ratio_frame_count: int = CHUNK_FRAMES,
) -> torch.Tensor:
    """Concatenate the multiscale features into ARTalk's ``[B,181,C]`` input."""

    return torch.cat(
        mixed_multiscale_audio_features(
            wav2vec_features,
            frame_width_ratios=frame_width_ratios,
            strength=strength,
            patch_lengths=patch_lengths,
            ratio_frame_count=ratio_frame_count,
        ),
        dim=1,
    )


def _prepare_ratio_chunks(
    width_ratio: RatioInput,
    *,
    batch_size: int,
    sequence_frames: int,
    padded_frames: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, ...]:
    """Broadcast full-clip ratios, pad the last value, and split into 100s."""

    raw = _as_ratio_tensor(width_ratio, device=device, dtype=_work_dtype(dtype))
    if raw.numel() == 1:
        ratios = _expand_frame_ratios(
            raw,
            batch_size=batch_size,
            frame_count=padded_frames,
            device=device,
            dtype=_work_dtype(dtype),
        )
        return tuple(ratios.split(CHUNK_FRAMES, dim=1))

    if raw.ndim == 1:
        supplied_frames = raw.shape[0]
    elif raw.ndim == 2:
        supplied_frames = raw.shape[1]
    else:
        raise ValueError(
            "Full-clip ratios must be scalar, [frames], or [B, frames], "
            f"got shape {tuple(raw.shape)}."
        )
    if supplied_frames not in (sequence_frames, padded_frames):
        raise ValueError(
            "Full-clip ratios must cover the unpadded or padded 25-fps sequence: "
            f"expected {sequence_frames} or {padded_frames}, got {supplied_frames}."
        )

    ratios = _expand_frame_ratios(
        raw,
        batch_size=batch_size,
        frame_count=supplied_frames,
        device=device,
        dtype=_work_dtype(dtype),
    )
    if supplied_frames < padded_frames:
        ratios = torch.cat(
            [ratios, ratios[:, -1:].expand(-1, padded_frames - supplied_frames)],
            dim=1,
        )
    return tuple(ratios.split(CHUNK_FRAMES, dim=1))

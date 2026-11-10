"""Project tensor adapter between ARTalk motion and FLEX-1 expression inputs.

Extracted from experiments/neural_renderer_lse/flex_lora_differentiable_v1.py.
The Rodrigues conversion and differentiable Savitzky-Golay implementation
provide the project's 106-to-135 parameter bridge. Postprocessing matches
ARTalk/inference.py deployment conventions (smoothing and fixed pose/jaw).
ARTalk, FlexAvatar, FLAME models, and their assets are external dependencies;
this adapter does not define or redistribute those models or assets.
"""
from __future__ import annotations
from functools import lru_cache
import torch

def _finite_tensor(value: torch.Tensor, dimensions: int, name: str) -> None:
    if not isinstance(value, torch.Tensor) or not value.is_floating_point():
        raise TypeError(f"{name} must be a floating-point torch tensor")
    if value.ndim != 2 or value.shape[1] != dimensions or len(value) == 0:
        raise ValueError(f"{name} must have nonempty [T,{dimensions}] shape")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} contains nonfinite values")


def axis_angle_to_row6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Differentiable Rodrigues rotation, first two ROWS; stable at zero.

    Double precision matches the verified NumPy adapter before casting back to
    the input dtype. Protected square roots and a Taylor branch also make the
    zero-angle backward finite; no tensor-to-NumPy conversion is performed.
    """
    _finite_tensor(axis_angle, 3, "axis_angle")
    vector = axis_angle.to(torch.float64)
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(-1, 3, 3)
    squared_angle = vector.square().sum(-1)
    safe_angle = squared_angle.clamp_min(1e-8).sqrt()
    small = squared_angle < 1e-8
    a = torch.where(small, 1 - squared_angle / 6 + squared_angle.square() / 120,
                    torch.sinc(safe_angle / torch.pi))
    b = torch.where(small, 0.5 - squared_angle / 24 + squared_angle.square() / 720,
                    0.5 * torch.sinc(safe_angle / (2 * torch.pi)).square())
    rotation = (torch.eye(3, device=vector.device, dtype=vector.dtype)[None]
                + a[:, None, None] * skew + b[:, None, None] * (skew @ skew))
    return rotation[:, :2].reshape(-1, 6).to(axis_angle.dtype)


def motion106_to_expression135(motion: torch.Tensor, config=None) -> torch.Tensor:
    """Physical FLAME100 plus jaw axis-angle, with canonical driving extras.

    Source tracking is exclusively an image-encoding input. Nonzero global
    motion is rejected rather than silently changing the fixed frontal camera.
    Single frames are allowed for low-memory rendering.
    """
    _finite_tensor(motion, 106, "motion")
    if bool(torch.count_nonzero(motion[:, 100:103])):
        raise ValueError("Fixed-pose FLEX-1 requires zero ARTalk global pose")
    identity = motion.new_tensor((1, 0, 0, 0, 1, 0))[None].expand(len(motion), -1)
    zeros = motion.new_zeros((len(motion), 3))
    return torch.cat((motion[:, :100], identity, identity, zeros[:, :2], identity,
                      axis_angle_to_row6d(motion[:, 103:106]), identity, zeros), -1)


@lru_cache(maxsize=2)
def _savgol_weights(window_length: int, polyorder: int) -> torch.Tensor:
    # Centered polynomial basis keeps the small fixed systems well conditioned.
    x = torch.arange(window_length, dtype=torch.float64) - window_length // 2
    design = torch.stack([x.pow(power) for power in range(polyorder + 1)], 1)
    return design @ torch.linalg.pinv(design)


def savgol_interp(motion: torch.Tensor, window_length: int, polyorder: int) -> torch.Tensor:
    """Torch equivalent of scipy.signal.savgol_filter(axis=0, mode='interp').

    The first/last half-windows use the polynomial fit to the first/last full
    window, exactly the released boundary rule. This is a linear differentiable
    filter; it does not pad with replicated or reflected samples.
    """
    if not isinstance(motion, torch.Tensor) or not motion.is_floating_point() or motion.ndim != 2:
        raise ValueError("savgol_interp expects floating [T,D]")
    if window_length % 2 != 1 or not 0 <= polyorder < window_length or len(motion) < window_length:
        raise ValueError("An odd window, smaller polynomial degree, and at least one full window are required")
    weights = _savgol_weights(window_length, polyorder).to(motion.device)
    value = motion.to(torch.float64)
    half = window_length // 2
    interior = torch.einsum("tdw,w->td", value.unfold(0, window_length, 1), weights[half])
    return torch.cat((weights[:half] @ value[:window_length], interior,
                      weights[half + 1:] @ value[-window_length:]), 0).to(motion.dtype)


def postprocess_motion(motion: torch.Tensor, clip_length: int | None = None,
                       fix_pose: bool = True) -> torch.Tensor:
    """Released smoothing -> truncation -> global zero -> jaw y/z zero.

    Motion input is physical, denormalized [T,106]. The released engine requires
    T>=9 for its pose filter, including when pose will subsequently be zeroed.
    """
    _finite_tensor(motion, 106, "motion")
    if clip_length is not None and (not isinstance(clip_length, int) or clip_length < 1):
        raise ValueError("clip_length must be a positive integer or None")
    smoothed = savgol_interp(motion, 5, 2)
    pose = savgol_interp(motion[:, 100:103], 9, 3)
    if fix_pose:
        pose = pose * 0
    result = torch.cat((smoothed[:, :100], pose, smoothed[:, 103:104],
                        smoothed[:, 104:106] * 0), -1)
    return result if clip_length is None else result[:clip_length]


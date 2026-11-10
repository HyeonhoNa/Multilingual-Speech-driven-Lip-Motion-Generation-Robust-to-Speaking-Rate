# Project-authored portions: GNU GPL version 3 only.
# See LICENSE, LICENSING.md and DEPENDENCY_EXCEPTION.md.
# Third-party portions retain their original notices and licenses.
"""ARTalk-derived inference adapter with project chunk-context handling.

Adapted through experiments/neural_renderer_lse/artalk_generator_lora_v1.py
from ARTalk/app/models.py::BitwiseARModel.inference and
ARTalk/app/modules/bitwise_vae.py::{BITWISE_VAE,MultiScaleBSQ}.
The inverse-quantization and autoregressive recurrence follow those upstream
implementations. Project adaptations provide differentiable bit conversion,
validation, frozen chunk contexts, and precomputed audio conditions. Upstream
model definitions and pretrained weights are loaded separately.

The MIT notice below applies to the ARTalk-derived portions; it does not assign
a license to other project code or to any model weights or data.
"""
# MIT License
#
# Copyright (c) 2025 Xuangeng Chu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations
from dataclasses import dataclass
from typing import Sequence
import torch
from torch import nn
from torch.nn import functional as F
PATCH_LENGTHS=(1,5,25,50,100)

def straight_through_bits(logits: torch.Tensor) -> torch.Tensor:
    """Logits [...,code_dim,2] -> exact hard {0,1} values, softmax gradients."""
    if logits.shape[-1] != 2:
        raise ValueError("Bit logits must have a final dimension of 2.")
    hard = logits.argmax(dim=-1).float()
    soft = logits.softmax(dim=-1)[..., 1]
    # Parenthesization matters: soft + (hard - soft).detach() can round away
    # from the hard value.  The finite soft-minus-itself term is exactly zero.
    return hard + (soft - soft.detach())


def differentiable_bits_to_features(
    quantizer: nn.Module, bits: torch.Tensor, *, multi_scale: bool = False,
) -> torch.Tensor:
    """Released MultiScaleBSQ inverse without its no_grad decorator."""
    schedule = tuple(quantizer.scale_schedule)
    batch_size, length, code_dim = bits.shape[0], schedule[-1], quantizer.codebook_dim
    if tuple(bits.shape[1:]) != (sum(schedule), code_dim):
        raise ValueError("Full inverse requires all multiscale bits and code dimensions.")
    original = (bits.float() * 2 - 1.0) / (code_dim ** 0.5)
    start, end = 0, schedule[0]
    if multi_scale:
        original = original.permute(0, 2, 1).contiguous()
        total = bits.new_zeros(batch_size, code_dim, length, dtype=torch.float32)
        next_scales = []
        for index in range(len(schedule) - 1):
            values = F.interpolate(original[..., start:end], size=length, mode="linear")
            total = total + values
            start, end = end, end + schedule[index + 1]
            next_scales.append(F.interpolate(total, size=schedule[index + 1], mode="area"))
        return torch.cat(next_scales, dim=-1).permute(0, 2, 1).contiguous()
    total = bits.new_zeros(batch_size, length, code_dim, dtype=torch.float32)
    for index in range(len(schedule) - 1):
        values = F.interpolate(
            original[:, start:end].permute(0, 2, 1).contiguous(), size=length, mode="linear",
        )
        total = total + values.permute(0, 2, 1).contiguous()
        start, end = end, end + schedule[index + 1]
    return total + original[:, start:]


def differentiable_ar_features(
    quantizer: nn.Module, scale_index: int, bits: torch.Tensor,
) -> torch.Tensor:
    """Released partial inverse; cumulative features stay differentiable."""
    schedule = tuple(quantizer.scale_schedule)
    if not 0 <= scale_index < len(schedule) - 1:
        raise ValueError("A partial inverse requires a non-final scale index.")
    if tuple(bits.shape[1:]) != (sum(schedule[:scale_index + 1]), quantizer.codebook_dim):
        raise ValueError("Partial inverse bits must cover exactly the current prefix.")
    total = bits.new_zeros(bits.shape[0], quantizer.codebook_dim, schedule[-1], dtype=torch.float32)
    original = (bits.float() * 2 - 1.0) / (quantizer.codebook_dim ** 0.5)
    original = original.permute(0, 2, 1).contiguous()
    start, end = 0, schedule[0]
    next_scales = []
    for index in range(scale_index + 1):
        values = F.interpolate(original[..., start:end], size=schedule[-1], mode="linear").contiguous()
        total = total + values
        start, end = end, end + schedule[index + 1]
        next_scales.append(F.interpolate(total.clone(), size=schedule[index + 1], mode="area").contiguous())
    return torch.cat(next_scales, dim=-1).permute(0, 2, 1).contiguous()


def decode_frozen_codec(
    codec: nn.Module, prev_code_bits: torch.Tensor, this_code_bits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode through frozen weights with gradients to this chunk's bits.

    The codec must already be frozen/evaluated (``install_lora`` does this).
    Previous-chunk bits are constants even if supplied as floating tensors.
    """
    previous = differentiable_bits_to_features(codec.quantize, prev_code_bits.detach())
    current = differentiable_bits_to_features(codec.quantize, this_code_bits)
    decoded = codec.decoder(torch.cat([previous, current], dim=1) + codec.dec_pos_embed,
                            attn_mask=codec.attn_mask)
    motion = codec.unnorm_with_stats(decoded)
    length = codec.patch_nums[-1]
    return motion[:, :length], motion[:, length:]


@dataclass(frozen=True)
class FrozenChunkContext:
    motion_style_cond: torch.Tensor
    prev_attn_feat: torch.Tensor
    prev_code_bits: torch.Tensor


@dataclass(frozen=True)
class ChunkOutput:
    raw_motion: torch.Tensor
    final_logits: torch.Tensor
    hard_bits: torch.Tensor
    intermediate_logits: tuple[torch.Tensor, ...] = ()


@dataclass(frozen=True)
class FrozenRollout:
    contexts: tuple[FrozenChunkContext, ...]
    raw_motion: torch.Tensor
    final_logits: tuple[torch.Tensor, ...]


def _validate_model(model: nn.Module) -> None:
    if tuple(model.patch_nums) != PATCH_LENGTHS:
        raise ValueError(f"The released v1 generator requires scales {PATCH_LENGTHS}.")
    if int(model.prev_ratio) < 1:
        raise ValueError("The released generator requires a positive prev_ratio.")
    if model.attn_depth != len(model.attn_blocks):
        raise ValueError("Attention depth does not match the installed blocks.")


def generate_chunk(
    model: nn.Module, audio_condition: torch.Tensor, context: FrozenChunkContext, *,
    straight_through: bool = True, return_intermediate_logits: bool = False,
) -> ChunkOutput:
    """Run the exact five-scale released recurrence with constant chunk context."""
    _validate_model(model)
    model.eval()
    lengths = tuple(model.patch_nums)
    condition_dim = model.audio_feature_dim
    if tuple(audio_condition.shape) != (1, sum(lengths), condition_dim):
        raise ValueError(f"Expected audio condition [1,{sum(lengths)},{condition_dim}].")
    style = context.motion_style_cond.detach()
    previous = context.prev_attn_feat.detach()
    previous_bits = context.prev_code_bits.detach()
    audio_condition = audio_condition.detach()
    width = model.pos_embed.shape[-1]
    if tuple(style.shape) != (1, 1, width):
        raise ValueError("Style condition must be [1,1,embedding_dim].")
    if tuple(previous.shape) != (1, sum(lengths) * model.prev_ratio, width):
        raise ValueError("Previous attention context has an invalid shape.")
    level_position = model.lvl_embed(model.lvl_idx) + model.pos_embed
    previous_level_position = model.lvl_embed(model.lvl_idx).repeat(1, model.prev_ratio, 1) + model.prev_pos_embed
    next_features = style
    all_logits = []
    for scale_index, _ in enumerate(lengths):
        prefix_length = sum(lengths[:scale_index + 1])
        condition = audio_condition[:, :prefix_length]
        bias = model.attn_bias_for_masking[:, :, :prefix_length,
                                         :prefix_length + sum(lengths) * model.prev_ratio]
        features = next_features + level_position[:, :next_features.shape[1]]
        for block in model.attn_blocks:
            features = block(features, previous + previous_level_position, condition, attn_bias=bias)
        logits = model.logits_head(model.cond_logits_head(features, condition))
        bit_logits = logits.view(logits.shape[0], logits.shape[1], -1, 2)
        hard_bits = bit_logits.argmax(dim=-1)
        bits = straight_through_bits(bit_logits) if straight_through else hard_bits
        if return_intermediate_logits:
            all_logits.append(logits)
        if scale_index < len(lengths) - 1:
            next_features = differentiable_ar_features(model.basic_vae.quantize, scale_index, bits)
            next_features = torch.cat([style, model.vqfeat_embed(next_features)], dim=1)
    _, motion = decode_frozen_codec(model.basic_vae, previous_bits, bits)
    return ChunkOutput(motion, logits, hard_bits, tuple(all_logits))


@torch.no_grad()
def make_initial_context(model: nn.Module, motion_style_cond: torch.Tensor) -> FrozenChunkContext:
    """Use the released zero-motion bootstrap and the caller's frozen style."""
    _validate_model(model)
    model.eval()
    style = motion_style_cond.detach()
    motion = style.new_zeros(1, model.patch_nums[-1], model.basic_vae.motion_dim)
    bits, _ = model.basic_vae.quant_to_vqidx(motion, this_motion=None)
    features = model.basic_vae.vqidx_to_ms_vqfeat(bits)
    previous = torch.cat([style, model.vqfeat_embed(features)], dim=1).repeat(1, model.prev_ratio, 1)
    return FrozenChunkContext(style, previous.detach(), bits.detach())


@torch.no_grad()
def advance_context(
    model: nn.Module, context: FrozenChunkContext, raw_motion: torch.Tensor,
) -> FrozenChunkContext:
    """Requantize generated raw motion; never backpropagate across audio chunks."""
    model.eval()
    bits, _ = model.basic_vae.quant_to_vqidx(raw_motion.detach(), this_motion=None)
    features = model.basic_vae.vqidx_to_ms_vqfeat(bits)
    style = context.motion_style_cond.detach()
    new_previous = torch.cat([style, model.vqfeat_embed(features)], dim=1)
    previous = torch.cat([context.prev_attn_feat.detach()[:, new_previous.shape[1]:], new_previous], dim=1)
    return FrozenChunkContext(style, previous.detach(), bits.detach())


def rollout_precomputed(
    model: nn.Module, audio_conditions: Sequence[torch.Tensor], motion_style_cond: torch.Tensor, *,
    straight_through: bool = True,
) -> FrozenRollout:
    """Fresh autoregressive rollout; graphs never span two audio chunks.

    Returns all padded 100-frame chunks.  The caller trims to utterance length
    and applies the existing motion smoothing.  Use no_grad for evaluation.
    """
    if len(audio_conditions) == 0:
        raise ValueError("At least one precomputed audio condition is required.")
    context = make_initial_context(model, motion_style_cond)
    contexts, motions, final_logits = [], [], []
    for condition in audio_conditions:
        contexts.append(context)
        output = generate_chunk(model, condition, context, straight_through=straight_through)
        motions.append(output.raw_motion)
        final_logits.append(output.final_logits)
        context = advance_context(model, context, output.raw_motion)
    return FrozenRollout(tuple(contexts), torch.cat(motions, dim=1), tuple(final_logits))


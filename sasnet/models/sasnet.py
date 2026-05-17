"""SASNet: SIREN backbone with frequency embedding + spatially-adaptive masks.

Architecture:

* ``input_layer``: a frozen frequency-embedding ``SineLayer`` whose neurons are
  partitioned into a low-frequency group followed by ``num_high_freq_bands``
  high-frequency band groups (see ``layers.SineLayer``).
* ``middle_layers``: a stack of ``SineLayer``s, each split into
  ``masked_hidden_layer_groups[i]`` equally-sized neuron groups for masking.
* ``output_layer``: a plain ``nn.Linear``.
* When ``use_masks=True``, a lightweight ``HashGrid`` (kaolin-wisp) + 1-hidden-layer
  ReLU MLP produces all per-coordinate masks in a single forward pass.

The hash-grid encoder is only imported when actually needed, so the package can
be installed and the unmasked path used without kaolin-wisp present.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn.functional as F
from torch import nn

from .layers import SineLayer


class _MaskDecoder(nn.Module):
    """ReLU-MLP with a single hidden layer and sigmoid output (paper: width 48)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        self.hidden = nn.Linear(in_features, hidden_features)
        self.out = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.out(F.relu(self.hidden(x))))


def _import_hash_grid():
    try:
        from wisp.models.grids import HashGrid  # type: ignore
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "SASNet with use_masks=True requires kaolin-wisp. Install from "
            "https://github.com/NVIDIAGameWorks/kaolin-wisp."
        ) from exc
    return HashGrid


class SASNet(nn.Module):
    """Spatially-Adaptive Sinusoidal Network.

    Args:
        in_features: input coordinate dimension (2 for images, 3 for volumes/SDF).
        hidden_features: list of widths for the input + middle SineLayers
            (length ``hidden_layers + 1``). The first entry is the FEmb width.
        hidden_layers: number of middle SineLayers.
        out_features: output dimension (e.g. 3 for RGB, 1 for SDF/density).
        period: spatial period of the input domain (used by the FEmb sampler).
            With coordinates in ``[-1, 1]^n`` this is typically ``2.0``.
        omega_0: SIREN ω₀ applied to the middle SineLayers.
        bandlimit / low_range / perc_low_freqs / num_high_freq_bands /
            has_low_freq_mask: passed through to the FEmb input layer.
        masked_hidden_layer_groups: number of neuron groups per middle layer
            (length ``hidden_layers``). Set to all ``0``s to disable hidden masking.
        use_masks: if False, the hash-grid / mask decoder is not constructed
            and the network behaves as a plain SIREN with FEmb initialization.
        hash_grid_kwargs: kwargs forwarded to ``HashGrid.from_geometric``.
        mask_decoder_hidden: width of the mask decoder hidden layer (paper: 48).
    """

    _DEFAULT_HASH_GRID_KWARGS = {
        "feature_dim": 2,
        "num_lods": 10,
        "multiscale_type": "cat",
        "feature_std": 1.0e-02,
        "feature_bias": 0.0,
        "codebook_bitwidth": 10,
        "min_grid_res": 16,
        "max_grid_res": 256,
        "blas": None,
    }

    def __init__(
        self,
        in_features: int,
        hidden_features: Sequence[int],
        hidden_layers: int,
        out_features: int,
        bias: bool = True,
        period: float = 2.0,
        omega_0: float = 30.0,
        bandlimit: Optional[int] = None,
        low_range: int = 15,
        perc_low_freqs: float = 0.7,
        num_high_freq_bands: int = 1,
        has_low_freq_mask: bool = False,
        masked_hidden_layer_groups: Optional[Sequence[int]] = None,
        use_masks: bool = False,
        hash_grid_kwargs: Optional[dict] = None,
        mask_decoder_hidden: int = 48,
    ) -> None:
        super().__init__()
        if len(hidden_features) != hidden_layers + 1:
            raise ValueError(
                "len(hidden_features) must equal hidden_layers + 1 "
                f"(got {len(hidden_features)} and {hidden_layers + 1})."
            )
        if masked_hidden_layer_groups is None:
            masked_hidden_layer_groups = [0] * hidden_layers
        if len(masked_hidden_layer_groups) != hidden_layers:
            raise ValueError(
                "len(masked_hidden_layer_groups) must equal hidden_layers."
            )

        self.in_features = in_features
        self.hidden_features = list(hidden_features)
        self.hidden_layers = hidden_layers
        self.out_features = out_features
        self.omega_0 = float(omega_0)
        self.period = float(period)
        self.use_masks = bool(use_masks)
        self._masked_hidden_layer_groups = list(masked_hidden_layer_groups)
        self._has_low_freq_mask = bool(has_low_freq_mask)
        self._num_high_freq_bands = int(num_high_freq_bands)

        self.input_layer = SineLayer(
            in_features=in_features,
            out_features=hidden_features[0],
            bias=bias,
            is_first=True,
            omega_0=omega_0,
            period=period,
            init_mode="sampling",
            bandlimit=bandlimit,
            low_range=low_range,
            perc_low_freqs=perc_low_freqs,
            num_high_freq_bands=num_high_freq_bands,
            has_low_freq_mask=has_low_freq_mask,
        )

        middle = []
        for i in range(hidden_layers):
            middle.append(
                SineLayer(
                    in_features=hidden_features[i],
                    out_features=hidden_features[i + 1],
                    bias=bias,
                    is_first=False,
                    omega_0=omega_0,
                    num_hidden_groups=int(masked_hidden_layer_groups[i]),
                )
            )
        self.middle_layers = nn.ModuleList(middle)

        self.output_layer = nn.Linear(hidden_features[-1], out_features)
        with torch.no_grad():
            bound = math.sqrt(6.0 / hidden_features[-1]) / self.omega_0
            self.output_layer.weight.uniform_(-bound, bound)

        # Mask sizes used to split the mask-decoder output.
        input_mask_size = num_high_freq_bands + (1 if has_low_freq_mask else 0)
        hidden_mask_sizes = [g for g in masked_hidden_layer_groups if g > 0]
        self._mask_sizes = ([input_mask_size] if input_mask_size > 0 else []) + hidden_mask_sizes
        self._has_input_masks = input_mask_size > 0

        if self.use_masks:
            HashGrid = _import_hash_grid()
            grid_kwargs = dict(self._DEFAULT_HASH_GRID_KWARGS)
            if hash_grid_kwargs is not None:
                grid_kwargs.update(hash_grid_kwargs)
            grid_kwargs.setdefault("coord_dim", in_features)
            self.grid = HashGrid.from_geometric(**grid_kwargs)
            feat_dim = grid_kwargs["feature_dim"]
            num_lods = grid_kwargs["num_lods"]
            decoder_in = (
                num_lods * feat_dim if grid_kwargs["multiscale_type"] == "cat" else feat_dim
            )
            self.decoder = _MaskDecoder(
                in_features=decoder_in,
                hidden_features=mask_decoder_hidden,
                out_features=sum(self._mask_sizes),
            )
        else:
            self.grid = None
            self.decoder = None

    def mask_generation(
        self, coords: torch.Tensor
    ) -> tuple[Optional[torch.Tensor], list[torch.Tensor]]:
        if self.grid is None or self.decoder is None:
            raise RuntimeError("mask_generation called but use_masks is False.")
        feats = self.grid.interpolate(coords, len(self.grid.active_lods) - 1)
        masks = self.decoder(feats)
        chunks = list(torch.split(masks, self._mask_sizes, dim=-1))
        if self._has_input_masks:
            return chunks[0], chunks[1:]
        return None, chunks

    def forward(self, coords: torch.Tensor) -> dict:
        input_masks: Optional[torch.Tensor] = None
        hidden_masks: list[torch.Tensor] = []
        if self.use_masks:
            input_masks, hidden_masks = self.mask_generation(coords)

        h = self.input_layer(coords, input_masks=input_masks)
        if hidden_masks:
            for i, layer in enumerate(self.middle_layers):
                mask = hidden_masks[i] if i < len(hidden_masks) else None
                h = layer(h, hidden_masks=mask)
        else:
            for layer in self.middle_layers:
                h = layer(h)
        out = self.output_layer(h)

        return {
            "model_out": out,
            "model_in": coords,
            "input_masks": input_masks,
            "hidden_masks": hidden_masks,
        }

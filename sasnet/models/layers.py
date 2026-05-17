"""Sine layer with optional frequency-embedding init and per-group masking.

This is the building block of SASNet:

* When ``is_first=True`` and ``init_mode="sampling"``, the linear weights are
  initialized to integer frequencies sampled from a low-frequency box and a
  high-frequency annulus (the "frequency embedding layer" of
  Novello et al. 2024, used in SASNet's first layer). These weights are
  frozen (``requires_grad=False``).
* When ``is_first=False``, the weights follow standard SIREN initialization
  and the output is optionally split into ``num_hidden_groups`` equally-sized
  groups for hidden-layer masking.
* ``forward`` accepts optional ``input_masks`` / ``hidden_masks`` tensors
  produced by the SASNet mask generator and applies them element-wise per
  group.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn


class SineLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
        period: float = 0.0,
        init_mode: str = "sampling",
        bandlimit: Optional[int] = None,
        low_range: int = 15,
        perc_low_freqs: float = 0.7,
        num_high_freq_bands: int = 1,
        has_low_freq_mask: bool = False,
        num_hidden_groups: int = 0,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = float(omega_0)
        self.is_first = is_first
        self.period = float(period)
        self.init_mode = init_mode
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.group_ends: list[int] = []

        self._bandlimit = int(bandlimit) if bandlimit is not None else int(omega_0)
        self._low_range = int(low_range)
        self._perc_low_freqs = float(perc_low_freqs)
        self._high_freq_bands = int(num_high_freq_bands)
        self._has_low_freq_mask = bool(has_low_freq_mask)
        self._seed = int(seed)

        if self.is_first and self.init_mode != "none":
            n_freqs = out_features - in_features
            if self._perc_low_freqs == 0:
                self.n_low_freqs = 0
            else:
                self.n_low_freqs = (
                    int(math.ceil(self._perc_low_freqs * out_features)) - in_features
                )
            self.n_high_freqs = n_freqs - self.n_low_freqs
        else:
            if num_hidden_groups > 0:
                if out_features % num_hidden_groups != 0:
                    raise ValueError(
                        "out_features must be divisible by num_hidden_groups."
                    )
                step = out_features // num_hidden_groups
                self.group_ends = [(i + 1) * step for i in range(num_hidden_groups)]
            else:
                self.group_ends = [out_features]

        self._init_weights()

    def _init_weights(self) -> None:
        if self.is_first:
            if self.init_mode == "sampling":
                self._init_periodic_sampling()
            elif self.init_mode == "none":
                with torch.no_grad():
                    bound = 1.0 / self.in_features
                    self.linear.weight.uniform_(-bound, bound)
            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode!r}.")
        else:
            with torch.no_grad():
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def _init_periodic_sampling(self) -> None:
        if self.out_features - self.in_features <= 0:
            raise ValueError(
                "Sampling init needs out_features > in_features "
                f"(got {self.out_features} and {self.in_features})."
            )
        if self.period == 0:
            raise ValueError(
                "Sampling init requires a non-zero period (input signal period)."
            )

        rng = np.random.default_rng(self._seed)

        low_candidates: Optional[np.ndarray] = None
        if self.n_low_freqs > 0:
            grids = np.meshgrid(
                *[
                    np.arange(-self._low_range if i > 0 else 0, self._low_range + 1)
                    for i in range(self.in_features)
                ],
                indexing="ij",
            )
            low_candidates = np.stack(grids, axis=-1).reshape(-1, self.in_features)
            low_candidates = low_candidates[np.any(low_candidates != 0, axis=-1)]

        high_candidates: Optional[np.ndarray] = None
        if self.n_high_freqs > 0:
            step = int(
                math.ceil(
                    math.sqrt(
                        max(
                            (self._bandlimit ** 2 - self._low_range ** 2)
                            / max(self.n_high_freqs, 1),
                            1.0,
                        )
                    )
                )
            )
            grids = np.meshgrid(
                *[
                    np.arange(
                        -self._bandlimit if i > 0 else 0,
                        self._bandlimit + 1,
                        step,
                    )
                    for i in range(self.in_features)
                ],
                indexing="ij",
            )
            high_candidates = np.stack(grids, axis=-1).reshape(-1, self.in_features)
            high_candidates = high_candidates[
                ~np.all(np.abs(high_candidates) <= self._low_range, axis=-1)
            ]

        if self.n_low_freqs > 0:
            replace = self.n_low_freqs >= len(low_candidates)
            chosen_low = rng.choice(low_candidates, size=self.n_low_freqs, replace=replace)
        else:
            chosen_low = np.empty((0, self.in_features))

        if self.n_high_freqs > 0:
            if self.n_high_freqs >= len(high_candidates):
                extra = rng.choice(
                    high_candidates,
                    size=self.n_high_freqs - len(high_candidates),
                    replace=True,
                )
                chosen_high = np.concatenate([high_candidates, extra], axis=0)
            else:
                chosen_high = rng.choice(
                    high_candidates, size=self.n_high_freqs, replace=False
                )
        else:
            chosen_high = np.empty((0, self.in_features))

        # Low-frequency group always starts with the canonical basis.
        if self.n_low_freqs > 0:
            low_freqs = np.concatenate([np.eye(self.in_features), chosen_low], axis=0)
            self.group_ends.append(self.n_low_freqs + self.in_features)
        else:
            low_freqs = np.eye(self.in_features)
            self.group_ends.append(self.in_features)

        # Split high frequencies into bands based on max(|w_i|).
        high_freq_groups: list[np.ndarray] = []
        if self._high_freq_bands > 1:
            cutoffs = np.linspace(
                self._low_range, self._bandlimit + 1, self._high_freq_bands + 1
            )
            for i in range(self._high_freq_bands):
                axis_max = np.abs(chosen_high).max(axis=-1)
                band = chosen_high[(axis_max >= cutoffs[i]) & (axis_max < cutoffs[i + 1])]
                high_freq_groups.append(band)
                self.group_ends.append(self.group_ends[-1] + band.shape[0])
        else:
            high_freq_groups = [chosen_high]
            self.group_ends.append(self.group_ends[-1] + chosen_high.shape[0])

        with torch.no_grad():
            for i, freqs in enumerate([low_freqs, *high_freq_groups]):
                if len(freqs) == 0:
                    continue
                start = self.group_ends[i] - freqs.shape[0]
                end = self.group_ends[i]
                self.linear.weight[start:end, :] = (
                    2 * math.pi
                    * torch.tensor(freqs, dtype=self.linear.weight.dtype)
                    / self.period
                )
        self.linear.weight.requires_grad = False

    @property
    def num_neuron_groups(self) -> int:
        return len(self.group_ends)

    def forward(
        self,
        x: torch.Tensor,
        input_masks: Optional[torch.Tensor] = None,
        hidden_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_first and self.init_mode != "none":
            x = torch.sin(self.linear(x))
            if input_masks is not None:
                start = 0 if self._has_low_freq_mask else self.group_ends[0]
                ends = self.group_ends if self._has_low_freq_mask else self.group_ends[1:]
                for g, end in enumerate(ends):
                    x[..., start:end] = (
                        x[..., start:end] * input_masks[..., g].unsqueeze(-1)
                    )
                    start = end
            return x

        x = torch.sin(self.omega_0 * self.linear(x))
        if hidden_masks is not None and hidden_masks.shape[-1] != 0:
            # Equal-sized hidden groups.
            *prefix, D = x.shape
            G = hidden_masks.shape[-1]
            if D % G != 0:
                raise ValueError(
                    f"Hidden dim {D} not divisible by group count {G}."
                )
            xv = x.reshape(*prefix, G, D // G)
            xv = xv * hidden_masks.reshape(*prefix, G, 1)
            x = xv.reshape(*prefix, D)
        return x

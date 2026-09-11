"""
SASNet: Spatially-Adaptive Sinusoidal Networks for INRs.

Feng et al., "SASNet: Spatially-Adaptive Sinusoidal Networks for INRs", CVPR 2026.
Reference implementation: https://github.com/Fengyee/SASNet_inr

Thin adapter around the official model so it plugs into this repo's task scripts:
it exposes the `(output, coords)` forward interface used by the other INRs. The model is
trained with the plain MSE of the task script (no mask-sparsity term).
"""

from torch import nn

# Requires the official package: `pip install -e /path/to/SASNet_inr` (plus kaolin-wisp for the masks).
from sasnet import SASNet as _SASNet


class SASNet(nn.Module):
    def __init__(self, in_features, out_features, hidden_features, masked_hidden_layer_groups,
                 omega_0, period, bandlimit, low_range, perc_low_freqs, num_high_freq_bands,
                 hash_grid, use_masks = True):
        super().__init__()

        hidden_features = list(hidden_features)
        self.net = _SASNet(
            in_features = in_features, out_features = out_features,
            hidden_features = hidden_features, hidden_layers = len(hidden_features) - 1,
            omega_0 = omega_0, period = period,
            bandlimit = bandlimit, low_range = low_range, perc_low_freqs = perc_low_freqs,
            num_high_freq_bands = num_high_freq_bands, has_low_freq_mask = False,
            masked_hidden_layer_groups = list(masked_hidden_layer_groups),
            use_masks = use_masks, hash_grid_kwargs = dict(hash_grid) if use_masks else None,
        )

    def forward(self, coords):
        flat = coords.reshape(-1, coords.shape[-1])
        out = self.net(flat)
        return out["model_out"].reshape(*coords.shape[:-1], -1), coords

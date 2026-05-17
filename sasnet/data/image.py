"""Single-image fitting dataset.

Coordinates follow the same convention as ``make_grid`` in the parent survey
repo (``utils.py``): values lie in ``[-1, 1]`` and, with ``center=True``, sit at
pixel centers (``-1 + c + 2c*i`` with ``c = 1/n``); with ``center=False`` they
reach the ``[-1, 1]`` boundary. The coordinate order matches the image shape
``(H, W)``, i.e. each row is ``(y, x)`` after flattening.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def make_grid(
    shape: Sequence[int],
    flatten: bool = True,
    center: bool = False,
) -> torch.Tensor:
    """Grid coordinates in ``[-1, 1]``. Mirrors ``utils.make_grid``."""
    coord_seqs = []
    for n in shape:
        if center:
            c = 1.0 / n
            seq = -1.0 + c + 2.0 * c * torch.arange(n).float()
        else:
            c = 1.0 / (n - 1)
            seq = -1.0 + 2.0 * c * torch.arange(n).float()
        coord_seqs.append(seq)
    grid = torch.stack(torch.meshgrid(*coord_seqs, indexing="ij"), dim=-1)
    if flatten:
        grid = grid.reshape(-1, grid.shape[-1])
    return grid


class ImageFitDataset(Dataset):
    """Single-image fitting: yields one ``(coords, rgb)`` pair per ``__getitem__``."""

    def __init__(self, path: str, center: bool = True) -> None:
        img = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
        h, w, _ = img.shape
        self.coords = make_grid((h, w), flatten=True, center=center)
        self.rgb = torch.from_numpy(img.reshape(-1, 3))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, _idx: int):
        return self.coords, self.rgb

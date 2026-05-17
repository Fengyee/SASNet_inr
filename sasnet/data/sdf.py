"""Point-cloud SDF dataset.

Ported from BACON/FINER's ``MeshSDF`` (``torch-ngp/FINER/sdf/bacon/dataio.py``).
Consumes an oriented point cloud (``.xyz`` with 6 columns: positions +
unit-length normals) and returns surface-perturbed coordinates with estimated
SDF targets.

The SDF for each query point ``p`` is estimated as
``dot(p - nearest_v, mean(normals of k nearest neighbors))``.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.spatial import KDTree
from torch.utils.data import Dataset


class PointCloudSDFDataset(Dataset):
    """SDF supervision from an oriented point cloud (positions + normals)."""

    def __init__(
        self,
        pointcloud_path: str,
        num_samples: int = 30 ** 3,
        coarse_scale: float = 1.0e-1,
        fine_scale: float = 1.0e-3,
        size: int = 10000,
        k_neighbors: int = 3,
    ) -> None:
        super().__init__()
        if int(num_samples) % 2 != 0:
            raise ValueError(
                f"num_samples must be even (got {num_samples}); the loader splits "
                "each batch into coarse- and fine-perturbed halves."
            )
        self.num_samples = int(num_samples)
        self.coarse_scale = float(coarse_scale)
        self.fine_scale = float(fine_scale)
        self.size = int(size)
        self.k_neighbors = int(k_neighbors)

        cloud = np.genfromtxt(pointcloud_path)
        if cloud.ndim != 2 or cloud.shape[1] < 6:
            raise ValueError(
                f"Expected an N x 6 point cloud (xyz + normals); "
                f"got shape {cloud.shape}."
            )
        self.v = cloud[:, :3].astype(np.float32)
        normals = cloud[:, 3:6].astype(np.float32)
        n_norm = np.linalg.norm(normals, axis=-1, keepdims=True)
        n_norm[n_norm == 0] = 1.0
        self.n = normals / n_norm

        self.v = self._normalize(self.v)
        self.kd_tree = KDTree(self.v)

    @staticmethod
    def _normalize(coords: np.ndarray) -> np.ndarray:
        coords = coords - np.mean(coords, axis=0, keepdims=True)
        coord_max = np.amax(coords)
        coord_min = np.amin(coords)
        coords = (coords - coord_min) / (coord_max - coord_min) * 0.9
        coords -= 0.45
        return coords

    def __len__(self) -> int:
        return self.size

    def _sample_surface(self) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.randint(0, self.v.shape[0], self.num_samples)
        points = self.v[idx].copy()
        half = points.shape[0] // 2
        points[::2] += np.random.laplace(
            scale=self.coarse_scale, size=(half, points.shape[-1])
        )
        points[1::2] += np.random.laplace(
            scale=self.fine_scale, size=(half, points.shape[-1])
        )
        # wrap points that fell outside the [-0.5, 0.5] cube
        points[points > 0.5] -= 1.0
        points[points < -0.5] += 1.0

        _, nn_idx = self.kd_tree.query(points, k=self.k_neighbors)
        nearest_v = self.v[nn_idx][:, 0]
        avg_normal = np.mean(self.n[nn_idx], axis=1)
        sdf = np.sum((points - nearest_v) * avg_normal, axis=-1, keepdims=True)
        return points.astype(np.float32), sdf.astype(np.float32)

    def __getitem__(self, _idx: int) -> tuple[dict, dict]:
        coords, sdf = self._sample_surface()
        return (
            {"coords": torch.from_numpy(coords)},
            {"sdf": torch.from_numpy(sdf)},
        )

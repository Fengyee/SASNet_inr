# SASNet: Spatially-Adaptive Sinusoidal Neural Networks

Official implementation of *SASNet: Spatially-Adaptive Sinusoidal Neural Networks*, **CVPR 2026**.

[Haoan Feng](https://fengyee.github.io)<sup>1</sup>, [Diana Aldana](https://scholar.google.com/citations?user=UBfNGnMAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Tiago Novello](https://sites.google.com/site/tiagonovellodebrito)<sup>2</sup>, [Leila De Floriani](https://geog.umd.edu/facultyprofile/de-floriani/leila)<sup>1</sup>
<sup>1</sup>University of Maryland, College Park &nbsp;&nbsp; <sup>2</sup>Institute for Pure and Applied Mathematics (IMPA)

- arXiv: <https://arxiv.org/abs/2503.09750>
- Project page: <https://fengyee.github.io/SASNet_inr/>
- Code: <https://github.com/Fengyee/SASNet_inr>

## TL;DR

SASNet pairs a **frozen frequency embedding layer** with a lightweight hash-grid MLP that produces **spatially-adaptive masks**. The masks select, per location and per frequency band, which sinusoidal neurons contribute to the output. This suppresses high-frequency leakage in smooth regions while preserving fine detail around edges and surfaces, yielding higher accuracy and faster convergence on image fitting, volumetric reconstruction, and SDF tasks at only modest overhead from the mask generation network.

## Install

```bash
conda create -n sasnet python=3.10
conda activate sasnet
pip install -e .
```

The hash-grid mask generator is required whenever you construct `SASNet(use_masks=True)` (the full SASNet model used in all paper experiments). It is built on [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp); install it from its upstream repository before enabling masks. The unmasked SIREN-style baseline (`SASNet(use_masks=False)`) runs without kaolin-wisp.

*Training scripts and configuration files are not included by now; please refer to the paper and project page for model configurations and hyperparameters.*

## Citation

```bibtex
@inproceedings{feng2026sasnet,
  title     = {SASNet: Spatially-Adaptive Sinusoidal Neural Networks},
  author    = {Feng, Haoan and Aldana, Diana and Novello, Tiago and De Floriani, Leila},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License

MIT (see `LICENSE`).

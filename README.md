# SASNet: Spatially-Adaptive Sinusoidal Networks for INRs

Official implementation of *SASNet: Spatially-Adaptive Sinusoidal Networks for INRs*, **CVPR 2026**.

[Haoan Feng](https://fengyee.github.io)<sup>1</sup>, [Diana Aldana](https://scholar.google.com/citations?user=UBfNGnMAAAAJ&hl=en&oi=ao)<sup>2</sup>, [Tiago Novello](https://sites.google.com/site/tiagonovellodebrito)<sup>2</sup>, [Leila De Floriani](https://geog.umd.edu/facultyprofile/de-floriani/leila)<sup>1</sup>
<sup>1</sup>University of Maryland, College Park &nbsp;&nbsp; <sup>2</sup>Institute for Pure and Applied Mathematics (IMPA)

- arXiv: <https://arxiv.org/abs/2503.09750>
- Project page: <https://fengyee.github.io/SASNet_inr/>
- Code: <https://github.com/Fengyee/SASNet_inr>

## Updates

**2026-09: image-fitting experiments and a fix for the masked model**

- **Fix:** training `SASNet(use_masks=True)` failed in the backward pass, because the input-masked FEmb output was written back into its input in place. It is now built out-of-place (`sasnet/models/layers.py`).
- **New:** [`experiments/image_fitting`](experiments/image_fitting) fits the *Tiger* image under the protocol of FDHO (ECCV 2026, Supp. Tab. 15). It contains:
  - two configs: `tiger_p.yaml` (the paper's layer widths, 130K parameters) and `tiger_d.yaml` (3 hidden layers, 214K parameters);
  - `fit_image.py` to train, and `analyze.py` to summarize the seeds and plot training curves, reconstructions, error maps and the learned masks;
  - `fdho_adapter/`, which runs the same configs inside the FDHO code base.
- **Results** (10 seeds, final PSNR, mean ± population std): **53.71 ± 0.04 dB** (P) and **63.73 ± 0.15 dB** (D). The experiment README also compares the masks on and off across network sizes: the masks help most when the network has clearly fewer parameters than the image has target values (pixels × channels).
- `pip install -e ".[experiments]"` installs the extra dependencies of these scripts (torchvision, pyyaml, matplotlib).

## TL;DR

SASNet pairs a **frozen frequency embedding layer** with a lightweight hash-grid MLP that produces **spatially-adaptive masks**. The masks select, per location and per frequency band, which sinusoidal neurons contribute to the output. This suppresses high-frequency leakage in smooth regions while preserving fine detail around edges and surfaces, yielding higher accuracy and faster convergence on image fitting, volumetric reconstruction, and SDF tasks at only modest overhead from the mask generation network.

## Install

```bash
conda create -n sasnet python=3.10
conda activate sasnet
pip install -e .
```

The hash-grid mask generator is required whenever you construct `SASNet(use_masks=True)` (the full SASNet model used in all paper experiments). It is built on [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp); install it from its upstream repository before enabling masks. The unmasked SIREN-style baseline (`SASNet(use_masks=False)`) runs without kaolin-wisp.

## Experiments

- [`experiments/image_fitting`](experiments/image_fitting): single-image fitting under the protocol of FDHO (ECCV 2026, Supp. Tab. 15, *Tiger*), with the two Tiger configs, training and analysis scripts, and an adapter for the FDHO code base.

*Training scripts for the remaining tasks (volumes, SDF) are not included yet; please refer to the paper and project page for their configurations and hyperparameters.*

## Related projects

- [**ImplicitTerrainV2**](https://fengyee.github.io/implicitterrainv2/) (ACM SIGSPATIAL 2026) · [code](https://github.com/Fengyee/implicitterrainv2)
  — adopts SASNet's band-wise frequency masking for the residual geometry
  model of a terrain INR, deriving the masks from a wavelet complexity field
  computed from the input instead of a jointly trained hash-grid branch.

## Citation

```bibtex
@inproceedings{feng2026sasnet,
  title     = {SASNet: Spatially-Adaptive Sinusoidal Networks for INRs},
  author    = {Feng, Haoan and Aldana, Diana and Novello, Tiago and De Floriani, Leila},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026}
}
```

## License

- **Code** (`sasnet/`, `experiments/`): MIT, see `LICENSE`.
- **Project page** (`index.html`, `static/`): built on the
  [Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template)
  (adopted from [Nerfies](https://nerfies.github.io)) and licensed under
  [CC BY-SA 4.0](http://creativecommons.org/licenses/by-sa/4.0/).

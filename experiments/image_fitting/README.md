# Image fitting (FDHO Tiger protocol)

This folder evaluates SASNet on the image-fitting benchmark of
[FDHO](https://github.com/alex-costanzino/fdho-release) (Costanzino et al., ECCV 2026,
Supp. Tab. 15, *Tiger*) under their exact protocol:

| | |
|---|---|
| Image | `torchvision.transforms.Resize(256)` (short side), so Tiger is **256×317** (H×W), 243,456 target values, pixels in [-1, 1] |
| Coordinates | endpoint-inclusive `linspace(-1, 1)` per axis |
| Training | 10,000 full-batch Adam steps on the MSE (no extra loss terms) |
| PSNR | on [0, 1]-rescaled, clamped outputs. **peak** = max over iterations (forward pass before each update), **final** = last iteration |
| Seeds | 98, 68, 41, 15, 39, 54, 82, 27, 11, 51; mean ± population std (`numpy.std`) |

## Configs and results

| Config | Architecture | Trainable params | Final PSNR (dB) | Peak PSNR (dB) |
|---|---|---|---|---|
| `configs/tiger_p.yaml` | FEmb 256 + 2 masked sine layers × 232 (the paper's image-fitting widths) | 129,841 | 53.71 ± 0.04 | 53.71 ± 0.04 |
| `configs/tiger_d.yaml` | FEmb 256 + 3 masked sine layers × 256 | 214,001 | 63.73 ± 0.15 | 63.73 ± 0.14 |

Both configs have fewer trainable parameters than the image has target values. The counts
include the hash grid (13,218) and the mask decoder, but not the frozen FEmb weights. Both
models use ω₀ = 30, bandlimit 42, low_range 9, 4 high-frequency bands and 8 mask groups per
hidden layer. The hash grid has 10 levels from resolution 8 to 128.

The paper's image-fitting config (Supp. Tab. "SASNet model configurations") uses the same
widths but targets larger images: ω₀ 43, band limit 60, low-frequency range 12, FEmb period 2
and a fixed lr of 1e-4. Here the frequency settings are scaled down to the 256-px image
(ω₀ 30, band limit 42, low range 9). The FEmb period is 2/0.95, because a period of 2 makes the
FEmb identical at x = −1 and x = +1, which FDHO's endpoint-inclusive grid both contains.

Training uses the MSE only, with Adam (β = (0.9, 0.99)) and a cosine decay of every learning
rate to 1% of its base value. The sinusoidal layers use lr 1e-3. The hash grid and mask decoder
use lr 5e-3 for P and 5e-4 for D. With 3 hidden layers, a higher mask lr makes the loss spikes
early in training larger, and the final PSNR drops and varies more across seeds.

## Setup

```bash
pip install -e ".[experiments]"   # from the repository root: torchvision, pyyaml, matplotlib
```

The masked model also needs [kaolin-wisp](https://github.com/NVIDIAGameWorks/kaolin-wisp)
(see the top-level README), whose hash grid runs on CUDA only. The Tiger image is
`data/images/tiger.png` in the FDHO repository.

## Train

```bash
python experiments/image_fitting/fit_image.py \
    --config experiments/image_fitting/configs/tiger_d.yaml \
    --image /path/to/fdho-release/data/images/tiger.png \
    --out_dir runs/tiger_d
```

This trains all 10 seeds one after another. To train seeds in parallel, launch one job per
seed with `--seeds <S>` and the same `--out_dir`. Each seed writes `seed<S>/history.npz`
(loss and PSNR per step), `seed<S>/final_model.pt` and `seed<S>/final.png`. The run directory
also gets `run_config.yaml` (the config plus the image path) and `summary.json` (the seeds of
that invocation). One seed takes 3–4 min on an RTX A5000.

## Analyze

```bash
python experiments/image_fitting/analyze.py runs/tiger_p runs/tiger_d --out_dir runs/analysis
```

The script recomputes final and peak PSNR from every `seed<S>/history.npz` in each run
directory, so seeds trained by separate jobs are aggregated together. It writes:

- `summary.md`: mean ± population std (as in FDHO), the sample std and every per-seed value.
- `training_curves.png`: PSNR per step for every seed.
- `reconstruction.png`: ground truth vs. reconstruction (`--seed`, default 98), with crops and
  |error| maps.
- `error_maps.png`: |error| of every seed.
- `masks_<run>.png`: the learned input masks (one per high-frequency band of the Fourier
  embedding) and hidden masks (one per neuron group).

The figures use the weights reloaded from `final_model.pt`, i.e. after the last update, so
their dB values can differ slightly from the table's final PSNR (the forward pass before that
update). This step also needs a CUDA GPU. `--image` overrides the image path stored in
`run_config.yaml`, for example when analyzing runs copied from another machine.

## Running inside the FDHO code base

`fdho_adapter/` contains what is needed to run SASNet with FDHO's own `02a_image_fitting.py`
(commit `f7132ac`):

- `sasnet.py`: the model wrapper. Copy it to `models/sasnet.py`.
- `config_image_fitting_sasnet.yaml`: both configs in FDHO's format. Copy it to `configs/`.
- `fdho.patch`: changes `models/models_bank.py` and `02a_image_fitting.py`. It registers the
  SASNet model with its optimizer groups and schedule, saves the per-step metric history
  (`history.npz`), and adds `--seeds` / `--out_root` flags. It does not change the data, the
  loss, the metrics or the seeds.

From the root of an `fdho-release` checkout, with this package and kaolin-wisp installed in the
same environment:

```bash
A=/path/to/SASNet_inr/experiments/image_fitting/fdho_adapter
git apply $A/fdho.patch
cp $A/sasnet.py models/sasnet.py
cp $A/config_image_fitting_sasnet.yaml configs/
python 02a_image_fitting.py --config configs/config_image_fitting_sasnet.yaml
```

This runs both configs on all 10 seeds and writes FDHO's `logs.json`. Like FDHO's own
checkpoints and images, `history.npz` sits in one folder per model and is overwritten by each
seed. To keep every seed's history, run one seed per call, e.g.
`--seeds 98 --out_root runs/fdho/seed98`.

Both code paths agree within seed-to-seed variation (10 seeds, final PSNR, dB):

| Config | `fit_image.py` | FDHO `02a_image_fitting.py` |
|---|---|---|
| P | 53.71 ± 0.04 | 53.69 ± 0.04 |
| D | 63.73 ± 0.15 | 63.76 ± 0.21 |

Individual seeds can differ by a few tenths of a dB between runs. The backward pass of the
kaolin-wisp hash grid uses `atomicAdd`, so training is not bitwise deterministic. FDHO's
`count_parameters` also counts the frozen FEmb weights (130,353 and 214,513).

## Note on model size

SASNet's masks are learned from the fitting residual: they turn on high-frequency neurons
where the backbone cannot fit the signal yet. This helps when the network has fewer parameters
than the signal has target values. When the backbone can interpolate the image by itself, the
masks no longer help.

The table compares each backbone with the masks on (SASNet) and off (plain FEmb + SIREN,
`use_masks: false`). Everything else follows the D recipe above; final PSNR in dB.

| Backbone (hidden layers) | Params, masks on | Params / targets | Masks off | SASNet | Seeds |
|---|---|---|---|---|---|
| 3 × 128 | 82K | 0.34 | 42.92 | 46.69 | 98 |
| 2 × 232 (P) | 130K | 0.53 | 50.08 | 53.72 | 98, 68, 41 |
| 3 × 192 | 140K | 0.57 | 50.26 | 53.90 | 98 |
| 3 × 256 (D) | 214K | 0.88 | 65.29 | 63.72 | 98, 68, 41 |
| 3 × 320 | 304K | 1.25 | 90.57 | 95.09 | 98 |
| 3 × 384 | 411K | 1.69 | 86.55 | 91.67 | 98 |
| 3 × 512 | 674K | 2.77 | 84.41 | 100.36 | 98 |

The P row uses the P recipe for SASNet. Well below the number of target values, the masks add
3.6–3.8 dB. At 0.88× (D), the unmasked backbone is already 1.6 dB better. Above 1×, both
networks fit the 8-bit image far past 60 dB. At that level the PSNR is dominated by
late-training oscillations (final and peak differ by up to 12 dB), so the on/off order there is
not meaningful. We recommend SASNet for networks well below the number of target values
(about half or less on this image).

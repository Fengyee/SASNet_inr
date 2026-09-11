"""Single-image fitting with SASNet.

The protocol follows the image-fitting benchmark of FDHO (Costanzino et al., ECCV 2026,
https://github.com/alex-costanzino/fdho-release) so results are directly comparable to their Supp. Tab. 15:

* the image is resized with ``torchvision.transforms.Resize(size)`` (short side = ``size``)
  and mapped to ``[-1, 1]``; coordinates are an endpoint-inclusive ``linspace(-1, 1)`` grid;
* full-batch Adam for ``steps`` iterations on the MSE (no mask-sparsity term);
* PSNR is computed on ``[0, 1]``-rescaled, clamped outputs. ``peak`` is the maximum PSNR over
  all iterations (measured on the forward pass before each update) and ``final`` is the PSNR of
  the last iteration's forward pass;
* the same 10 seeds and the population standard deviation (``numpy.std``) across seeds.

Usage::

    python experiments/image_fitting/fit_image.py \\
        --config experiments/image_fitting/configs/tiger_d.yaml \\
        --image path/to/tiger.png --out_dir runs/tiger_d [--seeds 98 68 41]

Seeds can also be trained by separate jobs into the same ``--out_dir`` (one ``--seeds S`` each);
``analyze.py`` aggregates every ``seed<S>/`` folder it finds. ``summary.json`` only covers the
seeds of the invocation that wrote it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import yaml
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor

from sasnet import SASNet

FDHO_SEEDS = [98, 68, 41, 15, 39, 54, 82, 27, 11, 51]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_image(path: str, size: int) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int]]:
    """Returns ``(coords [H*W, 2], pixels [H*W, 3] in [-1, 1], (H, W))``."""
    img = Compose([Resize(size), ToTensor()])(Image.open(path).convert("RGB")) * 2.0 - 1.0
    _, h, w = img.shape
    axes = [torch.linspace(-1, 1, steps=s) for s in (h, w)]
    coords = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, 2)
    return coords, img.permute(1, 2, 0).reshape(-1, 3), (h, w)


def psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    """PSNR of ``[-1, 1]`` signals, measured on ``[0, 1]`` after clamping."""
    pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)
    gt = torch.clamp((gt + 1.0) / 2.0, 0.0, 1.0)
    return (-10.0 * torch.log10(torch.mean((pred - gt) ** 2))).item()


def build_model(cfg: dict) -> SASNet:
    m = cfg["model"]
    hidden = list(m["hidden_features"])
    return SASNet(
        in_features=2, out_features=3,
        hidden_features=hidden, hidden_layers=len(hidden) - 1,
        omega_0=m["omega_0"], period=m["period"],
        bandlimit=m["bandlimit"], low_range=m["low_range"], perc_low_freqs=m["perc_low_freqs"],
        num_high_freq_bands=m["num_high_freq_bands"], has_low_freq_mask=False,
        masked_hidden_layer_groups=list(m["masked_hidden_layer_groups"]),
        use_masks=m.get("use_masks", True),
        hash_grid_kwargs=dict(m["hash_grid"]) if m.get("use_masks", True) else None,
    )


def build_optimizer(model: SASNet, cfg: dict, steps: int):
    """Adam (``betas`` from the config) with separate lrs for the sinusoidal backbone, hash grid
    and mask decoder.

    ``scheduler: cosine`` decays every group from its base lr to 1% of it over ``steps``;
    ``scheduler: none`` keeps the lrs constant.
    """
    t = cfg["train"]
    mask_modules = [mod for mod in (model.grid, model.decoder) if mod is not None]
    mask_ids = {id(p) for mod in mask_modules for p in mod.parameters()}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in mask_ids]
    groups = [{"params": backbone, "lr": t["lr"]}]
    if mask_modules:
        groups += [{"params": model.grid.parameters(), "lr": t["lr_grid"]},
                   {"params": model.decoder.parameters(), "lr": t["lr_decoder"]}]
    optim = torch.optim.Adam(groups, betas=tuple(t.get("betas", (0.9, 0.999))))

    sched = t.get("scheduler", "none")
    if sched == "cosine":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lambda s: 0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    elif sched == "none":
        scheduler = None
    else:
        raise ValueError(f"Unknown scheduler '{sched}' (use 'none' or 'cosine').")
    return optim, scheduler


def fit(cfg: dict, coords: torch.Tensor, gt: torch.Tensor, seed: int, out_dir: str,
        shape: tuple[int, int]) -> dict:
    set_seed(seed)
    device = torch.device("cuda")
    model = build_model(cfg).to(device)
    steps = cfg["train"]["steps"]
    optim, scheduler = build_optimizer(model, cfg, steps)
    coords, gt = coords.to(device), gt.to(device)

    history = {"loss": [], "psnr": []}
    peak = 0.0
    start = time.perf_counter()
    for _ in range(steps):
        pred = model(coords)["model_out"]
        loss = ((pred - gt) ** 2).mean()
        with torch.no_grad():
            p = psnr(pred, gt)
        peak = max(peak, p)
        history["loss"].append(loss.item())
        history["psnr"].append(p)

        optim.zero_grad()
        loss.backward()
        optim.step()
        if scheduler is not None:
            scheduler.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    os.makedirs(out_dir, exist_ok=True)
    np.savez(os.path.join(out_dir, "history.npz"), **{k: np.asarray(v) for k, v in history.items()})
    torch.save(model.state_dict(), os.path.join(out_dir, "final_model.pt"))
    img = ((pred.detach().clamp(-1, 1) + 1) / 2).reshape(*shape, 3).cpu().numpy()
    Image.fromarray((img * 255).round().astype(np.uint8)).save(os.path.join(out_dir, "final.png"))

    return {"seed": seed, "final_psnr": history["psnr"][-1], "peak_psnr": peak,
            "train_time_s": elapsed}


def summarize(runs: list[dict], n_params: int, n_targets: int) -> dict:
    finals = np.array([r["final_psnr"] for r in runs])
    peaks = np.array([r["peak_psnr"] for r in runs])
    return {
        "trainable_params": n_params, "target_values": n_targets,
        "final_psnr_mean": float(finals.mean()), "final_psnr_std": float(finals.std()),
        "peak_psnr_mean": float(peaks.mean()), "peak_psnr_std": float(peaks.std()),
        "final_psnr_sample_std": float(finals.std(ddof=1)) if len(runs) > 1 else 0.0,
        "runs": runs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=FDHO_SEEDS)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    coords, gt, shape = load_image(args.image, cfg["data"]["size"])
    n_params = sum(p.numel() for p in build_model(cfg).parameters() if p.requires_grad)
    print(f"image {shape[0]}x{shape[1]} ({gt.numel()} target values), "
          f"{n_params} trainable parameters")

    run_config = {**cfg, "image": os.path.abspath(args.image)}
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "run_config.yaml"), "w") as f:
        yaml.safe_dump(run_config, f, sort_keys=False)

    runs = []
    for seed in args.seeds:
        seed_dir = os.path.join(args.out_dir, f"seed{seed}")
        os.makedirs(seed_dir, exist_ok=True)
        with open(os.path.join(seed_dir, "run_config.yaml"), "w") as f:  # checked by analyze.py
            yaml.safe_dump(run_config, f, sort_keys=False)
        r = fit(cfg, coords, gt, seed, seed_dir, shape)
        print(f"seed {seed}: final {r['final_psnr']:.2f} dB, peak {r['peak_psnr']:.2f} dB "
              f"({r['train_time_s']:.0f} s)")
        runs.append(r)

    summary = summarize(runs, n_params, gt.numel())
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"final {summary['final_psnr_mean']:.2f} ± {summary['final_psnr_std']:.2f} dB, "
          f"peak {summary['peak_psnr_mean']:.2f} ± {summary['peak_psnr_std']:.2f} dB "
          f"(population std over {len(runs)} seeds)")


if __name__ == "__main__":
    main()

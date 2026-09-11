"""Summary table and figures for runs produced by ``fit_image.py``.

Usage::

    python experiments/image_fitting/analyze.py runs/tiger_p runs/tiger_d --out_dir runs/analysis

Each run directory is an ``--out_dir`` of ``fit_image.py``: ``run_config.yaml`` plus one
``seed<S>/`` folder per seed. The statistics are recomputed from every ``seed<S>/history.npz``
(final = last iteration, peak = max over iterations), so seeds trained by separate jobs into the
same directory are aggregated together. Needs a CUDA GPU (the hash grid is CUDA-only). Writes
to ``--out_dir``:

* ``summary.md``: mean ± population std (as in FDHO) and sample std of the final/peak PSNR,
  plus every per-seed value;
* ``training_curves.png``: PSNR per step for every seed;
* ``reconstruction.png``: ground truth vs. the reconstruction of ``--seed`` (full image, crops,
  |error|);
* ``error_maps.png``: |error| of every seed;
* ``masks_<run>.png``: the learned input masks (one per high-frequency band) and hidden masks
  (layer × neuron group) of ``--seed``.

Figures show the model reloaded from ``final_model.pt`` (the weights after the last update), so
their dB values can differ slightly from the table's final PSNR (the forward pass before that
update). ``--image`` overrides the image path stored in ``run_config.yaml``, e.g. on another machine.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from matplotlib.patches import Rectangle

from fit_image import FDHO_SEEDS, build_model, load_image

ERR_VMAX = 0.01  # |error| colour scale, in [0, 1] intensity units
TIGER_SHAPE = (256, 317)  # (H, W) after Resize(256)
TIGER_CROPS = {"eye": (90, 154, 120, 184), "whiskers": (175, 239, 75, 139)}  # r0, r1, c0, c1


def load_run(run_dir: str) -> dict:
    with open(os.path.join(run_dir, "run_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    found = []
    for h in glob.glob(os.path.join(run_dir, "seed*", "history.npz")):
        m = re.fullmatch(r"seed(\d+)", os.path.basename(os.path.dirname(h)))
        if m is None:  # e.g. a renamed "seed98_old" folder
            continue
        seed = int(m.group(1))
        seed_cfg = os.path.join(run_dir, f"seed{seed}", "run_config.yaml")
        if os.path.exists(seed_cfg):
            with open(seed_cfg) as f:
                if yaml.safe_load(f) != cfg:
                    raise ValueError(f"{seed_cfg} differs from {run_dir}/run_config.yaml: "
                                     "the seeds in one run directory must share a config")
        found.append(seed)
    if not found:
        raise FileNotFoundError(f"no seed<S>/history.npz in {run_dir}")
    seeds = sorted(found, key=lambda s: FDHO_SEEDS.index(s) if s in FDHO_SEEDS else len(FDHO_SEEDS) + s)
    curves = {s: np.load(os.path.join(run_dir, f"seed{s}", "history.npz"))["psnr"] for s in seeds}
    finals = np.array([curves[s][-1] for s in seeds])
    peaks = np.array([curves[s].max() for s in seeds])
    n_params = sum(p.numel() for p in build_model(cfg).parameters() if p.requires_grad)
    return {"name": os.path.basename(os.path.normpath(run_dir)), "dir": run_dir, "cfg": cfg,
            "seeds": seeds, "curves": curves, "finals": finals, "peaks": peaks, "n_params": n_params}


@torch.no_grad()
def reconstruct(run: dict, seed: int, coords: torch.Tensor, shape: tuple[int, int],
                device: torch.device) -> dict:
    """Reloads ``final_model.pt`` and returns the image and masks as ``[H, W, C]`` arrays."""
    model = build_model(run["cfg"]).to(device).eval()
    state = torch.load(os.path.join(run["dir"], f"seed{seed}", "final_model.pt"), map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    out = model(coords.to(device))
    to_hw = lambda x: x.reshape(*shape, -1).cpu().numpy()
    return {
        "pred": to_hw(((out["model_out"] + 1) / 2).clamp(0, 1)),
        "input_masks": to_hw(out["input_masks"]) if out["input_masks"] is not None else None,
        "hidden_masks": [to_hw(m) for m in out["hidden_masks"]],
    }


def psnr01(pred: np.ndarray, gt: np.ndarray) -> float:
    return float(-10 * np.log10(np.mean((pred - gt) ** 2)))


def write_summary(runs: list[dict], path: str) -> str:
    lines = ["| Run | Trainable params | Seeds | Final PSNR (dB) | Peak PSNR (dB) | Final sample std |",
             "|---|---|---|---|---|---|"]
    for r in runs:
        f, p = r["finals"], r["peaks"]
        sample_std = f"{f.std(ddof=1):.2f}" if len(f) > 1 else "n/a"
        lines.append(f"| {r['name']} | {r['n_params']:,} | {len(f)} | {f.mean():.2f} ± {f.std():.2f} | "
                     f"{p.mean():.2f} ± {p.std():.2f} | {sample_std} |")
    lines += ["", "± is the population std (numpy.std), as in FDHO.", "",
              "Per seed, final / peak PSNR (dB):", ""]
    for r in runs:
        cells = [f"{s}: {f:.2f} / {p:.2f}" for s, f, p in zip(r["seeds"], r["finals"], r["peaks"])]
        lines.append(f"- {r['name']}: " + ", ".join(cells))
    text = "\n".join(lines) + "\n"
    with open(path, "w") as f:
        f.write(text)
    return text


def fig_curves(runs: list[dict], path: str) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for i, r in enumerate(runs):
        for j, s in enumerate(r["seeds"]):
            label = r["name"] if j == 0 else None
            for a in ax:
                a.plot(r["curves"][s], color=f"C{i}", lw=0.6, alpha=0.7, label=label)
    curves = [c for r in runs for c in r["curves"].values()]
    tails = [c[int(0.8 * len(c)):] for c in curves]
    ax[1].set_xlim(int(0.8 * min(len(c) for c in curves)), max(len(c) for c in curves))
    ax[1].set_ylim(min(t.min() for t in tails) - 0.5, max(t.max() for t in tails) + 0.5)
    ax[0].set_title("PSNR per step (all seeds)")
    ax[1].set_title("last 20% of training")
    for a in ax:
        a.set_xlabel("step"); a.set_ylabel("PSNR (dB)"); a.grid(alpha=0.3)
    ax[0].legend()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_reconstruction(gt: np.ndarray, recs: dict, seed: int, crops: dict, path: str) -> None:
    """Columns: full image, crops, then |error| of the full image and of the crops."""
    rows = [("Ground truth", gt)] + [(f"{name}, seed {seed}", rec["pred"]) for name, rec in recs.items()]
    aspect = gt.shape[1] / gt.shape[0]
    ratios = ([aspect] + [1] * len(crops)) * 2
    fig, ax = plt.subplots(len(rows), len(ratios), figsize=(2.8 * sum(ratios), 2.9 * len(rows)),
                           squeeze=False, gridspec_kw={"width_ratios": ratios})
    e0 = 1 + len(crops)  # first |error| column
    for i, (title, img) in enumerate(rows):
        err = np.abs(img - gt).mean(-1)
        ax[i, 0].imshow(img)
        ax[i, 0].set_ylabel(title)
        ax[i, 0].set_title("full image" if i == 0 else f"{psnr01(img, gt):.2f} dB")
        ax[i, e0].set_title(f"|error|, full (mean {err.mean():.4f})" if i else "")
        for k, (cname, (r0, r1, c0, c1)) in enumerate(crops.items()):
            ax[i, 0].add_patch(Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec=f"C{k + 1}", lw=1.2))
            ax[i, 1 + k].imshow(img[r0:r1, c0:c1], interpolation="nearest")
            ax[i, 1 + k].set_title(cname if i == 0 else "")
            if i:
                ax[i, e0 + 1 + k].imshow(err[r0:r1, c0:c1], cmap="inferno", vmin=0, vmax=ERR_VMAX,
                                         interpolation="nearest")
                ax[i, e0 + 1 + k].set_title(f"|error|, {cname}")
        if i:
            im = ax[i, e0].imshow(err, cmap="inferno", vmin=0, vmax=ERR_VMAX)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([])
    for a in ax[0, e0:]:
        a.axis("off")
    fig.colorbar(im, ax=ax[1:, e0:], fraction=0.03, label=f"|pred − GT|, clipped at {ERR_VMAX}")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_error_maps(gt: np.ndarray, runs: list[dict], preds: dict, path: str) -> None:
    ncol = max(len(r["seeds"]) for r in runs)
    fig, ax = plt.subplots(len(runs), ncol, figsize=(2.6 * ncol, 3.0 * len(runs)), squeeze=False)
    for i, r in enumerate(runs):
        for j in range(ncol):
            a = ax[i, j]
            a.set_xticks([]); a.set_yticks([])
            if j >= len(r["seeds"]):
                a.axis("off")
                continue
            s = r["seeds"][j]
            pred = preds[(r["name"], s)]
            im = a.imshow(np.abs(pred - gt).mean(-1), cmap="inferno", vmin=0, vmax=ERR_VMAX)
            a.set_title(f"seed {s}: {psnr01(pred, gt):.2f} dB", fontsize=9)
        ax[i, 0].set_ylabel(r["name"])
    fig.colorbar(im, ax=ax, fraction=0.02, label=f"|pred − GT|, clipped at {ERR_VMAX}")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_masks(name: str, rec: dict, seed: int, path: str) -> None:
    """Input masks (first row) and hidden masks (one row per layer, one column per group)."""
    rows = ([("input", rec["input_masks"])] if rec["input_masks"] is not None else []) + \
        [(f"hidden {l + 1}", m) for l, m in enumerate(rec["hidden_masks"])]
    ncol = max(m.shape[-1] for _, m in rows)
    fig, ax = plt.subplots(len(rows), ncol, figsize=(2.4 * ncol, 2.6 * len(rows)), squeeze=False)
    for i, (label, masks) in enumerate(rows):
        for g in range(ncol):
            a = ax[i, g]
            a.set_xticks([]); a.set_yticks([])
            if g >= masks.shape[-1]:
                a.axis("off")
                continue
            im = a.imshow(masks[..., g], cmap="jet", vmin=0, vmax=1)
            unit = "band" if label == "input" else "group"
            a.set_title(f"{unit} {g + 1} (mean {masks[..., g].mean():.2f})", fontsize=8)
        ax[i, 0].set_ylabel(label)
    fig.colorbar(im, ax=ax, fraction=0.02)
    fig.suptitle(f"{name}, seed {seed}: input masks (per high-frequency band) and hidden masks")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("run_dirs", nargs="+")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--seed", type=int, default=FDHO_SEEDS[0],
                        help="seed shown in the reconstruction and mask figures")
    parser.add_argument("--image", default=None,
                        help="image path (default: the one stored in run_config.yaml)")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("analyze.py needs a CUDA GPU (the kaolin-wisp hash grid is CUDA-only)")
    os.makedirs(args.out_dir, exist_ok=True)

    runs = [load_run(d) for d in args.run_dirs]
    if len({r["name"] for r in runs}) != len(runs):
        raise ValueError("run directories must have distinct names")
    size = runs[0]["cfg"]["data"]["size"]
    names = {os.path.basename(r["cfg"]["image"]) for r in runs}
    if len(names) != 1 or any(r["cfg"]["data"]["size"] != size for r in runs):
        raise ValueError("all runs must fit the same image at the same size")
    image = args.image or runs[0]["cfg"]["image"]
    missing = [r["name"] for r in runs if args.seed not in r["seeds"]]
    if missing:
        raise ValueError(f"seed {args.seed} is missing from {missing}")

    print(write_summary(runs, os.path.join(args.out_dir, "summary.md")))
    fig_curves(runs, os.path.join(args.out_dir, "training_curves.png"))

    device = torch.device("cuda")
    coords, gt, shape = load_image(image, size)
    gt = ((gt.reshape(*shape, 3) + 1) / 2).clamp(0, 1).numpy()
    preds, recs = {}, {}
    for r in runs:
        for s in r["seeds"]:
            rec = reconstruct(r, s, coords, shape, device)
            preds[(r["name"], s)] = rec["pred"]
            if s == args.seed:
                recs[r["name"]] = rec

    crops = TIGER_CROPS if shape == TIGER_SHAPE else {}
    fig_reconstruction(gt, recs, args.seed, crops, os.path.join(args.out_dir, "reconstruction.png"))
    fig_error_maps(gt, runs, preds, os.path.join(args.out_dir, "error_maps.png"))
    for name, rec in recs.items():
        if rec["hidden_masks"]:
            fig_masks(name, rec, args.seed, os.path.join(args.out_dir, f"masks_{name}.png"))
    print(f"figures written to {args.out_dir}")


if __name__ == "__main__":
    main()

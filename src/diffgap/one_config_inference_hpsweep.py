from __future__ import annotations

import os
import argparse
import random
from typing import Dict, Optional, Literal, Tuple, Callable, List

import cv2
import torch.nn.functional as F

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from accelerate import Accelerator
from accelerate.utils import set_seed

from diffusers import DDPMScheduler, DPMSolverMultistepScheduler

from joblib import load as joblib_load
import pandas as pd

from diffgap import (
    ConditionalDiffusionSampler,
    ConditionalInpaintingDataset,
    compare_methods_fixed,
    inpaint_opencv_batch,
    linear_interp2d_batch,
    denoise_batch,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import contextlib
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# -------------------------------------------------------------------
# 1) Helper functions / dataset classes
# -------------------------------------------------------------------

def load_data_dict(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    data = joblib_load(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from {path}, got {type(data)}")
    return data


class CityConditionalDataset(Dataset):
    def __init__(self, data_dict):
        self.samples = []
        self.H = None
        self.W = None

        for city, value in data_dict.items():
            images_np, cond_np, elv_np = value
            images_np = np.asarray(images_np, dtype=np.float32)

            N, H, W = images_np.shape

            cond_np = np.asarray(cond_np, dtype=np.float32)
            elv_np = np.asarray(elv_np, dtype=np.float32)

            # --- cond shapes ---
            if cond_np.ndim == 2:
                cond_np = np.broadcast_to(cond_np, (N, H, W))
            elif cond_np.ndim == 3 and cond_np.shape == (1, H, W):
                cond_np = np.broadcast_to(cond_np[0], (N, H, W))

            # --- elv shapes ---
            if elv_np.ndim == 2:
                elv_np = np.broadcast_to(elv_np, (N, H, W))
            elif elv_np.ndim == 3 and elv_np.shape == (1, H, W):
                elv_np = np.broadcast_to(elv_np[0], (N, H, W))

            if self.H is None:
                self.H, self.W = H, W

            for i in range(N):
                self.samples.append((images_np[i], cond_np[i], elv_np[i], city))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_np, cond_np, elv_np, city = self.samples[idx]
        return (
            torch.from_numpy(img_np).float().unsqueeze(0),
            torch.from_numpy(cond_np).float().unsqueeze(0),
            torch.from_numpy(elv_np).float().unsqueeze(0),
            city,
        )


def mix(diff_reconstructed, ground_truth, ground_revealed, masks, coef=1.0):
    diff_reconstructed = diff_reconstructed.detach().cpu()
    ground_truth = ground_truth.cpu()
    ground_revealed = ground_revealed.cpu()
    mask = masks.cpu()

    complete_prediction = diff_reconstructed * (1 - mask) + ground_revealed

    baseline = inpaint_opencv_batch(ground_revealed, 1 - mask, radius=20, method="telea")
    baseline_lr = linear_interp2d_batch(ground_revealed, 1 - mask)

    mixed = (coef * diff_reconstructed + (1 - coef) * baseline) * (1 - mask) + ground_revealed

    complete_prediction = denoise_batch(complete_prediction)
    ground_truth = denoise_batch(ground_truth)
    baseline = denoise_batch(baseline)
    baseline_lr = denoise_batch(baseline_lr)
    mixed = denoise_batch(mixed)

    return {
        "raw_diffusion_prediction": complete_prediction,
        "baseline_CV": baseline,
        "baseline_linear": baseline_lr,
        "mixed_prediction": mixed,
        "ground_truth": ground_truth,
        "ground_revealed": ground_revealed,
        "mask": mask,
    }


# -------------------------------------------------------------------
# 2) Main evaluation function with Accelerate
# -------------------------------------------------------------------

def run_one_config(
    data_path: str,
    ckpt: str,
    output_dir: str,
    seed: int,
    N_test: int,
    cloud_coverage: float,
    octaves: int,
    batch_size: int,
    N_samples: int,
    wind_degree: int,
    timesteps: int,
    grad_steps: int,
    guidance_stride: int,
):
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}")

    set_seed(seed)

    # Load and flatten dataset
    data_dict = load_data_dict(data_path)
    dataset = CityConditionalDataset(data_dict)
    H, W = dataset.H, dataset.W

    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    _, test_ds = random_split(dataset, [train_len, val_len], generator=generator)

    base_test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    imgs, conds, elevs, cities = [], [], [], []
    for img, cond, elv, city in base_test_loader:
        imgs.append(img)
        conds.append(cond)
        elevs.append(elv)
        cities.extend(list(city))

    test_images = torch.cat(imgs, dim=0)[:N_test]
    test_conds = torch.cat(conds, dim=0)[:N_test]
    test_elevs = torch.cat(elevs, dim=0)[:N_test]
    cities = cities[:N_test]

    test_data_np = test_images.squeeze(1).numpy()
    test_conds_np = test_conds.squeeze(1).numpy()
    test_elevs_np = test_elevs.squeeze(1).numpy()

    # Schedulers
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=1e-4,
        beta_end=2e-2,
        beta_schedule="linear",
    )
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        trained_betas=noise_scheduler.betas.detach().cpu().numpy(),
        prediction_type="epsilon",
        solver_order=3,
        use_karras_sigmas=True,
        timestep_spacing="trailing",
    )

    cloudy = ConditionalInpaintingDataset(
        test_data_np,
        test_conds_np,
        test_elevs_np,
        cities,
        mask_type="cloud_fbm",
        mask_kwargs=dict(
            coverage=cloud_coverage,
            style="cirrus",
            octaves=octaves,
            persistence=cloud_coverage,
            lacunarity=2.0,
            streak=2.0,
            wind_dir_deg=wind_degree,
            morph_puff=0.1,
        ),
    )
    cloudy_loader = DataLoader(cloudy, batch_size=batch_size, shuffle=False, num_workers=4)

    # Load UNet
    from diffusers import UNet2DModel
    unet = UNet2DModel(
        sample_size=H,
        in_channels=3,
        out_channels=1,
        block_out_channels=(64, 128, 256, 256),
        layers_per_block=2,
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
        dropout=0.1,
    )
    unet.load_state_dict(torch.load(ckpt, map_location="cpu"))
    unet.to(device)

    unet, cloudy_loader = accelerator.prepare(unet, cloudy_loader)
    sampler = ConditionalDiffusionSampler(unet, scheduler)

    # ---- NEW: tail guidance (15% of timesteps) ----
    tail_guidance = max(1, int(0.15 * timesteps))
    accelerator.print(f"Using tail_guidance = {tail_guidance}")

    # Sampling
    all_samples = []
    for i in range(N_samples):
        accelerator.print(f"Sampling {i+1}/{N_samples}")
        preds = []

        for batch in cloudy_loader:
            img = batch["image"].to(device)
            rev = batch["revealed"].to(device)
            msk = batch["mask"].to(device)
            cnd = batch["cond"].to(device)
            elv = batch["elevations"].to(device)

            s_tmp, _ = sampler.sample_copaint(
                rev,
                msk,
                cnd,
                elv,
                timesteps=timesteps,
                learning_rate=10.0,
                grad_steps=grad_steps,
                guidance_stride=guidance_stride,
                tail_guidance=tail_guidance,
                do_projection=True,
            )

            s_tmp = accelerator.gather(s_tmp)
            if accelerator.is_main_process:
                preds.append(s_tmp.cpu())

        if accelerator.is_main_process:
            out = torch.cat(preds, dim=0)[:N_test]
            all_samples.append(out)

    # Metrics
    if accelerator.is_main_process:
        stacked = torch.stack(all_samples, dim=0)
        s_mean = stacked.mean(dim=0)

        # rebuild test set
        imgs, rvld, masks, cnds = [], [], [], []
        for batch in DataLoader(cloudy, batch_size=batch_size):
            imgs.append(batch["image"])
            rvld.append(batch["revealed"])
            masks.append(batch["mask"])
            cnds.append(batch["cond"])

        gt = torch.cat(imgs, dim=0)[:N_test]
        rv = torch.cat(rvld, dim=0)[:N_test]
        mk = torch.cat(masks, dim=0)[:N_test]

        urban_mask = torch.zeros_like(gt)
        urban_mask[:, :, 25:80, 40:100] = 1

        pr = mix(s_mean, gt, rv, mk, coef=0.6)

        tbl = compare_methods_fixed(
            truth=pr["ground_truth"],
            predictions={
                "diffusion": pr["mixed_prediction"],
                "linear": pr["baseline_linear"],
            },
            mask=1 - mk,
            region="hole",
            norm="-1_1",
            custom_minmax=(262.7, 333.86),
            urban_mask=urban_mask,
            uhi_phys_minmax=(262.7, 333.86),
            uhi_mode="smape",
            uhi_min_denom_k=0.5,
        )

        row_diff = tbl[tbl["method"] == "diffusion"].iloc[0]
        row_lin = tbl[tbl["method"] == "linear"].iloc[0]

        metrics = [
            "RMSE_mean","RMSE_std",
            "PSNR_mean","PSNR_std",
            "SSIM_mean","SSIM_std",
            "UHI_error_mean","UHI_error_std",
            "R2_mean","R2_std"
        ]

        out = {
            "N": int(row_diff["N"]),
            "region": row_diff["region"],
            "UHI_error_mode": row_diff["UHI_error_mode"],
            "method_diff": row_diff["method"],
            "method_linear": row_lin["method"],
            "cloud_coverage": cloud_coverage,
            "octaves": octaves,
            "wind_degree": wind_degree,
            "N_test": N_test,
            "N_samples": N_samples,
            "timesteps": timesteps,
            "grad_steps": grad_steps,
            "guidance_stride": guidance_stride,
            "tail_guidance": tail_guidance,  # saved but not user-controlled
            "seed": seed,
        }

        for m in metrics:
            out[f"diff_{m}"] = row_diff[m]
            out[f"linear_{m}"] = row_lin[m]

        out_df = pd.DataFrame([out])
        os.makedirs(output_dir, exist_ok=True)
        out_name = f"metrics_cov{cloud_coverage}_oct{octaves}_wind{wind_degree}.csv"
        out_df.to_csv(os.path.join(output_dir, out_name), index=False)
        accelerator.print(f"Saved CSV → {out_name}")

    accelerator.wait_for_everyone()


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--output_dir", type=str, default="./metrics_out")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--N_test", type=int, default=400)
    p.add_argument("--cloud_coverage", type=float, required=True)
    p.add_argument("--octaves", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--N_samples", type=int, default=10)
    p.add_argument("--wind_degree", type=int, required=True)

    # diffusion sampling params
    p.add_argument("--timesteps", type=int, default=100)
    p.add_argument("--grad_steps", type=int, default=1)
    p.add_argument("--guidance_stride", type=int, default=2)

    return p.parse_args()


def main():
    args = parse_args()
    run_one_config(
        data_path=args.data_path,
        ckpt=args.ckpt,
        output_dir=args.output_dir,
        seed=args.seed,
        N_test=args.N_test,
        cloud_coverage=args.cloud_coverage,
        octaves=args.octaves,
        batch_size=args.batch_size,
        N_samples=args.N_samples,
        wind_degree=args.wind_degree,
        timesteps=args.timesteps,
        grad_steps=args.grad_steps,
        guidance_stride=args.guidance_stride,
    )


if __name__ == "__main__":
    main()

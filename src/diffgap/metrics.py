from __future__ import annotations
from typing import Dict, Optional, Literal
import math
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import pandas as pd
from .utils import apply_region, data_range_from_normalization

def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x - y) ** 2))

def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-20:
        return torch.tensor(float("inf"), device=x.device)
    return 20.0 * torch.log10(torch.tensor(data_range, device=x.device) / torch.sqrt(mse))

def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    # shapes: (N,1,H,W)
    ssim_metric = SSIM(data_range=data_range)
    return ssim_metric(x, y)

def _mean_std_no_nan(t: torch.Tensor):
    t = t.detach()
    finite = torch.isfinite(t)
    if not finite.any():
        return float('nan'), float('nan')
    vals = t[finite]
    return float(vals.mean().cpu()), float(vals.std(unbiased=False).cpu())

@torch.no_grad()
def compare_methods_fixed(
    truth: torch.Tensor,
    predictions: dict,
    mask: torch.Tensor | None = None,
    region: str = "hole",
    norm: str = "0_1",
    custom_minmax = (0.0, 1.0),
):
    from diffgap.utils import apply_region, data_range_from_normalization
    data_range = data_range_from_normalization(norm, *custom_minmax)

    rows = []
    ssim_metric = SSIM(data_range=data_range).to(truth.device)

    for name, pred in predictions.items():
        x, y = pred.clone(), truth.clone()
        if region != "all" and mask is not None:
            x = apply_region(mask, region, x)
            y = apply_region(mask, region, y)

        # per-image RMSE
        def _rmse_per_img(a,b):
            # compute over all pixels; if you're truly masking a region, see note below
            return torch.sqrt(torch.mean((a-b)**2, dim=(1,2,3)))
        per_rmse = _rmse_per_img(x, y)

        # per-image PSNR
        def _psnr(one, two):
            mse = torch.mean((one-two)**2)
            if mse <= 1e-20:
                return torch.tensor(float("inf"), device=one.device)
            return 20.0*torch.log10(torch.tensor(data_range, device=one.device)/torch.sqrt(mse))
        per_psnr = torch.stack([_psnr(x[i:i+1], y[i:i+1]) for i in range(x.size(0))]).squeeze()

        # per-image SSIM
        per_ssim = torch.stack([ssim_metric(x[i:i+1], y[i:i+1]) for i in range(x.size(0))]).squeeze()

        RMSE_mean, RMSE_std = _mean_std_no_nan(per_rmse)
        PSNR_mean, PSNR_std = _mean_std_no_nan(per_psnr)
        SSIM_mean, SSIM_std = _mean_std_no_nan(per_ssim)

        rows.append(dict(
            method=name,
            RMSE_mean=RMSE_mean, RMSE_std=RMSE_std,
            PSNR_mean=PSNR_mean, PSNR_std=PSNR_std,
            SSIM_mean=SSIM_mean, SSIM_std=SSIM_std,
            N=int(x.size(0)),
            region=region,
        ))

    return pd.DataFrame(rows, columns=[
        "method","RMSE_mean","RMSE_std","PSNR_mean","PSNR_std","SSIM_mean","SSIM_std","N","region"
    ])

# Monkey-patch
M.compare_methods = compare_methods_fixed

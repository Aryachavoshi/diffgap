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

@torch.no_grad()
def compare_methods(
    truth: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    region: Literal["all","revealed","hole"] = "hole",
    norm: Literal["0_1","-1_1","custom"] = "0_1",
    custom_minmax = (0.0, 1.0),
) -> pd.DataFrame:
    """
    Computes RMSE, PSNR, SSIM for multiple methods.
    - truth, predictions[*]: (N,1,H,W)
    - mask: 1=revealed (keep), 0=hole
    - region: which pixels to evaluate on
    - norm: sets data_range used by PSNR/SSIM
    """
    data_range = data_range_from_normalization(norm, *custom_minmax)

    rows = []
    for name, pred in predictions.items():
        x = pred.clone()
        y = truth.clone()
        if region != "all" and mask is not None:
            x = apply_region(mask, region, x)
            y = apply_region(mask, region, y)

        # reduce over pixels, keep per-image values first
        def _rmse_per_img(a,b):
            # avoid division by zero: if region has zero valid pixels, skip
            valid = (a==a) & (b==b)
            if valid.sum(dim=(1,2,3)).min() == 0:
                return torch.full((a.size(0),), float("nan"), device=a.device)
            return torch.sqrt(torch.mean((a-b)**2, dim=(1,2,3)))
        per_rmse = _rmse_per_img(x, y)
        per_psnr = torch.stack([psnr(x[i:i+1], y[i:i+1], data_range=data_range) for i in range(x.size(0))]).squeeze()
        # SSIM expects values in [0, data_range]: we assume your inputs are already normalized to that range
        ssim_metric = SSIM(data_range=data_range)
        per_ssim = torch.stack([ssim_metric(x[i:i+1], y[i:i+1]) for i in range(x.size(0))]).squeeze()

        row = dict(
            method=name,
            RMSE_mean=float(torch.nanmean(per_rmse).cpu()),
            RMSE_std=float(torch.nanstd(per_rmse).cpu()),
            PSNR_mean=float(torch.nanmean(per_psnr).cpu()),
            PSNR_std=float(torch.nanstd(per_psnr).cpu()),
            SSIM_mean=float(torch.nanmean(per_ssim).cpu()),
            SSIM_std=float(torch.nanstd(per_ssim).cpu()),
            N=int(x.size(0)),
            region=region
        )
        rows.append(row)

    return pd.DataFrame(rows, columns=[
        "method","RMSE_mean","RMSE_std","PSNR_mean","PSNR_std","SSIM_mean","SSIM_std","N","region"
    ])

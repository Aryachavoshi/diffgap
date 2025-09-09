from __future__ import annotations
from typing import Dict, Optional, Literal, Tuple
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
    ssim_metric = SSIM(data_range=data_range)
    return ssim_metric(x, y)

def _mean_std_no_nan(t: torch.Tensor):
    t = t.detach()
    finite = torch.isfinite(t)
    if not finite.any():
        return float('nan'), float('nan')
    vals = t[finite]
    return float(vals.mean().cpu()), float(vals.std(unbiased=False).cpu())

def _unnormalize(
    x: torch.Tensor,
    norm: str,
    phys_minmax: Optional[Tuple[float,float]],
    custom_minmax: Tuple[float,float]
) -> torch.Tensor:
    """
    Map normalized tensor to physical units if phys_minmax is provided.
    Supported norms:
      - "0_1": x in [0,1] -> x*(max-min)+min
      - "-1_1": x in [-1,1] -> ((x+1)/2)*(max-min)+min
    Otherwise returns x unchanged.
    """
    if phys_minmax is None:
        return x
    lo, hi = phys_minmax
    if norm == "0_1":
        return x * (hi - lo) + lo
    elif norm in ("-1_1", "neg1_1"):
        return ( (x + 1.0) * 0.5 ) * (hi - lo) + lo
    else:
        # assume already physical if using z-score or other scheme
        return x

def _compute_uhi(batch: torch.Tensor, urban_mask: torch.Tensor) -> torch.Tensor:
    """
    batch: (N,1,H,W) in physical units if desired
    urban_mask: (1,1,H,W) or (N,1,H,W); 1 for urban, 0 for rural
    returns per-image UHI = mean(urban) - mean(rural), shape (N,)
    """
    # Broadcast mask if given as (1,1,H,W)
    if urban_mask.dim() == 4 and urban_mask.size(0) == 1:
        m = urban_mask.expand(batch.size(0), -1, -1, -1)
    else:
        m = urban_mask
    m = m.to(batch.device, batch.dtype)

    urban = torch.where(m > 0.5, batch, torch.tensor(float('nan'), device=batch.device, dtype=batch.dtype))
    rural = torch.where(m <= 0.5, batch, torch.tensor(float('nan'), device=batch.device, dtype=batch.dtype))

    # per-image nanmeans
    urb_mean = torch.nanmean(urban.view(batch.size(0), -1), dim=1)
    rur_mean = torch.nanmean(rural.view(batch.size(0), -1), dim=1)
    return urb_mean - rur_mean  # (N,)

@torch.no_grad()
def compare_methods_fixed(
    truth: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    region: str = "hole",
    norm: str = "0_1",
    custom_minmax = (0.0, 1.0),
    *,
    # --- NEW: UHI options ---
    urban_mask: Optional[torch.Tensor] = None,        # e.g., a binary mask of shape (1,1,H,W) or (N,1,H,W)
    uhi_phys_minmax: Optional[Tuple[float,float]] = (262.7, 333.86),  # set None to skip un-normalization
    uhi_as_percent: bool = True,                      # compute % error vs |truth UHI|
    uhi_eps: float = 1e-12                            # guard against division by zero
) -> pd.DataFrame:
    """
    Returns a DataFrame with RMSE/PSNR/SSIM and (optionally) UHI % error stats per method.
    """
    from diffgap.utils import apply_region, data_range_from_normalization
    data_range = data_range_from_normalization(norm, *custom_minmax)

    rows = []
    ssim_metric = SSIM(data_range=data_range).to(truth.device)

    # Precompute truth UHI if requested
    if urban_mask is not None:
        truth_phys = _unnormalize(truth, norm=norm, phys_minmax=uhi_phys_minmax, custom_minmax=custom_minmax)
        truth_uhi = _compute_uhi(truth_phys, urban_mask)  # (N,)
    else:
        truth_uhi = None

    for name, pred in predictions.items():
        x, y = pred.clone(), truth.clone()
        if region != "all" and mask is not None:
            x = apply_region(mask, region, x)
            y = apply_region(mask, region, y)

        # -------- Pixel-wise metrics (per-image) --------
        per_rmse = torch.sqrt(torch.mean((x - y) ** 2, dim=(1, 2, 3)))

        def _psnr(one, two):
            mse = torch.mean((one - two) ** 2)
            if mse <= 1e-20:
                return torch.tensor(float("inf"), device=one.device)
            return 20.0 * torch.log10(torch.tensor(data_range, device=one.device) / torch.sqrt(mse))

        per_psnr = torch.stack([_psnr(x[i:i+1], y[i:i+1]) for i in range(x.size(0))]).squeeze()
        per_ssim = torch.stack([ssim_metric(x[i:i+1], y[i:i+1]) for i in range(x.size(0))]).squeeze()

        RMSE_mean, RMSE_std = _mean_std_no_nan(per_rmse)
        PSNR_mean, PSNR_std = _mean_std_no_nan(per_psnr)
        SSIM_mean, SSIM_std = _mean_std_no_nan(per_ssim)

        # -------- NEW: UHI error (per-image) --------
        if truth_uhi is not None:
            pred_phys = _unnormalize(x, norm=norm, phys_minmax=uhi_phys_minmax, custom_minmax=custom_minmax)
            pred_uhi = _compute_uhi(pred_phys, urban_mask)  # (N,)

            if uhi_as_percent:
                denom = torch.clamp(torch.abs(truth_uhi), min=uhi_eps)
                per_uhi_err = (pred_uhi - truth_uhi) * 100.0 / denom
            else:
                per_uhi_err = (pred_uhi - truth_uhi)  # absolute error in physical units

            UHI_err_mean, UHI_err_std = _mean_std_no_nan(per_uhi_err)
        else:
            UHI_err_mean, UHI_err_std = float('nan'), float('nan')

        rows.append(dict(
            method=name,
            RMSE_mean=RMSE_mean, RMSE_std=RMSE_std,
            PSNR_mean=PSNR_mean, PSNR_std=PSNR_std,
            SSIM_mean=SSIM_mean, SSIM_std=SSIM_std,
            UHI_pct_error_mean=UHI_err_mean if uhi_as_percent else float('nan'),
            UHI_pct_error_std=UHI_err_std if uhi_as_percent else float('nan'),
            # If you choose absolute error, you can also add columns for that case.
            N=int(x.size(0)),
            region=region,
        ))

    cols = [
        "method","RMSE_mean","RMSE_std","PSNR_mean","PSNR_std","SSIM_mean","SSIM_std",
        "UHI_pct_error_mean","UHI_pct_error_std","N","region"
    ]
    return pd.DataFrame(rows, columns=cols)

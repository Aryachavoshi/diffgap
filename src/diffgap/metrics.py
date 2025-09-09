
from __future__ import annotations
from typing import Dict, Optional, Literal, Tuple
import torch
import pandas as pd
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


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



@torch.no_grad()
def compare_methods_fixed(
    truth: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    region: str = "hole",
    norm: str = "0_1",
    custom_minmax = (0.0, 1.0),
    *,
    urban_mask: Optional[torch.Tensor] = None,
    uhi_phys_minmax: Optional[Tuple[float,float]] = (262.7, 333.86),
    # robust UHI error options:
    uhi_mode: Literal["pct","abs","smape"] = "pct",
    uhi_min_denom_k: float = 0.5,   # floor for |truth UHI| in K for pct/smape
) -> pd.DataFrame:
    """
    Pixel metrics computed on `region` (e.g., 'hole' or 'all').
    UHI metrics computed on FULL frames (unmasked), independent of `region`.
    """
    from diffgap.utils import apply_region, data_range_from_normalization
    data_range = data_range_from_normalization(norm, *custom_minmax)

    rows = []
    ssim_metric = SSIM(data_range=data_range).to(truth.device)

    # ---- Prepare full-frame truth for UHI (unnormalize inline if requested) ----
    if urban_mask is not None:
        if uhi_phys_minmax is not None:
            lo, hi = uhi_phys_minmax
            if norm == "0_1":
                truth_full_phys = truth * (hi - lo) + lo
            elif norm in ("-1_1", "neg1_1"):
                truth_full_phys = ((truth + 1.0) * 0.5) * (hi - lo) + lo
            else:
                truth_full_phys = truth  # assume already physical units
        else:
            truth_full_phys = truth  # already in physical units

        # Broadcast urban mask to batch if needed
        if urban_mask.dim() == 4 and urban_mask.size(0) == 1:
            um = urban_mask.to(truth.device, truth.dtype).expand(truth.size(0), -1, -1, -1)
        else:
            um = urban_mask.to(truth.device, truth.dtype)

        # Compute truth UHI per image: mean(urban) - mean(rural)
        truth_urban = torch.where(um > 0.5, truth_full_phys, torch.tensor(float('nan'), device=truth.device, dtype=truth.dtype))
        truth_rural = torch.where(um <= 0.5, truth_full_phys, torch.tensor(float('nan'), device=truth.device, dtype=truth.dtype))
        truth_uhi = torch.nanmean(truth_urban.view(truth.size(0), -1), dim=1) - \
                    torch.nanmean(truth_rural.view(truth.size(0), -1), dim=1)
    else:
        truth_uhi = None  # no UHI metrics

    for name, pred_full in predictions.items():
        # ------------- Pixel metrics on region -------------
        x_reg, y_reg = pred_full.clone(), truth.clone()
        if region != "all" and mask is not None:
            x_reg = apply_region(mask, region, x_reg)
            y_reg = apply_region(mask, region, y_reg)

        per_rmse = torch.sqrt(torch.mean((x_reg - y_reg) ** 2, dim=(1,2,3)))

        def _psnr(a, b):
            mse = torch.mean((a - b) ** 2)
            if mse <= 1e-20:
                return torch.tensor(float("inf"), device=a.device)
            return 20.0 * torch.log10(torch.tensor(data_range, device=a.device) / torch.sqrt(mse))

        per_psnr = torch.stack([_psnr(x_reg[i:i+1], y_reg[i:i+1]) for i in range(x_reg.size(0))]).squeeze()
        per_ssim = torch.stack([ssim_metric(x_reg[i:i+1], y_reg[i:i+1]) for i in range(x_reg.size(0))]).squeeze()

        RMSE_mean, RMSE_std = _mean_std_no_nan(per_rmse)
        PSNR_mean, PSNR_std = _mean_std_no_nan(per_psnr)
        SSIM_mean, SSIM_std = _mean_std_no_nan(per_ssim)

        # ------------- UHI on FULL frames (not region-masked) -------------
        if truth_uhi is not None:
            # Unnormalize prediction inline if requested
            if uhi_phys_minmax is not None:
                lo, hi = uhi_phys_minmax
                if norm == "0_1":
                    pred_full_phys = pred_full * (hi - lo) + lo
                elif norm in ("-1_1", "neg1_1"):
                    pred_full_phys = ((pred_full + 1.0) * 0.5) * (hi - lo) + lo
                else:
                    pred_full_phys = pred_full
            else:
                pred_full_phys = pred_full

            # Broadcast mask like above
            if urban_mask.dim() == 4 and urban_mask.size(0) == 1:
                um = urban_mask.to(pred_full.device, pred_full.dtype).expand(pred_full.size(0), -1, -1, -1)
            else:
                um = urban_mask.to(pred_full.device, pred_full.dtype)

            pred_urban = torch.where(um > 0.5, pred_full_phys, torch.tensor(float('nan'), device=pred_full.device, dtype=pred_full.dtype))
            pred_rural = torch.where(um <= 0.5, pred_full_phys, torch.tensor(float('nan'), device=pred_full.device, dtype=pred_full.dtype))
            pred_uhi = torch.nanmean(pred_urban.view(pred_full.size(0), -1), dim=1) - \
                       torch.nanmean(pred_rural.view(pred_full.size(0), -1), dim=1)

            if uhi_mode == "pct":
                denom = torch.clamp(torch.abs(truth_uhi), min=uhi_min_denom_k)  # K
                per_uhi_err = (pred_uhi - truth_uhi) * 100.0 / denom
            elif uhi_mode == "smape":
                denom = torch.clamp(torch.abs(pred_uhi) + torch.abs(truth_uhi), min=2*uhi_min_denom_k)
                per_uhi_err = 200.0 * torch.abs(pred_uhi - truth_uhi) / denom
            else:  # "abs": error in K
                per_uhi_err = (pred_uhi - truth_uhi)

            UHI_err_mean, UHI_err_std = _mean_std_no_nan(per_uhi_err)
        else:
            UHI_err_mean, UHI_err_std = float('nan'), float('nan')

        rows.append(dict(
            method=name,
            RMSE_mean=RMSE_mean, RMSE_std=RMSE_std,
            PSNR_mean=PSNR_mean, PSNR_std=PSNR_std,
            SSIM_mean=SSIM_mean, SSIM_std=SSIM_std,
            UHI_error_mean=UHI_err_mean,   # units depend on uhi_mode
            UHI_error_std=UHI_err_std,
            UHI_error_mode=uhi_mode,
            N=int(pred_full.size(0)),
            region=region,
        ))

    return pd.DataFrame(rows, columns=[
        "method","RMSE_mean","RMSE_std","PSNR_mean","PSNR_std","SSIM_mean","SSIM_std",
        "UHI_error_mean","UHI_error_std","UHI_error_mode","N","region"
    ])

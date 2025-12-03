
from __future__ import annotations
from typing import Dict, Optional, Literal, Tuple
import torch
import pandas as pd
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM


def rmse(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((x - y) ** 2))

def psnr(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
    mse = torch.mean((x - y) ** 2)
    if mse <= 1e-20:
        return torch.tensor(float("inf"), device=x.device)
    return 20.0 * torch.log10(torch.tensor(data_range, device=x.device) / torch.sqrt(mse))

def ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0) -> torch.Tensor:
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
    uhi_phys_minmax: Optional[Tuple[float, float]] = (262.7, 333.86),
    uhi_mode: Literal["pct", "abs", "smape"] = "pct",
    uhi_min_denom_k: float = 0.5,
) -> pd.DataFrame:
    """
    Pixel metrics:
      - RMSE (in *physical* space, e.g., Kelvin, if custom_minmax & norm given)
      - PSNR, SSIM, R2 (using appropriate data_range / physical units)
    UHI metrics computed on FULL frames (unmasked), independent of `region`.
    """
    from diffgap.utils import apply_region

    # Decide PSNR/SSIM data_range
    if custom_minmax is not None:
        lo_cm, hi_cm = custom_minmax
        data_range = float(hi_cm - lo_cm)
    else:
        lo_cm, hi_cm = 0.0, 1.0
        if norm == "0_1":
            data_range = 1.0
        elif norm in ("-1_1", "neg1_1"):
            data_range = 2.0
        else:
            tr = truth.detach()
            finite = torch.isfinite(tr)
            if finite.any():
                trv = tr[finite]
                data_range = float((trv.max() - trv.min()).cpu())
            else:
                data_range = 1.0

    rows = []
    ssim_metric = SSIM(data_range=data_range).to(truth.device)

    # Prepare full-frame truth for UHI
    if urban_mask is not None:
        if uhi_phys_minmax is not None:
            lo_uhi, hi_uhi = uhi_phys_minmax
            if norm == "0_1":
                truth_full_phys = truth * (hi_uhi - lo_uhi) + lo_uhi
            elif norm in ("-1_1", "neg1_1"):
                truth_full_phys = ((truth + 1.0) * 0.5) * (hi_uhi - lo_uhi) + lo_uhi
            else:
                truth_full_phys = truth
        else:
            truth_full_phys = truth

        if urban_mask.dim() == 4 and urban_mask.size(0) == 1:
            um = urban_mask.to(truth.device, truth.dtype).expand(truth.size(0), -1, -1, -1)
        else:
            um = urban_mask.to(truth.device, truth.dtype)

        urb_bin = (um > 0.5)
        rur_bin = ~urb_bin

        valid_truth = torch.isfinite(truth_full_phys)
        urb_count = (urb_bin & valid_truth).sum(dim=(1, 2, 3)).clamp(min=1)
        rur_count = (rur_bin & valid_truth).sum(dim=(1, 2, 3)).clamp(min=1)

        truth_safe = torch.nan_to_num(truth_full_phys, nan=0.0, posinf=0.0, neginf=0.0)
        urb_sum = (truth_safe * urb_bin).sum(dim=(1, 2, 3))
        rur_sum = (truth_safe * rur_bin).sum(dim=(1, 2, 3))

        truth_uhi = (urb_sum / urb_count) - (rur_sum / rur_count)   # (N,)
    else:
        truth_uhi = None
        um = None  # for type checkers

    for name, pred_full in predictions.items():
        # ------------- Pixel region selection (normalized space) -------------
        x_reg = pred_full.clone()
        y_reg = truth.clone()
        if region != "all" and mask is not None:
            x_reg = apply_region(mask, region, x_reg)
            y_reg = apply_region(mask, region, y_reg)

        # ------------- Unnormalize to physical space for pixel metrics -------------
        if custom_minmax is not None:
            lo, hi = custom_minmax
            if norm == "0_1":
                x_phys = x_reg * (hi - lo) + lo
                y_phys = y_reg * (hi - lo) + lo
            elif norm in ("-1_1", "neg1_1"):
                x_phys = ((x_reg + 1.0) * 0.5) * (hi - lo) + lo
                y_phys = ((y_reg + 1.0) * 0.5) * (hi - lo) + lo
            else:
                x_phys, y_phys = x_reg, y_reg
        else:
            x_phys, y_phys = x_reg, y_reg

        # RMSE in physical units
        per_rmse = torch.sqrt(torch.mean((x_phys - y_phys) ** 2, dim=(1, 2, 3)))

        # PSNR
        def _psnr(a, b):
            mse = torch.mean((a - b) ** 2)
            if mse <= 1e-20:
                return torch.tensor(float("inf"), device=a.device)
            return 20.0 * torch.log10(torch.tensor(data_range, device=a.device) / torch.sqrt(mse))

        per_psnr = torch.stack([_psnr(x_phys[i:i+1], y_phys[i:i+1]) for i in range(x_phys.size(0))]).squeeze()

        # SSIM
        per_ssim = torch.stack([ssim_metric(x_phys[i:i+1], y_phys[i:i+1]) for i in range(x_phys.size(0))]).squeeze()

        # R² in physical space (per image)
        per_r2_list = []
        for i in range(x_phys.size(0)):
            yt = y_phys[i].view(-1)
            yp = x_phys[i].view(-1)
            valid = torch.isfinite(yt) & torch.isfinite(yp)
            if valid.sum() <= 1:
                r2_i = torch.tensor(float('nan'), device=x_phys.device)
            else:
                yt_v = yt[valid]
                yp_v = yp[valid]
                ss_res = torch.sum((yt_v - yp_v) ** 2)
                ss_tot = torch.sum((yt_v - yt_v.mean()) ** 2)
                if ss_tot <= 1e-20:
                    r2_i = torch.tensor(float('nan'), device=x_phys.device)
                else:
                    r2_i = 1.0 - ss_res / ss_tot
            per_r2_list.append(r2_i)
        per_r2 = torch.stack(per_r2_list)

        RMSE_mean, RMSE_std = _mean_std_no_nan(per_rmse)
        PSNR_mean, PSNR_std = _mean_std_no_nan(per_psnr)
        SSIM_mean, SSIM_std = _mean_std_no_nan(per_ssim)
        R2_mean,   R2_std   = _mean_std_no_nan(per_r2)

        # ------------- UHI on FULL frames -------------
        if truth_uhi is not None:
            if uhi_phys_minmax is not None:
                lo_uhi, hi_uhi = uhi_phys_minmax
                if norm == "0_1":
                    pred_full_phys = pred_full * (hi_uhi - lo_uhi) + lo_uhi
                elif norm in ("-1_1", "neg1_1"):
                    pred_full_phys = ((pred_full + 1.0) * 0.5) * (hi_uhi - lo_uhi) + lo_uhi
                else:
                    pred_full_phys = pred_full
            else:
                pred_full_phys = pred_full

            urb_bin = (um > 0.5)
            rur_bin = ~urb_bin
            valid_pred = torch.isfinite(pred_full_phys)

            urb_count_p = (urb_bin & valid_pred).sum(dim=(1, 2, 3)).clamp(min=1)
            rur_count_p = (rur_bin & valid_pred).sum(dim=(1, 2, 3)).clamp(min=1)

            pred_safe = torch.nan_to_num(pred_full_phys, nan=0.0, posinf=0.0, neginf=0.0)
            urb_sum_p = (pred_safe * urb_bin).sum(dim=(1, 2, 3))
            rur_sum_p = (pred_safe * rur_bin).sum(dim=(1, 2, 3))
            pred_uhi = (urb_sum_p / urb_count_p) - (rur_sum_p / rur_count_p)

            if uhi_mode == "pct":
                denom = torch.clamp(torch.abs(truth_uhi), min=uhi_min_denom_k)
                per_uhi_err = (pred_uhi - truth_uhi) * 100.0 / denom
            elif uhi_mode == "smape":
                denom = torch.clamp(torch.abs(pred_uhi) + torch.abs(truth_uhi), min=2 * uhi_min_denom_k)
                per_uhi_err = 200.0 * torch.abs(pred_uhi - truth_uhi) / denom
            else:  # "abs"
                per_uhi_err = (pred_uhi - truth_uhi)

            UHI_err_mean, UHI_err_std = _mean_std_no_nan(per_uhi_err)
        else:
            UHI_err_mean, UHI_err_std = float('nan'), float('nan')

        rows.append(dict(
            method=name,
            RMSE_mean=RMSE_mean, RMSE_std=RMSE_std,
            PSNR_mean=PSNR_mean, PSNR_std=PSNR_std,
            SSIM_mean=SSIM_mean, SSIM_std=SSIM_std,
            R2_mean=R2_mean, R2_std=R2_std,
            UHI_error_mean=UHI_err_mean,
            UHI_error_std=UHI_err_std,
            UHI_error_mode=uhi_mode,
            N=int(pred_full.size(0)),
            region=region,
        ))

    return pd.DataFrame(rows, columns=[
        "method",
        "RMSE_mean", "RMSE_std",
        "PSNR_mean", "PSNR_std",
        "SSIM_mean", "SSIM_std",
        "R2_mean", "R2_std",
        "UHI_error_mean", "UHI_error_std", "UHI_error_mode",
        "N", "region"
    ])

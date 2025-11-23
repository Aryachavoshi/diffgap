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
    inpaint_opencv_batch,
    linear_interp2d_batch,
    denoise_batch,
)
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
import contextlib
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class ConditionalDiffusionSampler(nn.Module):
    """
    Gradient-guided, post-conditioned sampling for inpainting (mask=1 are revealed pixels).
    Works with a trained ε-prediction model and a Diffusers scheduler (DDIM/DDPM/etc.)
    """
    def __init__(self, trained_model: nn.Module, scheduler) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = trained_model.to(self.device).eval()
        for p in self.trained_model.parameters():
            p.requires_grad_(False)

        self.scheduler = scheduler

        self.num_timesteps = int(scheduler.config.num_train_timesteps)  # <- fix deprecation
        # expect diffusers-style scheduler with .betas
        self.b_t  = scheduler.betas.to(self.device)
        self.a_t  = (1.0 - self.b_t)
        self.ab_t = torch.cumprod(self.a_t, dim=0).to(self.device)

    # Intentionally keep grads through x_t for inner optimization
    def eps_to_x0(self, x_t: torch.Tensor, t_idx: int, eps: torch.Tensor) -> torch.Tensor:
        ab = self.ab_t[t_idx]
        return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()

    def v_to_x0(self, x_t: torch.Tensor, t_idx: int, v: torch.Tensor) -> torch.Tensor:
        ab = self.ab_t[t_idx]
        s_a = ab.sqrt()
        s_o = (1 - ab).sqrt()
        # from diffusers: x0 = s_a * x_t - s_o * v
        return s_a * x_t - s_o * v


    @torch.no_grad()
    def project_revealed(self, x_t: torch.Tensor, x0_target: torch.Tensor, t_idx: int, mask: torch.Tensor) -> torch.Tensor:
        ab = self.ab_t[t_idx]
        mean = ab.sqrt() * x0_target
        std  = (1 - ab).sqrt()
        z = torch.randn_like(x_t)
        x_t_revealed = mean + std * z
        return mask * x_t_revealed + (1.0 - mask) * x_t

    def sample_copaint(
        self,
        revealed_image: torch.Tensor,
        masks: torch.Tensor,
        conds: torch.Tensor,
        elevs: torch.Tensor,
        timesteps: int = 20,
        grad_steps: int = 5,
        learning_rate: float = 0.1,
        use_amp: bool = True,
        do_projection: bool = True,
        return_intermediates_every: int = 5,
        guidance_stride: int = 4,
        tail_guidance: int = 5
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        revealed_image: (B,1,H,W) where zeros indicate holes, non-zero are revealed/keep
        """
        scheduler = self.scheduler
        scheduler.set_timesteps(timesteps, device=self.device)

        # Always drive the loop from the scheduler's actual timesteps
        schedule_ts = scheduler.timesteps.to(self.device)
        n_outer     = len(schedule_ts)
        if revealed_image.dim() == 3:
            revealed_image = revealed_image.unsqueeze(1)
        revealed_image = revealed_image.to(self.device)

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        masks = masks.to(self.device)
        if conds.dim() == 3:
            conds = conds.unsqueeze(1)
        conds = conds.to(self.device)
        if elevs.dim() == 3:
            elevs = elevs.unsqueeze(1)
        elevs = elevs.to(self.device)


        # ---- Initialize x_t ----
        # start from standard Gaussian
        x_t = torch.randn_like(revealed_image, device=self.device)


        with torch.no_grad():
                x_t = self.project_revealed(x_t, revealed_image, int(schedule_ts[0]), masks) ####

        def amp_ctx():
            if use_amp and self.device.type == "cuda":
                return torch.amp.autocast(device_type="cuda")
            return contextlib.nullcontext()

        intermediates: List[torch.Tensor] = []

        for i, t in enumerate(schedule_ts):
            t_idx_scalar = int(t.item())
            t_tensor = torch.full(
                (revealed_image.size(0),), t_idx_scalar, device=self.device, dtype=torch.long
            )
            print(f"=== Outer step {i+1}/{n_outer}, t={t_idx_scalar} ===")

            # sparse guidance: stride or last K steps
            in_tail = i >= max(0, n_outer - tail_guidance)
            on_stride = (i % max(1, guidance_stride) == 0)
            do_guidance_now = in_tail or on_stride

            x_t = x_t.detach().requires_grad_(True)
            if do_guidance_now and grad_steps > 0:
                for g in range(grad_steps):
                    with amp_ctx():
                        inp = torch.concat([x_t, conds, elevs], dim=1)
                        eps_pred = self.trained_model(inp, t_tensor).sample
                        x0_pred  = self.eps_to_x0(x_t, t_idx_scalar, eps_pred)
                        pred_revealed = x0_pred * masks
                        loss = ((pred_revealed - revealed_image) ** 2).sum() / (masks.sum() + 1e-8)
                    print(f"   Grad step {g+1}/{grad_steps} | loss={loss.item():.6f}")
                    loss.backward()

                    #print(f"   Learning rate: {learning_rate / (t_idx_scalar+1)}")
                    with torch.no_grad():
                        x_t -= (learning_rate) * x_t.grad     ############ new
                        x_t.grad = None
                    del eps_pred, x0_pred, pred_revealed, loss
            else:
                print("   (skip guidance this step)")

            # scheduler step x_t -> x_{t-1}
            with torch.no_grad(), amp_ctx():
                inp = torch.concat([x_t, conds, elevs], dim=1)
                eps_pred = self.trained_model(inp, t_tensor).sample
                step_out = scheduler.step(eps_pred, t, x_t)
                x_t = step_out.prev_sample
                del eps_pred

            # Optional projection: keep revealed region consistent with q(x_t|x0)
            if do_projection and do_guidance_now:
                with torch.no_grad():
                    x_t = self.project_revealed(x_t, revealed_image, t_idx_scalar, masks)

            if (i % max(1, return_intermediates_every) == 0):
                intermediates.append(x_t.detach().cpu().clone())
            print(f"   Scheduler step done for t={t_idx_scalar}")

        return x_t.detach(), intermediates

# -------------------------------------------------------------------
# 1) Helper functions / dataset classes
# -------------------------------------------------------------------

def load_data_dict(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Expected:
        data[city] = [images_np, cond_np, elv_np]

    images_np: (N, H, W), standardized
    cond_np, elv_np:
      - (H, W)        -> one map per city (broadcast to all images)
      - (1, H, W)     -> one map per city (broadcast to all images)
      - (N, H, W)     -> one map per image
    """
    data = joblib_load(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict from {path}, got {type(data)}")
    return data


class CityConditionalDataset(Dataset):
    """
    Flattens all cities into (img, cond, elevation, city_name) tuples.

    Returns:
      img:   (1, H, W)
      cond:  (1, H, W)
      elv:   (1, H, W)
      city:  string
    """
    def __init__(self, data_dict: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        self.samples = []  # list of (img_np, cond_np, elv_np, city_name)
        self.H = None
        self.W = None

        for city, value in data_dict.items():
            if not (isinstance(value, (list, tuple)) and len(value) >= 3):
                raise ValueError(f"data['{city}'] must be [images_np, cond_np, elv_np].")

            images_np, cond_np, elv_np = value[0], value[1], value[2]
            images_np = np.asarray(images_np, dtype=np.float32)

            if images_np.ndim != 3:
                raise ValueError(f"Images for {city} must be (N,H,W). Got {images_np.shape}")
            N, H, W = images_np.shape

            cond_np = np.asarray(cond_np, dtype=np.float32)
            elv_np = np.asarray(elv_np, dtype=np.float32)

            # ----- flexible handling of cond shapes -----
            if cond_np.ndim == 2:
                cond_np = np.broadcast_to(cond_np, (N, H, W))
            elif cond_np.ndim == 3:
                if cond_np.shape == (1, H, W):
                    cond_np = np.broadcast_to(cond_np[0], (N, H, W))
                elif cond_np.shape == (N, H, W):
                    pass
                else:
                    raise ValueError(
                        f"cond_np for {city} must be (H,W), (1,H,W), or (N,H,W). "
                        f"Got {cond_np.shape}, images {images_np.shape}"
                    )
            else:
                raise ValueError(
                    f"cond_np for {city} must be (H,W), (1,H,W), or (N,H,W). Got {cond_np.shape}"
                )

            # ----- flexible handling of elevation shapes -----
            if elv_np.ndim == 2:
                elv_np = np.broadcast_to(elv_np, (N, H, W))
            elif elv_np.ndim == 3:
                if elv_np.shape == (1, H, W):
                    elv_np = np.broadcast_to(elv_np[0], (N, H, W))
                elif elv_np.shape == (N, H, W):
                    pass
                else:
                    raise ValueError(
                        f"elv_np for {city} must be (H,W), (1,H,W), or (N,H,W). "
                        f"Got {elv_np.shape}, images {images_np.shape}"
                    )
            else:
                raise ValueError(
                    f"elv_np for {city} must be (H,W), (1,H,W), or (N,H,W). Got {elv_np.shape}"
                )

            if self.H is None:
                self.H, self.W = H, W
            else:
                if (H, W) != (self.H, self.W):
                    raise ValueError(
                        f"All cities must share same H,W. {city} has {(H,W)}, expected {(self.H,self.W)}"
                    )

            for i in range(N):
                self.samples.append((images_np[i], cond_np[i], elv_np[i], city))

        if len(self.samples) == 0:
            raise ValueError("No samples found in data_dict.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_np, cond_np, elv_np, city = self.samples[idx]
        img = torch.from_numpy(img_np).float().unsqueeze(0)   # (1,H,W)
        cond = torch.from_numpy(cond_np).float().unsqueeze(0) # (1,H,W)
        elv = torch.from_numpy(elv_np).float().unsqueeze(0)   # (1,H,W)
        return img, cond, elv, city


class ConditionalInpaintingDataset(Dataset):
    """
    Wraps grayscale images (N,H,W) into (N,1,H,W) tensors and produces masks.
    mask_type ∈ {"rectangle","banded","irregular","random_percentage","half","cloud_fbm"}
    cloud_fbm parameters via mask_kwargs.
    """
    def __init__(
        self,
        images: np.ndarray,
        conds: np.ndarray,
        elevations: np.ndarray,
        city_names: List[str],
        mask_type: str = "rectangle",
        mask_generator: Optional[Callable] = None,
        mask_percentage: float = 0.3,
        mask_kwargs: Optional[Dict] = None,
    ):
        self.images = torch.from_numpy(images).float().unsqueeze(1)  # (N,1,H,W)
        self.conds = torch.from_numpy(conds).float()
        self.elevations = torch.from_numpy(elevations).float()
        self.city_names = city_names
        self.mask_type = mask_type
        self.mask_percentage = float(mask_percentage)
        self.mask_kwargs = mask_kwargs or {}
        self.mask_generator = mask_generator or self._get_mask_generator(mask_type)

    def _get_mask_generator(self, mask_type: str) -> Callable:
        return {
            "rectangle": self.rectangle_mask,
            "banded": self.banded_mask,
            "irregular": self.irregular_mask,
            "random_percentage": self.random_percentage_mask,
            "half": self.half_mask,
            "cloud_fbm": (lambda img: self.cloud_fbm_mask(img, **self.mask_kwargs)),
        }.get(mask_type) or (lambda img: (_ for _ in ()).throw(ValueError(f"Unknown mask_type: {mask_type}")))

    # ---------- simple masks ----------
    def rectangle_mask(self, image: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(image)
        h, w = image.shape[-2:]
        h_length = torch.randint(h // 4, h // 2, (1,))
        w_length = torch.randint(w // 4, w // 2, (1,))
        h_start = torch.randint(0, max(1, h - int(h_length)), (1,))
        w_start = torch.randint(0, max(1, w - int(w_length)), (1,))
        mask[:, h_start:h_start + h_length, w_start:w_start + w_length] = 0
        return mask

    def banded_mask(self, image: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(image)
        h, w = image.shape[-2:]
        band_width = torch.randint(max(1, h // 12), max(2, h // 4), (1,))
        band_pos = torch.randint(0, max(1, h - int(band_width)), (1,))
        if torch.rand(1) > 0.5:
            mask[:, band_pos:band_pos + band_width, :] = 0
        else:
            mask[:, :, band_pos:band_pos + band_width] = 0
        return mask

    def irregular_mask(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        m = np.ones((h, w), dtype=np.uint8)
        max_area = 0.4 * h * w
        current = 0
        num_shapes = random.randint(3, 7)
        for _ in range(num_shapes):
            if current >= max_area:
                break
            shape = random.choice(["circle", "square"])
            size = random.randint(int(0.1 * min(h, w)), int(0.3 * min(h, w)))
            x, y = random.randint(0, w - size), random.randint(0, h - size)
            if shape == "circle":
                center = (x + size // 2, y + size // 2)
                cv2.circle(m, center, size // 2, 0, -1)
            else:
                cv2.rectangle(m, (x, y), (x + size, y + size), 0, -1)
            current = (m == 0).sum()
        mask_np = m.astype(np.float32)  # 1 keep, 0 hide
        return torch.from_numpy(mask_np).view(1, h, w)

    def random_percentage_mask(self, image: torch.Tensor) -> torch.Tensor:
        h, w = image.shape[-2:]
        total = h * w
        k = int(self.mask_percentage * total)
        idxs = random.sample(range(total), max(0, min(total, k)))
        mask_np = np.ones((h, w), dtype=np.float32)
        for idx in idxs:
            y, x = divmod(idx, w)
            mask_np[y, x] = 0.0
        return torch.from_numpy(mask_np).view(1, h, w)

    def half_mask(self, image: torch.Tensor) -> torch.Tensor:
        mask = torch.ones_like(image)
        h, w = image.shape[-2:]
        if torch.rand(1) > 0.5:
            (mask[:, :, :w // 2] if torch.rand(1) > 0.5 else mask[:, :, w // 2:]).zero_()
        else:
            (mask[:, :h // 2, :] if torch.rand(1) > 0.5 else mask[:, h // 2:, :]).zero_()
        return mask

    # ---------- cloud-like FBM masks ----------
    def _fbm_noise(self, h: int, w: int, octaves=5, persistence=0.55, lacunarity=2.0, seed=None) -> np.ndarray:
        g = torch.Generator()
        if seed is not None:
            g.manual_seed(int(seed))
        noise = torch.zeros(1, 1, h, w)
        amp = 1.0
        for _ in range(octaves):
            scale = lacunarity ** (octaves - 1 - _)
            hh = max(1, int(round(h / scale)))
            ww = max(1, int(round(w / scale)))
            n = torch.randn(1, 1, hh, ww, generator=g)
            n = F.interpolate(n, size=(h, w), mode="bicubic", align_corners=False)
            noise += amp * n
            amp *= persistence
        noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)
        return noise[0, 0].numpy()

    def _anisotropic_blur(self, img: np.ndarray, sigma_x=3.0, sigma_y=1.0, angle_deg=0.0) -> np.ndarray:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        blurred = cv2.GaussianBlur(rot, ksize=(0, 0), sigmaX=sigma_x, sigmaY=sigma_y, borderType=cv2.BORDER_REFLECT)
        M_inv = cv2.getRotationMatrix2D((w/2, h/2), -angle_deg, 1.0)
        return cv2.warpAffine(blurred, M_inv, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    def cloud_fbm_mask(
        self,
        image: torch.Tensor,
        coverage: float = 0.4,
        style: str = "cumulus",
        octaves: int = 5,
        persistence: float = 0.55,
        lacunarity: float = 2.0,
        streak: float = 0.0,
        wind_dir_deg: float = 0.0,
        seed: Optional[int] = None,
        morph_puff: int = 1,
    ) -> torch.Tensor:
        h, w = image.shape[-2:]
        n = self._fbm_noise(h, w, octaves=octaves, persistence=persistence, lacunarity=lacunarity, seed=seed)
        if style == "cumulus":
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=0.6, sigmaY=0.6)
        elif style == "stratus":
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=2.5, sigmaY=1.0)
            if streak > 0:
                n = self._anisotropic_blur(n, sigma_x=6.0 * streak + 1e-6, sigma_y=1.0, angle_deg=wind_dir_deg)
        elif style == "cirrus":
            n = (n - n.min()) / (n.max() - n.min() + 1e-8)
            n = cv2.GaussianBlur(n, (0, 0), sigmaX=0.3, sigmaY=0.3)
            if streak > 0:
                n = self._anisotropic_blur(n, sigma_x=3.0 * streak + 1e-6, sigma_y=0.6, angle_deg=wind_dir_deg)

        thr = float(np.quantile(n, 1.0 - float(coverage)))
        cloud = (n >= thr).astype(np.uint8)
        if morph_puff > 0:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            cloud = cv2.morphologyEx(cloud, cv2.MORPH_CLOSE, k, iterations=int(morph_puff))
        mask_np = 1.0 - cloud.astype(np.float32)  # 1 keep, 0 cloud
        return torch.from_numpy(mask_np).view(1, h, w)

    # ---------- dataset API ----------
    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        image = self.images[idx]
        conds = self.conds[idx]
        elevations = self.elevations[idx]
        city_name = self.city_names[idx]
        mask = self.mask_generator(image)
        revealed = image * mask
        return {
            "image": image,
            "revealed": revealed,
            "mask": mask,
            "cond": conds,
            "elevations": elevations,
            "city_names": city_name,
        }


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


def mix(diff_reconstructed, ground_truth, ground_revealed, masks, coef=1.0):
    diff_reconstructed = diff_reconstructed.detach().cpu()
    ground_truth = ground_truth.cpu()
    ground_revealed = ground_revealed.cpu()
    mask = masks.cpu()

    # diffusion-only completion
    complete_prediction = diff_reconstructed * (1 - mask) + ground_revealed

    # baselines
    baseline = inpaint_opencv_batch(ground_revealed, 1 - mask, radius=20, method="telea")
    linear_baseline = linear_interp2d_batch(ground_revealed, 1 - mask)

    mixed = (coef * diff_reconstructed + (1 - coef) * baseline) * (1 - mask) + ground_revealed

    # denoise all for fairness
    complete_prediction = denoise_batch(complete_prediction, method="gaussian", ksize=7, sigmaX=1)
    ground_truth = denoise_batch(ground_truth, method="gaussian", ksize=3, sigmaX=1)
    baseline = denoise_batch(baseline, method="gaussian", ksize=3, sigmaX=1)
    baseline_lr = denoise_batch(linear_baseline, method="gaussian", ksize=3, sigmaX=1)
    mixed = denoise_batch(mixed, method="gaussian", ksize=7, sigmaX=1)

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
):
    accelerator = Accelerator()
    device = accelerator.device
    accelerator.print(f"Using device: {device}, num_processes={accelerator.num_processes}")

    set_seed(seed)

    # ---- load data dict and flatten ----
    data_dict = load_data_dict(data_path)
    dataset = CityConditionalDataset(data_dict)
    H, W = dataset.H, dataset.W
    accelerator.print(f"Dataset size: {len(dataset)}, image size: {H}x{W}")

    # ---- train/val split ----
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(seed)
    train_ds, test_ds = random_split(dataset, [train_len, val_len], generator=generator)

    # ---- build CPU test tensors ----
    base_test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    test_imgs_list = []
    test_conds_list = []
    test_elev_list = []
    test_city_list: List[str] = []

    for img, cond, elv, city in base_test_loader:
        test_imgs_list.append(img)
        test_conds_list.append(cond)
        test_elev_list.append(elv)
        test_city_list.extend(list(city))

    test_images = torch.cat(test_imgs_list, dim=0)   # (N_val,1,H,W)
    test_conds = torch.cat(test_conds_list, dim=0)   # (N_val,1,H,W)
    test_elevs = torch.cat(test_elev_list, dim=0)    # (N_val,1,H,W)

    test_images = test_images[:N_test]
    test_conds = test_conds[:N_test]
    test_elevs = test_elevs[:N_test]
    test_city_list = test_city_list[:N_test]

    accelerator.print(f"Using N_test={N_test}")

    test_data_np = test_images.squeeze(1).numpy()   # (N_test,H,W)
    test_conds_np = test_conds.squeeze(1).numpy()   # (N_test,H,W)
    test_elevs_np = test_elevs.squeeze(1).numpy()   # (N_test,H,W)

    # ---- diffusion schedulers ----
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
        thresholding=False,
        use_karras_sigmas=True,
        timestep_spacing="trailing",
    )

    # ---- cloudy dataset with FBM clouds ----
    cloudy = ConditionalInpaintingDataset(
        test_data_np,
        test_conds_np,
        test_elevs_np,
        test_city_list,
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

    cloudy_loader = DataLoader(
        cloudy,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ---- build UNet model and load checkpoint ----
    from diffusers import UNet2DModel
    unet = UNet2DModel(
        sample_size=H,
        in_channels=3,  # image + cond + elevation (handled inside sampler)
        out_channels=1,
        layers_per_block=2,
        block_out_channels=(64, 128, 256, 256),
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "AttnDownBlock2D"),
        up_block_types=("AttnUpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
        attention_head_dim=8,
        dropout=0.1,
    )
    state = torch.load(ckpt, map_location="cpu")
    unet.load_state_dict(state)
    unet.to(device)

    # ---- wrap model + loader with accelerate ----
    unet, cloudy_loader = accelerator.prepare(unet, cloudy_loader)

    sampler = ConditionalDiffusionSampler(unet, scheduler)

    # ---- multi-GPU inference ----
    all_samples = []

    for sample_idx in range(N_samples):
        accelerator.print(f"Sampling round {sample_idx+1}/{N_samples} ...")
        preds_this_round = []

        for batch in cloudy_loader:
            test_images_b = batch["image"].to(device)
            test_revealed_b = batch["revealed"].to(device)
            test_masks_b = batch["mask"].to(device)
            test_conds_b = batch["cond"].to(device)
            test_elev_b = batch["elevations"].to(device)

            s_tmp, _ = sampler.sample_copaint(
                test_revealed_b,
                test_masks_b,
                test_conds_b,
                test_elev_b,
                timesteps=timesteps,
                learning_rate=10.0,
                grad_steps=1,
                guidance_stride=2,
                tail_guidance=20,
                do_projection=True,
            )

            s_tmp = accelerator.gather(s_tmp)

            if accelerator.is_main_process:
                preds_this_round.append(s_tmp.cpu())

        if accelerator.is_main_process:
            s_round = torch.cat(preds_this_round, dim=0)  # (N_test,1,H,W)
            s_round = s_round[:N_test]
            all_samples.append(s_round)

    # ---- metrics & CSV only on main process ----
    if accelerator.is_main_process:
        stacked_samples = torch.stack(all_samples, dim=0)  # (N_samples, N_test, 1,H,W)
        s_mean = stacked_samples.mean(dim=0)               # (N_test,1,H,W)

        # rebuild ground_truth, revealed, masks from CPU loader
        test_imgs_full = []
        test_rvld_full = []
        test_msks_full = []
        test_cnds_full = []

        for batch in DataLoader(cloudy, batch_size=batch_size, shuffle=False):
            test_imgs_full.append(batch["image"])
            test_rvld_full.append(batch["revealed"])
            test_msks_full.append(batch["mask"])
            test_cnds_full.append(batch["cond"])

        test_images_all = torch.cat(test_imgs_full, dim=0)[:N_test]
        test_revealed_all = torch.cat(test_rvld_full, dim=0)[:N_test]
        masks_all = torch.cat(test_msks_full, dim=0)[:N_test]
        test_conds_all = torch.cat(test_cnds_full, dim=0)[:N_test]

        ground_truth = test_images_all.cpu()
        urban_mask = torch.zeros_like(ground_truth)
        urban_mask[:, :, 25:80, 40:100] = 1

        pr = mix(s_mean, ground_truth, test_revealed_all, masks_all, coef=0.6)
        diff_pred = pr["raw_diffusion_prediction"]
        baseline = pr["baseline_CV"]
        baseline_lr = pr["baseline_linear"]
        mixed = pr["mixed_prediction"]
        ground_truth = pr["ground_truth"]
        ground_revealed = pr["ground_revealed"]
        mask = pr["mask"]

        # physical-space metrics
        tbl = compare_methods_fixed(
            truth=ground_truth,
            predictions={
                "diffusion": mixed,       # NOTE: using mixed prediction (diff + CV), as in ref function
                "linear": baseline_lr,
            },
            mask=1 - masks_all.cpu(),    # hole = 1 where cloud was
            region="hole",
            norm="-1_1",
            custom_minmax=(262.7, 333.86),
            urban_mask=urban_mask,
            uhi_phys_minmax=(262.7, 333.86),
            uhi_mode="smape",
            uhi_min_denom_k=0.5,
        )

        # ---- merge diffusion + baseline (linear) into a single-row table ----
        row_diff = tbl[tbl["method"] == "diffusion"].copy()
        row_lin = tbl[tbl["method"] == "linear"].copy()

        if len(row_diff) != 1 or len(row_lin) != 1:
            raise RuntimeError(
                f"Expected exactly one row for each method; got "
                f"{len(row_diff)} diffusion rows and {len(row_lin)} linear rows"
            )

        row_diff = row_diff.iloc[0]
        row_lin = row_lin.iloc[0]

        metric_cols = [
            "RMSE_mean", "RMSE_std",
            "PSNR_mean", "PSNR_std",
            "SSIM_mean", "SSIM_std",
            "UHI_error_mean", "UHI_error_std",
            "R2_mean", "R2_std"
        ]

        combined = {}
        combined["N"] = int(row_diff["N"])
        combined["region"] = row_diff["region"]
        combined["UHI_error_mode"] = row_diff["UHI_error_mode"]
        combined["method_diff"] = row_diff["method"]
        combined["method_linear"] = row_lin["method"]

        for col in metric_cols:
            combined[f"diff_{col}"] = row_diff[col]
            combined[f"linear_{col}"] = row_lin[col]

        combined["cloud_coverage"] = cloud_coverage
        combined["octaves"] = octaves
        combined["wind_degree"] = wind_degree
        combined["N_test"] = N_test
        combined["N_samples"] = N_samples
        combined["timesteps"] = timesteps
        combined["seed"] = seed

        out_df = pd.DataFrame([combined])

        os.makedirs(output_dir, exist_ok=True)
        out_name = f"metrics_cov{cloud_coverage}_oct{octaves}_wind{wind_degree}.csv"
        out_path = os.path.join(output_dir, out_name)
        out_df.to_csv(out_path, index=False)
        accelerator.print(f"Saved metrics row to: {out_path}")

    accelerator.wait_for_everyone()


# -------------------------------------------------------------------
# 3) CLI entry point
# -------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True,
                   help="Path to Multicity_dict.joblib")
    p.add_argument("--ckpt", type=str, required=True,
                   help="Path to UNet checkpoint (e.g. unet_epoch050.pth)")
    p.add_argument("--output_dir", type=str, default="./metrics_out",
                   help="Where to save one-row CSV per config")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--N_test", type=int, default=400)
    p.add_argument("--cloud_coverage", type=float, required=True)
    p.add_argument("--octaves", type=int, required=True)
    p.add_argument("--batch_size", type=int, default=40)
    p.add_argument("--N_samples", type=int, default=10)
    p.add_argument("--wind_degree", type=int, required=True)
    p.add_argument("--timesteps", type=int, default=100)
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
    )


if __name__ == "__main__":
    main()

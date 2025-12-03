from __future__ import annotations
import random
from typing import Callable, Dict, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import cv2

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


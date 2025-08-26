import numpy as np
import torch
import cv2
from scipy import interpolate

def inpaint_opencv_batch(images: torch.Tensor, mask: torch.Tensor, radius: int = 3, method: str = "telea") -> torch.Tensor:
    """
    images: (B,1,H,W) float
    mask:   (B,1,H,W) 1=revealed, 0=hole
    returns restored in original per-image range
    """
    assert images.ndim == 4 and images.size(1) == 1
    B, _, H, W = images.shape
    out = torch.empty_like(images)
    flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS

    imgs_np = images.detach().cpu().numpy()
    mask_np = (mask.detach().cpu().numpy() <= 0).astype(np.uint8)  # 1 where HOLE

    for i in range(B):
        img = imgs_np[i, 0]
        m = mask_np[i, 0] * 255
        vmin, vmax = float(img.min()), float(img.max())
        if vmax <= vmin + 1e-12:
            out[i, 0] = torch.from_numpy(img)
            continue
        u8 = (np.clip((img - vmin)/(vmax - vmin), 0, 1)*255.0 + 0.5).astype(np.uint8)
        restored_u8 = cv2.inpaint(u8, m, radius, flag)
        restored = restored_u8.astype(np.float32)/255.0*(vmax - vmin) + vmin
        out[i, 0] = torch.from_numpy(restored)

    return out.to(images.device, images.dtype)

def linear_interp2d_batch(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Simple 2D linear interpolation in holes.
    mask: 1=revealed, 0=hole
    """
    B, _, H, W = images.shape
    out = torch.empty_like(images)
    imgs_np = images.detach().cpu().numpy()
    holes_np = (mask.detach().cpu().numpy() <= 0)

    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    for i in range(B):
        img = imgs_np[i, 0]
        m = holes_np[i, 0].astype(bool)

        known_x = X[~m].ravel()
        known_y = Y[~m].ravel()
        known_v = img[~m].ravel()
        missing_x = X[m].ravel()
        missing_y = Y[m].ravel()

        if known_v.size > 3 and missing_x.size > 0:
            interp_vals = interpolate.griddata(
                points=np.stack([known_x, known_y], axis=-1),
                values=known_v,
                xi=np.stack([missing_x, missing_y], axis=-1),
                method="linear",
            )
            nan_mask = np.isnan(interp_vals)
            if nan_mask.any():
                interp_vals[nan_mask] = interpolate.griddata(
                    points=np.stack([known_x, known_y], axis=-1),
                    values=known_v,
                    xi=np.stack([missing_x[nan_mask], missing_y[nan_mask]], axis=-1),
                    method="nearest",
                )
            filled = img.copy()
            filled[m] = interp_vals
        else:
            filled = img
        out[i, 0] = torch.from_numpy(filled.astype(np.float32))

    return out.to(images.device, images.dtype)

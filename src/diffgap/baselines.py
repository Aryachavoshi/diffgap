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
    mask_np = (mask.detach().cpu().numpy() > 0).astype(np.uint8)  # 1 where HOLE

    for i in range(B):
        img = imgs_np[i, 0]                     # (H,W)
        m   = mask_np[i, 0] * 255               # uint8 0/255

        # per-image scale to 8-bit
        vmin, vmax = float(np.min(img)), float(np.max(img))
        if vmax <= vmin + 1e-12:
            img_u8 = np.zeros_like(img, dtype=np.uint8)
        else:
            img_u8 = np.clip((img - vmin) / (vmax - vmin), 0, 1)
            img_u8 = (img_u8 * 255.0 + 0.5).astype(np.uint8)

        # inpaint
        restored_u8 = cv2.inpaint(img_u8, m, radius, flag)

        # back to original float range
        restored = restored_u8.astype(np.float32) / 255.0
        restored = restored * (vmax - vmin) + vmin
        out[i, 0] = torch.from_numpy(restored)

    return out.to(images.device, images.dtype)
    
def linear_interp2d_batch(images: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Simple 2D linear interpolation in holes.
    mask: 1=revealed, 0=hole
    """
    """
    Very simple 2D linear interpolation for cloud-masked grayscale images.

    Args:
      images: (B,1,H,W) float tensor (NaNs allowed for holes)
      mask:   (B,1,H,W) bool/0-1 tensor; True/1 where HOLE (cloud) to fill

    Returns:
      (B,1,H,W) float tensor with linear interpolation filled
    """
    B, _, H, W = images.shape
    out = torch.empty_like(images)

    imgs_np = images.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()

    # grid of coordinates
    X, Y = np.meshgrid(np.arange(W), np.arange(H))

    for i in range(B):
        img = imgs_np[i, 0]
        m   = mask_np[i, 0].astype(bool)

        # known points (where mask==0)
        known_x = X[~m].ravel()
        known_y = Y[~m].ravel()
        known_v = img[~m].ravel()

        # missing points (where mask==1)
        missing_x = X[m].ravel()
        missing_y = Y[m].ravel()

        if known_v.size > 3:  # need at least 3 pts for linear
            interp_vals = interpolate.griddata(
                points=np.stack([known_x, known_y], axis=-1),
                values=known_v,
                xi=np.stack([missing_x, missing_y], axis=-1),
                method='linear'
            )

            # fallback for any remaining NaNs (outside convex hull)
            nan_mask = np.isnan(interp_vals)
            if nan_mask.any():
                interp_vals[nan_mask] = interpolate.griddata(
                    points=np.stack([known_x, known_y], axis=-1),
                    values=known_v,
                    xi=np.stack([missing_x[nan_mask], missing_y[nan_mask]], axis=-1),
                    method='nearest'
                )

            # assign back
            filled = img.copy()
            filled[m] = interp_vals
        else:
            # degenerate: just copy original
            filled = img

        out[i, 0] = torch.from_numpy(filled.astype(np.float32))

    return out.to(images.device, images.dtype)

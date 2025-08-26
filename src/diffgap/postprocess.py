import numpy as np
import torch
import cv2

def denoise_batch(tensor_images: torch.Tensor, method: str = "nlm", **kwargs) -> torch.Tensor:
    """
    tensor_images: (B,1,H,W), values in any range (we will scale to [0,1] for OpenCV ops)
    method ∈ {"nlm","gaussian","median","bilateral"}
    """
    imgs = tensor_images.detach().cpu().numpy()  # (B,1,H,W)
    B, _, H, W = imgs.shape
    out = np.empty_like(imgs)

    for i in range(B):
        img = imgs[i, 0]
        vmin, vmax = float(img.min()), float(img.max())
        if vmax <= vmin + 1e-12:
            out[i, 0] = img
            continue

        u8 = (np.clip((img - vmin)/(vmax - vmin), 0, 1) * 255.0 + 0.5).astype(np.uint8)

        if method == "nlm":
            u8 = cv2.fastNlMeansDenoising(u8,
                                           None,
                                           h=kwargs.get("h", 10),
                                           templateWindowSize=kwargs.get("templateWindowSize", 7),
                                           searchWindowSize=kwargs.get("searchWindowSize", 21))
        elif method == "gaussian":
            k = int(kwargs.get("ksize", 5)) | 1
            u8 = cv2.GaussianBlur(u8, (k, k), sigmaX=kwargs.get("sigmaX", 1.5))
        elif method == "median":
            k = int(kwargs.get("ksize", 5)) | 1
            u8 = cv2.medianBlur(u8, k)
        elif method == "bilateral":
            u8 = cv2.bilateralFilter(u8,
                                     d=kwargs.get("d", 9),
                                     sigmaColor=kwargs.get("sigmaColor", 75),
                                     sigmaSpace=kwargs.get("sigmaSpace", 75))
        else:
            raise ValueError(f"Unsupported method: {method}")

        den = u8.astype(np.float32)/255.0*(vmax - vmin) + vmin
        out[i, 0] = den

    return torch.from_numpy(out).to(tensor_images.device, tensor_images.dtype)

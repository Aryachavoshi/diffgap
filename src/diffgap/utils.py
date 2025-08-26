from typing import Literal, Optional
import torch

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def apply_region(mask: Optional[torch.Tensor], region: Literal["all","revealed","hole"], x: torch.Tensor):
    """
    mask: 1=revealed/keep, 0=hole. Returns x masked to requested region.
    """
    if mask is None or region == "all":
        return x
    if region == "revealed":
        return x * mask
    if region == "hole":
        return x * (~mask)
    raise ValueError(f"Unknown region: {region}")

def data_range_from_normalization(norm: Literal["0_1","-1_1","custom"], vmin=0.0, vmax=1.0):
    if norm == "0_1":
        return 1.0
    if norm == "-1_1":
        return 2.0
    if norm == "custom":
        return float(vmax - vmin)
    raise ValueError("norm must be one of {'0_1','-1_1','custom'}")

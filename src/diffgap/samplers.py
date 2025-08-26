from __future__ import annotations
import contextlib
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class DiffusionSampler(nn.Module):
    """
    Gradient-guided, post-conditioned sampling for inpainting where the mask=1 indicates revealed pixels.
    Works with a trained ε-prediction model and a Diffusers scheduler (DDIM/DDPM/etc.)
    """
    def __init__(self, trained_model: nn.Module, scheduler) -> None:
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.trained_model = trained_model.to(self.device).eval()
        for p in self.trained_model.parameters():
            p.requires_grad_(False)

        self.scheduler = scheduler
        self.num_timesteps = int(scheduler.num_train_timesteps)

        # expect diffusers-style scheduler with .betas
        self.b_t  = scheduler.betas.to(self.device)
        self.a_t  = (1.0 - self.b_t)
        self.ab_t = torch.cumprod(self.a_t, dim=0).to(self.device)

    # Intentionally keep grads through x_t for inner optimization
    def eps_to_x0(self, x_t: torch.Tensor, t_idx: int, eps: torch.Tensor) -> torch.Tensor:
        ab = self.ab_t[t_idx]
        return (x_t - (1 - ab).sqrt() * eps) / ab.sqrt()

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
        timesteps: int = 20,
        grad_steps: int = 5,
        learning_rate: float = 0.1,
        use_amp: bool = True,
        do_projection: bool = True,
        return_intermediates_every: int = 5,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        revealed_image: (B,1,H,W) where zeros indicate holes, non-zero are revealed/keep
        """
        scheduler = self.scheduler
        scheduler.set_timesteps(timesteps)
        schedule_ts = scheduler.timesteps.to(self.device)

        if revealed_image.dim() == 3:
            revealed_image = revealed_image.unsqueeze(1)
        revealed_image = revealed_image.to(self.device)

        mask = (revealed_image != 0).float()
        x_t = torch.randn_like(revealed_image, device=self.device)

        def amp_ctx():
            if use_amp and self.device.type == "cuda":
                return torch.amp.autocast(device_type="cuda")
            return contextlib.nullcontext()

        intermediates: List[torch.Tensor] = []

        for i, t in enumerate(schedule_ts):
            t_idx_scalar = int(t.item())
            t_tensor = torch.full((revealed_image.size(0),), t_idx_scalar, device=self.device, dtype=torch.long)

            x_t = x_t.detach().requires_grad_(True)

            # inner gradient descent to match revealed pixels in x0
            for _ in range(grad_steps):
                with amp_ctx():
                    eps_pred = self.trained_model(x_t, t_tensor).sample
                    x0_pred  = self.eps_to_x0(x_t, t_idx_scalar, eps_pred)
                    pred_revealed = x0_pred * mask
                    loss = ((pred_revealed - revealed_image) ** 2).sum() / (mask.sum() + 1e-8)

                loss.backward()
                with torch.no_grad():
                    x_t -= learning_rate * x_t.grad
                    x_t.grad = None

                del eps_pred, x0_pred, pred_revealed, loss

            # scheduler update x_{t-1}
            with torch.no_grad(), amp_ctx():
                eps_pred = self.trained_model(x_t, t_tensor).sample
                step_out = scheduler.step(eps_pred, t, x_t)
                x_t = step_out.prev_sample
                del eps_pred

            if do_projection:
                with torch.no_grad():
                    x_t = self.project_revealed(x_t, revealed_image, t_idx_scalar, mask)

            if (i % max(1, return_intermediates_every) == 0) or (t_idx_scalar == 99):
                intermediates.append(x_t.detach().cpu().clone())

        return x_t.detach(), intermediates

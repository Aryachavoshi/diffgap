from __future__ import annotations
import contextlib
import torch
import torch.nn as nn
from typing import List, Tuple, Optional

class DiffusionSampler(nn.Module):
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
        timesteps: int = 20,
        grad_steps: int = 5,
        learning_rate: float = 0.1,
        use_amp: bool = True,
        do_projection: bool = True,
        return_intermediates_every: int = 5,
        guidance_stride: int = 4,
        tail_guidance: int = 5,
        t_start_idx: Optional[int] = None,   # start at this index (lower noise)
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        revealed_image: (B,1,H,W) where zeros indicate holes, non-zero are revealed/keep
        """
        scheduler = self.scheduler

        if t_start_idx is not None:
            t_start_idx = int(t_start_idx)
            assert 0 <= t_start_idx < self.num_timesteps, "t_start_idx out of range"

            ts = torch.linspace(float(t_start_idx), 0.0, steps=timesteps, device=self.device).round().long()

            # If Karras sigmas were enabled, you cannot use custom timesteps. Disable them.
            if getattr(self.scheduler.config, "use_karras_sigmas", False):
                self.scheduler.config.use_karras_sigmas = False

            try:
                # Newer diffusers: pass custom grid directly
                scheduler.set_timesteps(timesteps=ts, device=self.device)
            except TypeError:
                # Older versions: set length, then overwrite grid and init buffers
                scheduler.set_timesteps(len(ts), device=self.device)
                scheduler.timesteps = ts
                if hasattr(scheduler, "num_inference_steps"):
                    scheduler.num_inference_steps = ts.shape[0]

                # >>> IMPORTANT: pre-size buffers to solver_order <<<
                order = int(getattr(scheduler.config, "solver_order", 2))
                if hasattr(scheduler, "model_outputs"):
                    scheduler.model_outputs = [None] * order
                if hasattr(scheduler, "derivatives"):
                    scheduler.derivatives = [None] * order
                if hasattr(scheduler, "lower_order_nums"):
                    scheduler.lower_order_nums = 0
        else:
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
        
        # ---- Initialize x_t ----
        # start from standard Gaussian
        x_t = torch.randn_like(revealed_image, device=self.device)


        # If we have a t_start_idx, "imprint" the revealed pixels by sampling q(x_t|x0) on masks==1
        if t_start_idx is not None:
            with torch.no_grad():
                x_t = self.project_revealed(x_t, revealed_image, int(t_start_idx), masks)
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
                        eps_pred = self.trained_model(x_t, t_tensor).sample
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
                eps_pred = self.trained_model(x_t, t_tensor).sample
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

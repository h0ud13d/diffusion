# diffusion.py
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    v_prediction: bool = True

def _betas_cosine(T: int, s: float = 0.008) -> torch.Tensor:
    steps = np.arange(T + 1, dtype=np.float64)
    t = steps / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    f = f / f[0]
    betas = 1 - (f[1:] / f[:-1])
    betas = np.clip(betas, 1e-8, 0.999)
    return torch.from_numpy(betas).float()

class GaussianDiffusion1D(nn.Module):
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        if cfg.beta_schedule == "cosine":
            betas = _betas_cosine(cfg.timesteps)
        else:
            raise NotImplementedError
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def _to_v(self, x0, eps, t):
        c1 = self.sqrt_alpha_bar.to(x0.device)[t].view(-1,1,1)
        c2 = self.sqrt_one_minus_alpha_bar.to(x0.device)[t].view(-1,1,1)
        return c1 * eps - c2 * x0

    def _from_v_to_x0_eps(self, x_t, v, t):
        c1 = self.sqrt_alpha_bar.to(x_t.device)[t].view(-1,1,1)
        c2 = self.sqrt_one_minus_alpha_bar.to(x_t.device)[t].view(-1,1,1)
        x0 = c1 * x_t - c2 * v
        eps = c2 * x_t + c1 * v
        return x0, eps

    def q_sample(self, x0, t, noise=None):
        if noise is None: noise = torch.randn_like(x0)
        sab  = self.sqrt_alpha_bar.to(x0.device)[t].view(-1,1,1)
        somb = self.sqrt_one_minus_alpha_bar.to(x0.device)[t].view(-1,1,1)
        return sab * x0 + somb * noise

    def loss_on(self, x0, t=None, use_v=True):
        B = x0.size(0); dev = x0.device
        if t is None:
            t = torch.randint(0, self.cfg.timesteps, (B,), device=dev)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        if use_v and self.cfg.v_prediction:
            v_target = self._to_v(x0, noise, t)
            v_pred = self.model(x_t, t)
            return F.mse_loss(v_pred, v_target)
        else:
            eps_pred = self.model(x_t, t)
            return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_step(self, x_t, t):
        dev = x_t.device
        if self.cfg.v_prediction:
            v = self.model(x_t, t)
            x0_pred, eps_pred = self._from_v_to_x0_eps(x_t, v, t)
        else:
            sqrt_omb   = self.sqrt_one_minus_alpha_bar.to(dev)[t].view(-1,1,1)
            alpha_bar_t = self.alpha_bar.to(dev)[t].view(-1,1,1)
            eps_pred   = self.model(x_t, t)
            x0_pred    = (x_t - sqrt_omb * eps_pred) / torch.sqrt(alpha_bar_t + 1e-8)

        betas_t  = self.betas.to(dev)[t].view(-1,1,1)
        alphas_t = 1.0 - betas_t
        mean = torch.sqrt(alphas_t) * x0_pred + (1 - alphas_t) * eps_pred

        if (t == 0).all():
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(betas_t) * noise

    @torch.no_grad()
    def sample_with_mask(self, x_known, known_mask, steps=None):
        dev = x_known.device
        T = self.cfg.timesteps if steps is None else steps
        x_t = torch.randn_like(x_known)
        for t_inv in reversed(range(T)):
            t = torch.full((x_t.size(0),), t_inv, device=dev, dtype=torch.long)
            x_t = self.p_step(x_t, t)

            z = torch.randn_like(x_known)
            sab  = self.sqrt_alpha_bar.to(dev)[t].view(-1,1,1)
            somb = self.sqrt_one_minus_alpha_bar.to(dev)[t].view(-1,1,1)
            x_known_t = sab * x_known + somb * z
            x_t = known_mask * x_known_t + (1.0 - known_mask) * x_t
        return x_t

class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1 - self.decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)

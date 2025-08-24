# enhanced_diffusion.py - Advanced diffusion with classifier-free guidance and DDIM sampling
from dataclasses import dataclass
from typing import Optional, Union, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

@dataclass
class EnhancedDiffusionConfig:
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    v_prediction: bool = True
    classifier_free_guidance: bool = True
    guidance_scale: float = 7.5
    drop_prob: float = 0.1  # Probability of dropping conditions during training

def _betas_cosine(T: int, s: float = 0.008) -> torch.Tensor:
    """Improved cosine noise schedule"""
    steps = np.arange(T + 1, dtype=np.float64)
    t = steps / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    f = f / f[0]
    betas = 1 - (f[1:] / f[:-1])
    betas = np.clip(betas, 1e-8, 0.999)
    return torch.from_numpy(betas).float()

def _betas_linear(T: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear noise schedule"""
    return torch.linspace(beta_start, beta_end, T)

def _betas_sigmoid(T: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Sigmoid noise schedule - smoother transitions"""
    x = torch.linspace(-6, 6, T)
    betas = torch.sigmoid(x) * (beta_end - beta_start) + beta_start
    return betas

class EnhancedGaussianDiffusion1D(nn.Module):
    def __init__(self, model: nn.Module, cfg: EnhancedDiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        
        # Choose noise schedule
        if cfg.beta_schedule == "cosine":
            betas = _betas_cosine(cfg.timesteps)
        elif cfg.beta_schedule == "linear":
            betas = _betas_linear(cfg.timesteps)
        elif cfg.beta_schedule == "sigmoid":
            betas = _betas_sigmoid(cfg.timesteps)
        else:
            raise NotImplementedError(f"Unknown schedule: {cfg.beta_schedule}")
        
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))
        
        # For DDIM sampling
        self.register_buffer("sqrt_recip_alpha_bar", torch.sqrt(1.0 / alpha_bar))
        self.register_buffer("sqrt_recipm1_alpha_bar", torch.sqrt(1.0 / alpha_bar - 1))

    def _to_v(self, x0: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convert to v-parameterization"""
        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_alpha_bar * eps - sqrt_one_minus_alpha_bar * x0

    def _from_v_to_x0_eps(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> tuple:
        """Convert from v-parameterization to x0 and eps"""
        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        x0 = sqrt_alpha_bar * x_t - sqrt_one_minus_alpha_bar * v
        eps = sqrt_one_minus_alpha_bar * x_t + sqrt_alpha_bar * v
        return x0, eps

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * noise

    def _apply_classifier_free_guidance(self, model_output: torch.Tensor, 
                                      model_output_uncond: torch.Tensor,
                                      guidance_scale: float) -> torch.Tensor:
        """Apply classifier-free guidance"""
        return model_output_uncond + guidance_scale * (model_output - model_output_uncond)

    def loss_on(self, x0: torch.Tensor, t: Optional[torch.Tensor] = None, 
                context: Optional[torch.Tensor] = None, use_v: bool = True) -> torch.Tensor:
        """Compute diffusion loss with optional classifier-free guidance training"""
        B = x0.size(0)
        device = x0.device
        
        if t is None:
            t = torch.randint(0, self.cfg.timesteps, (B,), device=device)
        
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        
        # Classifier-free guidance training
        if self.cfg.classifier_free_guidance and context is not None:
            # Randomly drop conditions
            mask = torch.rand(B, device=device) > self.cfg.drop_prob
            context_masked = context.clone() if context is not None else None
            if context_masked is not None:
                # Set dropped conditions to None/zeros
                context_masked[~mask] = 0
        else:
            context_masked = context
        
        # Model prediction - handle both original UNet1D and enhanced models
        if use_v and self.cfg.v_prediction:
            v_target = self._to_v(x0, noise, t)
            try:
                # Try enhanced model with context
                v_pred = self.model(x_t, t, context_masked)
            except TypeError:
                # Fallback to original UNet1D (no context)
                v_pred = self.model(x_t, t)
            return F.mse_loss(v_pred, v_target)
        else:
            try:
                # Try enhanced model with context
                eps_pred = self.model(x_t, t, context_masked)
            except TypeError:
                # Fallback to original UNet1D (no context)
                eps_pred = self.model(x_t, t)
            return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_step_ddpm(self, x_t: torch.Tensor, t: torch.Tensor, 
                   context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Single DDPM sampling step"""
        device = x_t.device
        
        # Model prediction
        if self.cfg.classifier_free_guidance and context is not None:
            # Unconditional prediction
            v_uncond = self.model(x_t, t, None)
            # Conditional prediction
            v_cond = self.model(x_t, t, context)
            # Apply guidance
            v_pred = self._apply_classifier_free_guidance(v_cond, v_uncond, self.cfg.guidance_scale)
        else:
            v_pred = self.model(x_t, t, context)
        
        # Convert predictions
        if self.cfg.v_prediction:
            x0_pred, eps_pred = self._from_v_to_x0_eps(x_t, v_pred, t)
        else:
            eps_pred = v_pred
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            alpha_bar = self.alpha_bar[t].view(-1, 1, 1)
            x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / torch.sqrt(alpha_bar + 1e-8)
        
        # Compute mean
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
        sqrt_alphas_t = torch.sqrt(self.alphas[t]).view(-1, 1, 1)
        
        mean = (1 / sqrt_alphas_t) * (x_t - (betas_t / sqrt_one_minus_alpha_bar) * eps_pred)
        
        # Add noise (except at final step)
        if (t == 0).all():
            return mean
        
        noise = torch.randn_like(x_t)
        sqrt_betas_t = torch.sqrt(betas_t)
        return mean + sqrt_betas_t * noise

    @torch.no_grad()
    def p_step_ddim(self, x_t: torch.Tensor, t: torch.Tensor, t_prev: torch.Tensor,
                   context: Optional[torch.Tensor] = None, eta: float = 0.0) -> torch.Tensor:
        """Single DDIM sampling step"""
        device = x_t.device
        
        # Model prediction
        if self.cfg.classifier_free_guidance and context is not None:
            v_uncond = self.model(x_t, t, None)
            v_cond = self.model(x_t, t, context)
            v_pred = self._apply_classifier_free_guidance(v_cond, v_uncond, self.cfg.guidance_scale)
        else:
            v_pred = self.model(x_t, t, context)
        
        # Convert to x0 prediction
        if self.cfg.v_prediction:
            x0_pred, eps_pred = self._from_v_to_x0_eps(x_t, v_pred, t)
        else:
            eps_pred = v_pred
            sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
            alpha_bar = self.alpha_bar[t].view(-1, 1, 1)
            x0_pred = (x_t - sqrt_one_minus_alpha_bar * eps_pred) / torch.sqrt(alpha_bar + 1e-8)
        
        # DDIM sampling
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1)
        alpha_bar_t_prev = self.alpha_bar[t_prev].view(-1, 1, 1)
        
        sigma_t = eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t) * 
                                   (1 - alpha_bar_t / alpha_bar_t_prev))
        
        # Predicted x_{t-1}
        pred_dir = torch.sqrt(1 - alpha_bar_t_prev - sigma_t**2) * eps_pred
        x_prev = torch.sqrt(alpha_bar_t_prev) * x0_pred + pred_dir
        
        # Add noise
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev

    @torch.no_grad()
    def sample(self, shape: tuple, context: Optional[torch.Tensor] = None, 
              method: str = "ddpm", steps: Optional[int] = None, 
              eta: float = 0.0, progress: bool = True) -> torch.Tensor:
        """Sample from the diffusion model"""
        device = next(self.model.parameters()).device
        
        if steps is None:
            steps = self.cfg.timesteps
        
        # Create sampling schedule
        if method == "ddim":
            # DDIM uses a subset of timesteps
            timesteps = torch.linspace(self.cfg.timesteps - 1, 0, steps, dtype=torch.long)
            timesteps = timesteps.to(device)
        else:
            # DDPM uses all timesteps
            timesteps = torch.arange(self.cfg.timesteps - 1, -1, -1, dtype=torch.long)
            timesteps = timesteps.to(device)
        
        # Initialize with noise
        x_t = torch.randn(shape, device=device)
        
        iterator = tqdm(timesteps, desc=f"Sampling ({method})") if progress else timesteps
        
        for i, t in enumerate(iterator):
            t_batch = t.repeat(shape[0])
            
            if method == "ddim":
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
                t_prev_batch = t_prev.repeat(shape[0])
                x_t = self.p_step_ddim(x_t, t_batch, t_prev_batch, context, eta)
            else:
                x_t = self.p_step_ddpm(x_t, t_batch, context)
        
        return x_t

    @torch.no_grad()
    def sample_with_mask(self, x_known: torch.Tensor, known_mask: torch.Tensor,
                        context: Optional[torch.Tensor] = None, method: str = "ddpm",
                        steps: Optional[int] = None, progress: bool = True) -> torch.Tensor:
        """Sample with partial observations (inpainting)"""
        device = x_known.device
        
        if steps is None:
            steps = self.cfg.timesteps if method == "ddpm" else 50
        
        # Create sampling schedule
        if method == "ddim":
            timesteps = torch.linspace(self.cfg.timesteps - 1, 0, steps, dtype=torch.long)
            timesteps = timesteps.to(device)
        else:
            timesteps = torch.arange(self.cfg.timesteps - 1, -1, -1, dtype=torch.long)
            timesteps = timesteps.to(device)
        
        x_t = torch.randn_like(x_known)
        
        iterator = tqdm(timesteps, desc=f"Inpainting ({method})") if progress else timesteps
        
        for i, t in enumerate(iterator):
            t_batch = t.repeat(x_known.size(0))
            
            # Denoise step
            if method == "ddim":
                t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else torch.tensor(0, device=device)
                t_prev_batch = t_prev.repeat(x_known.size(0))
                x_t = self.p_step_ddim(x_t, t_batch, t_prev_batch, context)
            else:
                x_t = self.p_step_ddpm(x_t, t_batch, context)
            
            # Apply known values constraint
            if not (t == 0).all():
                # Add noise to known values
                noise = torch.randn_like(x_known)
                sqrt_alpha_bar = self.sqrt_alpha_bar[t].view(-1, 1, 1)
                sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1)
                x_known_t = sqrt_alpha_bar * x_known + sqrt_one_minus_alpha_bar * noise
                
                # Mix known and generated values
                x_t = known_mask * x_known_t + (1.0 - known_mask) * x_t
        
        return x_t

class EMAv2:
    """Enhanced EMA with warmup and decay scheduling"""
    def __init__(self, model: nn.Module, decay: float = 0.999, warmup_steps: int = 1000):
        self.decay = decay
        self.warmup_steps = warmup_steps
        self.step = 0
        self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        self.step += 1
        
        # Compute effective decay with warmup
        if self.step < self.warmup_steps:
            decay = min(self.decay, self.step / self.warmup_steps)
        else:
            decay = self.decay
        
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(decay).add_(v.detach(), alpha=1 - decay)

    def copy_to(self, model: nn.Module):
        model.load_state_dict(self.shadow, strict=True)
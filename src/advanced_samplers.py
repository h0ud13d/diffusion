# advanced_samplers.py - State-of-the-art sampling algorithms for diffusion models
import torch
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Callable, List
from tqdm import tqdm
import math

class DPMSolverMultistep:
    """DPM-Solver++ for fast high-quality sampling (Lu et al. 2022)"""
    
    def __init__(self, model, alphas_cumprod: torch.Tensor, 
                 prediction_type: str = "v", 
                 algorithm_type: str = "dpmsolver++"):
        self.model = model
        self.alphas_cumprod = alphas_cumprod
        self.prediction_type = prediction_type
        self.algorithm_type = algorithm_type
        
        # Precompute values for efficiency
        self.register_buffer("sigmas", torch.sqrt((1 - alphas_cumprod) / alphas_cumprod))
        self.register_buffer("log_sigmas", torch.log(self.sigmas))
        
    def register_buffer(self, name: str, tensor: torch.Tensor):
        setattr(self, name, tensor)
    
    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep"""
        log_sigma = torch.log(sigma)
        dists = log_sigma - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).long()
    
    def t_to_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """Convert timestep to sigma"""
        t = t.clamp(0, len(self.sigmas) - 1)
        return self.sigmas[t]
    
    def get_model_input_time(self, t_continuous: torch.Tensor) -> torch.Tensor:
        """Convert continuous time to model input format"""
        return (t_continuous * 1000.0).round().long().clamp(0, 1000 - 1)
    
    def model_fn(self, x: torch.Tensor, t_continuous: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Model wrapper that handles different parameterizations"""
        t_input = self.get_model_input_time(t_continuous).squeeze()
        if t_input.dim() == 0:
            t_input = t_input.unsqueeze(0).expand(x.shape[0])
        
        model_output = self.model(x.transpose(-1, -2), t_input, context)
        model_output = model_output.transpose(-1, -2)
        
        return model_output
    
    def data_prediction_fn(self, x: torch.Tensor, t: torch.Tensor, 
                          context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert model output to data prediction"""
        model_output = self.model_fn(x, t, context)
        
        if self.prediction_type == "epsilon":
            sigma = self.t_to_sigma(self.get_model_input_time(t))
            while sigma.dim() < x.dim():
                sigma = sigma.unsqueeze(-1)
            x0 = (x - sigma * model_output) / (1 + sigma**2).sqrt()
        elif self.prediction_type == "v":
            sigma = self.t_to_sigma(self.get_model_input_time(t))
            while sigma.dim() < x.dim():
                sigma = sigma.unsqueeze(-1)
            alpha = 1 / (1 + sigma**2).sqrt()
            x0 = alpha * x - sigma * alpha * model_output
        else:  # x0
            x0 = model_output
        
        return x0
    
    def noise_prediction_fn(self, x: torch.Tensor, t: torch.Tensor,
                           context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convert model output to noise prediction"""
        model_output = self.model_fn(x, t, context)
        
        if self.prediction_type == "epsilon":
            return model_output
        elif self.prediction_type == "v":
            sigma = self.t_to_sigma(self.get_model_input_time(t))
            while sigma.dim() < x.dim():
                sigma = sigma.unsqueeze(-1)
            alpha = 1 / (1 + sigma**2).sqrt()
            eps = sigma * alpha * x + alpha * model_output
        else:  # x0
            sigma = self.t_to_sigma(self.get_model_input_time(t))
            while sigma.dim() < x.dim():
                sigma = sigma.unsqueeze(-1)
            eps = (x - model_output) / sigma
        
        return eps
    
    def get_time_steps(self, num_steps: int, device: torch.device) -> torch.Tensor:
        """Generate optimal time steps for DPM-Solver"""
        if num_steps == 1:
            return torch.tensor([1.0], device=device)
        else:
            # Exponential schedule for better performance
            t_start, t_end = 1.0, 1.0 / len(self.sigmas)
            return torch.exp(torch.linspace(
                math.log(t_start), math.log(t_end), num_steps + 1, device=device
            ))[:-1]
    
    def dpm_solver_first_update(self, x: torch.Tensor, s: torch.Tensor, t: torch.Tensor,
                               context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """First-order DPM-Solver update"""
        ns = self.noise_prediction_fn(x, s, context)
        
        sigma_s = self.t_to_sigma(self.get_model_input_time(s))
        sigma_t = self.t_to_sigma(self.get_model_input_time(t))
        
        while sigma_s.dim() < x.dim():
            sigma_s = sigma_s.unsqueeze(-1)
        while sigma_t.dim() < x.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        
        alpha_s = 1 / (1 + sigma_s**2).sqrt()
        alpha_t = 1 / (1 + sigma_t**2).sqrt()
        
        h = torch.log(alpha_t) - torch.log(alpha_s)
        
        if self.algorithm_type == "dpmsolver":
            x_t = (sigma_t / sigma_s) * x - (alpha_t * (torch.exp(-h) - 1.0)) * ns
        else:  # dpmsolver++
            x_t = (alpha_t / alpha_s) * x - (sigma_t * (torch.exp(h) - 1.0)) * ns
            
        return x_t
    
    def multistep_dpm_solver_second_update(self, x: torch.Tensor, model_prev_list: List[torch.Tensor],
                                          t_prev_list: List[torch.Tensor], t: torch.Tensor,
                                          context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Second-order multistep DPM-Solver update"""
        ns = self.noise_prediction_fn(x, t_prev_list[-1], context)
        model_prev_list.append(ns)
        
        sigma_prev_0 = self.t_to_sigma(self.get_model_input_time(t_prev_list[-2]))
        sigma_prev_1 = self.t_to_sigma(self.get_model_input_time(t_prev_list[-1]))
        sigma_t = self.t_to_sigma(self.get_model_input_time(t))
        
        while sigma_prev_0.dim() < x.dim():
            sigma_prev_0 = sigma_prev_0.unsqueeze(-1)
        while sigma_prev_1.dim() < x.dim():
            sigma_prev_1 = sigma_prev_1.unsqueeze(-1)
        while sigma_t.dim() < x.dim():
            sigma_t = sigma_t.unsqueeze(-1)
        
        alpha_prev_0 = 1 / (1 + sigma_prev_0**2).sqrt()
        alpha_prev_1 = 1 / (1 + sigma_prev_1**2).sqrt()
        alpha_t = 1 / (1 + sigma_t**2).sqrt()
        
        h_0 = torch.log(alpha_prev_1) - torch.log(alpha_prev_0)
        h_1 = torch.log(alpha_t) - torch.log(alpha_prev_1)
        r1 = h_1 / h_0
        
        D1_0 = model_prev_list[-1]
        D1_1 = (1.0 / r1) * (model_prev_list[-1] - model_prev_list[-2])
        
        if self.algorithm_type == "dpmsolver":
            x_t = (sigma_t / sigma_prev_1) * x - (alpha_t * (torch.exp(-h_1) - 1.0)) * D1_0 - 0.5 * (alpha_t * (torch.exp(-h_1) - 1.0)) * D1_1
        else:  # dpmsolver++
            x_t = (alpha_t / alpha_prev_1) * x - (sigma_t * (torch.exp(h_1) - 1.0)) * D1_0 - 0.5 * (sigma_t * (torch.exp(h_1) - 1.0)) * D1_1
            
        return x_t
    
    def sample(self, shape: Tuple[int, ...], 
               context: Optional[torch.Tensor] = None,
               num_steps: int = 20,
               order: int = 2,
               progress: bool = True) -> torch.Tensor:
        """Main sampling method"""
        device = next(self.model.parameters()).device
        
        # Initialize with noise
        x = torch.randn(shape, device=device)
        
        # Get time steps
        timesteps = self.get_time_steps(num_steps, device)
        
        # Initialize for multistep
        model_prev_list = []
        t_prev_list = []
        
        iterator = tqdm(enumerate(timesteps), total=len(timesteps), disable=not progress)
        iterator.set_description("DPM-Solver Sampling")
        
        for i, t in iterator:
            t_prev_list.append(t)
            
            if i == 0:
                # First step - always use first-order
                if i + 1 < len(timesteps):
                    x = self.dpm_solver_first_update(x, t, timesteps[i + 1], context)
                else:
                    # Final step
                    t_next = torch.tensor([0.0], device=device)
                    x = self.dpm_solver_first_update(x, t, t_next, context)
            elif len(t_prev_list) < order:
                # Not enough previous steps for high-order
                if i + 1 < len(timesteps):
                    x = self.dpm_solver_first_update(x, t, timesteps[i + 1], context)
                else:
                    t_next = torch.tensor([0.0], device=device)
                    x = self.dpm_solver_first_update(x, t, t_next, context)
            else:
                # Use multistep update
                if i + 1 < len(timesteps):
                    x = self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, timesteps[i + 1], context)
                else:
                    t_next = torch.tensor([0.0], device=device)
                    x = self.multistep_dpm_solver_second_update(x, model_prev_list, t_prev_list, t_next, context)
            
            # Keep only necessary history
            if len(t_prev_list) >= order:
                model_prev_list = model_prev_list[-(order-1):]
                t_prev_list = t_prev_list[-(order-1):]
        
        return x

class HeunSampler:
    """Heun's method for high-quality sampling (Karras et al.)"""
    
    def __init__(self, model, sigma_min: float = 0.0292, sigma_max: float = 14.6146):
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def get_sigmas_karras(self, n: int, device: torch.device, rho: float = 7.0) -> torch.Tensor:
        """Karras et al. sigma schedule"""
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = self.sigma_min ** (1 / rho)
        max_inv_rho = self.sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])
    
    def sample(self, shape: Tuple[int, ...],
               context: Optional[torch.Tensor] = None,
               num_steps: int = 50,
               s_churn: float = 0.0,
               s_tmin: float = 0.0,
               s_tmax: float = float('inf'),
               s_noise: float = 1.0,
               progress: bool = True) -> torch.Tensor:
        """Heun sampling with stochasticity control"""
        device = next(self.model.parameters()).device
        
        # Get sigma schedule
        sigmas = self.get_sigmas_karras(num_steps, device)
        
        # Initialize
        x = torch.randn(shape, device=device) * sigmas[0]
        
        iterator = tqdm(enumerate(sigmas[:-1]), total=len(sigmas)-1, disable=not progress)
        iterator.set_description("Heun Sampling")
        
        for i, sigma in iterator:
            gamma = min(s_churn / num_steps, 2 ** 0.5 - 1) if s_tmin <= sigma <= s_tmax else 0.0
            
            # Add noise
            if gamma > 0:
                eps = torch.randn_like(x) * s_noise
                sigma_hat = sigma * (1 + gamma)
                x = x + eps * (sigma_hat ** 2 - sigma ** 2) ** 0.5
            else:
                sigma_hat = sigma
            
            # Denoising step
            t = self.sigma_to_t(sigma_hat)
            denoised = self.model(x.transpose(-1, -2), t, context).transpose(-1, -2)
            d = (x - denoised) / sigma_hat
            dt = sigmas[i + 1] - sigma_hat
            
            # Heun's method
            if sigmas[i + 1] == 0:
                # Last step
                x = x + d * dt
            else:
                # Predictor
                x_2 = x + d * dt
                # Corrector
                t_2 = self.sigma_to_t(sigmas[i + 1])
                denoised_2 = self.model(x_2.transpose(-1, -2), t_2, context).transpose(-1, -2)
                d_2 = (x_2 - denoised_2) / sigmas[i + 1]
                # Average derivatives
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
        
        return x
    
    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep (placeholder - should be implemented based on noise schedule)"""
        # This is a simplified conversion - in practice, you'd use the actual noise schedule
        return (sigma * 1000).long().clamp(0, 999)

class EDMSampler:
    """Elucidating the Design Space of Diffusion-Based Generative Models (Karras et al. 2022)"""
    
    def __init__(self, model, sigma_min: float = 0.002, sigma_max: float = 80.0):
        self.model = model
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def get_sigmas_edm(self, n: int, device: torch.device) -> torch.Tensor:
        """EDM sigma schedule"""
        ramp = torch.linspace(0, 1, n, device=device)
        sigmas = (self.sigma_max ** (1/7) + ramp * (self.sigma_min ** (1/7) - self.sigma_max ** (1/7))) ** 7
        return torch.cat([sigmas, torch.zeros(1, device=device)])
    
    def sample(self, shape: Tuple[int, ...],
               context: Optional[torch.Tensor] = None,
               num_steps: int = 50,
               sigma_min: Optional[float] = None,
               sigma_max: Optional[float] = None,
               rho: float = 7.0,
               S_churn: float = 0.0,
               S_min: float = 0.0,
               S_max: float = float('inf'),
               S_noise: float = 1.0,
               progress: bool = True) -> torch.Tensor:
        """EDM sampling algorithm"""
        device = next(self.model.parameters()).device
        
        sigma_min = sigma_min or self.sigma_min
        sigma_max = sigma_max or self.sigma_max
        
        # Time step discretization
        step_indices = torch.arange(num_steps, device=device)
        t_steps = (sigma_max ** (1/rho) + step_indices / (num_steps - 1) * 
                  (sigma_min ** (1/rho) - sigma_max ** (1/rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])
        
        # Initialize
        x_next = torch.randn(shape, device=device) * t_steps[0]
        
        iterator = tqdm(enumerate(zip(t_steps[:-1], t_steps[1:])), total=num_steps, disable=not progress)
        iterator.set_description("EDM Sampling")
        
        for i, (t_cur, t_next) in iterator:
            x_cur = x_next
            
            # Increase noise temporarily
            gamma = min(S_churn / num_steps, 2 ** 0.5 - 1) if S_min <= t_cur <= S_max else 0
            t_hat = t_cur + gamma * t_cur
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2) ** 0.5 * torch.randn_like(x_cur) * S_noise
            
            # Euler step
            t_input = self.sigma_to_t(t_hat)
            denoised = self.model(x_hat.transpose(-1, -2), t_input, context).transpose(-1, -2)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur
            
            # Apply 2nd order correction
            if i < num_steps - 1:
                t_input_next = self.sigma_to_t(t_next)
                denoised_next = self.model(x_next.transpose(-1, -2), t_input_next, context).transpose(-1, -2)
                d_prime = (x_next - denoised_next) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
        
        return x_next
    
    def sigma_to_t(self, sigma: torch.Tensor) -> torch.Tensor:
        """Convert sigma to timestep"""
        return (sigma * 250).long().clamp(0, 999)
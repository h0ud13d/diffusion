# advanced_diffusion.py - State-of-the-art diffusion model for quantitative finance
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from tqdm import tqdm

@dataclass
class AdvancedDiffusionConfig:
    """Configuration for advanced diffusion model"""
    # Core diffusion parameters
    timesteps: int = 1000
    beta_schedule: str = "cosine"  # cosine, linear, quadratic, sigmoid
    prediction_type: str = "v"  # epsilon, x0, v
    
    # Advanced features
    classifier_free_guidance: bool = True
    guidance_scale: float = 7.5
    guidance_rescale: float = 0.0  # Guidance rescaling factor
    
    # Sampling improvements
    use_karras_sigmas: bool = True  # Improved noise scheduling
    sigma_min: float = 0.0292
    sigma_max: float = 14.6146
    rho: float = 7.0  # Karras et al. parameter
    
    # Training improvements
    min_snr_gamma: float = 5.0  # Min-SNR weighting
    use_offset_noise: bool = True  # Offset noise for better dark/light generation
    noise_offset: float = 0.1
    
    # Conditioning
    drop_prob: float = 0.1  # Unconditional training probability
    cross_attention_dim: int = 768
    
    # Advanced loss weighting
    loss_type: str = "huber"  # mse, mae, huber, focal
    snr_weighting: bool = True  # Signal-to-noise ratio weighting

def get_beta_schedule(schedule: str, timesteps: int, **kwargs) -> torch.Tensor:
    """Advanced beta schedules for noise scheduling"""
    
    if schedule == "cosine":
        # Improved cosine schedule (Nichol & Dhariwal)
        s = kwargs.get("s", 0.008)
        steps = torch.arange(timesteps + 1, dtype=torch.float64)
        x = (steps / timesteps + s) / (1 + s)
        alphas_cumprod = torch.cos(x * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999).float()
        
    elif schedule == "linear":
        beta_start = kwargs.get("beta_start", 0.0001)
        beta_end = kwargs.get("beta_end", 0.02)
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        
    elif schedule == "quadratic":
        beta_start = kwargs.get("beta_start", 0.0001)
        beta_end = kwargs.get("beta_end", 0.02)
        return torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
        
    elif schedule == "sigmoid":
        beta_start = kwargs.get("beta_start", 0.0001)
        beta_end = kwargs.get("beta_end", 0.02)
        betas = torch.linspace(-6, 6, timesteps, dtype=torch.float32)
        return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start
        
    else:
        raise ValueError(f"Unknown beta schedule: {schedule}")

def get_karras_sigmas(timesteps: int, sigma_min: float, sigma_max: float, rho: float = 7.0) -> torch.Tensor:
    """Karras et al. noise scheduling for improved sampling"""
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + torch.linspace(0, 1, timesteps) * (min_inv_rho - max_inv_rho)) ** rho
    return sigmas

class AdvancedTransformerBlock(nn.Module):
    """Advanced transformer block with financial-specific features"""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, 
                 dropout: float = 0.0, attention_bias: bool = False,
                 use_flash_attention: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.use_flash_attention = use_flash_attention
        
        # Layer normalization (pre-norm for stability)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # Multi-head attention
        self.attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, bias=attention_bias, batch_first=True
        )
        
        # MLP
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
        # Adaptive layer scale (LLaMA-style)
        self.ls1 = nn.Parameter(torch.ones(dim) * 1e-6)
        self.ls2 = nn.Parameter(torch.ones(dim) * 1e-6)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm + self-attention + residual with layer scale
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=attention_mask)
        x = x + self.ls1 * attn_out
        
        # Pre-norm + MLP + residual with layer scale
        x = x + self.ls2 * self.mlp(self.norm2(x))
        
        return x

class CrossAttentionBlock(nn.Module):
    """Cross-attention for conditioning on market features"""
    
    def __init__(self, query_dim: int, context_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = query_dim // num_heads
        
        self.norm_q = nn.LayerNorm(query_dim, eps=1e-6)
        self.norm_k = nn.LayerNorm(context_dim, eps=1e-6)
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        
        # Normalize
        q = self.norm_q(x)
        k = self.norm_k(context)
        v = context
        
        # Project to q, k, v
        q = self.to_q(q).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(k).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(v).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        out = self.to_out(out)
        
        return x + out

class AdvancedDiffusionTransformer(nn.Module):
    """State-of-the-art transformer-based diffusion model"""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = 768,
                 num_layers: int = 12,
                 num_heads: int = 12,
                 max_seq_len: int = 2048,
                 context_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 use_rotary_pos_emb: bool = True):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        self.use_rotary_pos_emb = use_rotary_pos_emb
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Time embedding (improved sinusoidal)
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim // 4),
            nn.Linear(hidden_dim // 4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Position embeddings
        if not use_rotary_pos_emb:
            self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.02)
        else:
            self.rotary_emb = RotaryPositionalEmbedding(hidden_dim // num_heads)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            AdvancedTransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Cross-attention blocks (if conditioning)
        if context_dim is not None:
            self.cross_attn_blocks = nn.ModuleList([
                CrossAttentionBlock(hidden_dim, context_dim, num_heads, dropout)
                for _ in range(num_layers)
            ])
        else:
            self.cross_attn_blocks = None
        
        # Output layers
        self.norm_out = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Improved weight initialization"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self, x: torch.Tensor, timestep: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Time embedding
        t_emb = self.time_embed(timestep)  # (B, hidden_dim)
        t_emb = t_emb.unsqueeze(1).expand(-1, L, -1)  # (B, L, hidden_dim)
        x = x + t_emb
        
        # Position embeddings
        if not self.use_rotary_pos_emb:
            x = x + self.pos_embed[:, :L]
        
        # Transformer blocks with optional cross-attention
        for i, block in enumerate(self.blocks):
            x = block(x)
            
            # Cross-attention (if available)
            if self.cross_attn_blocks is not None and context is not None:
                x = self.cross_attn_blocks[i](x, context)
        
        # Output projection
        x = self.norm_out(x)
        x = self.output_proj(x)
        
        return x

class SinusoidalPositionEmbedding(nn.Module):
    """Improved sinusoidal position embedding"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        embeddings = torch.log(torch.tensor(self.max_period, device=device)) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = x[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        if self.dim % 2 == 1:
            embeddings = F.pad(embeddings, (0, 1))
        return embeddings

class RotaryPositionalEmbedding(nn.Module):
    """RoPE positional embedding for better sequence modeling"""
    
    def __init__(self, dim: int, max_period: int = 10000):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (max_period ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
    
    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin

class AdvancedDiffusionModel(nn.Module):
    """Complete advanced diffusion model with state-of-the-art features"""
    
    def __init__(self, config: AdvancedDiffusionConfig, model: nn.Module):
        super().__init__()
        self.config = config
        self.model = model
        
        # Noise scheduling
        if config.use_karras_sigmas:
            sigmas = get_karras_sigmas(
                config.timesteps, config.sigma_min, config.sigma_max, config.rho
            )
            alphas_cumprod = 1 / (1 + sigmas**2)
            betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
            betas = torch.cat([torch.tensor([0.0]), betas])
        else:
            betas = get_beta_schedule(config.beta_schedule, config.timesteps)
        
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        
        # Register buffers
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        
        # For v-prediction
        if config.prediction_type == "v":
            self.register_buffer("sqrt_alphas_cumprod_prev", torch.sqrt(
                F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
            ))
    
    def get_velocity(self, x_start: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Convert to v-parameterization"""
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_start.device)[timesteps]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[timesteps]
        
        # Add dimensions for broadcasting
        while len(sqrt_alphas_cumprod.shape) < len(x_start.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
        
        v = sqrt_alphas_cumprod * noise - sqrt_one_minus_alphas_cumprod * x_start
        return v
    
    def predict_start_from_v(self, x_t: torch.Tensor, v: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from v-prediction"""
        sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_t.device)[t]
        sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_t.device)[t]
        
        while len(sqrt_alphas_cumprod.shape) < len(x_t.shape):
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)
        
        return sqrt_alphas_cumprod * x_t - sqrt_one_minus_alphas_cumprod * v
    
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process with optional offset noise"""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Add offset noise for better performance
        if self.config.use_offset_noise and self.training:
            offset = torch.randn(x_start.shape[0], 1, 1, device=x_start.device)
            noise = noise + self.config.noise_offset * offset
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.to(x_start.device)[t]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)[t]
        
        while len(sqrt_alphas_cumprod_t.shape) < len(x_start.shape):
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)
            sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod_t.unsqueeze(-1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def compute_loss(self, x_start: torch.Tensor, t: torch.Tensor, 
                    context: Optional[torch.Tensor] = None,
                    noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute training loss with advanced techniques"""
        B = x_start.shape[0]
        device = x_start.device
        
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # Forward process
        x_t = self.q_sample(x_start, t, noise)
        
        # Classifier-free guidance training
        if self.config.classifier_free_guidance and context is not None:
            # Randomly drop context
            mask = torch.rand(B, device=device) < self.config.drop_prob
            if context.dim() == 2:  # (B, context_dim)
                context = context.clone()
                context[mask] = 0
            elif context.dim() == 3:  # (B, seq_len, context_dim)
                context = context.clone()
                context[mask] = 0
        
        # Model prediction
        model_output = self.model(x_t.transpose(-1, -2), t, context)
        model_output = model_output.transpose(-1, -2)
        
        # Target depends on prediction type
        if self.config.prediction_type == "epsilon":
            target = noise
        elif self.config.prediction_type == "x0":
            target = x_start
        elif self.config.prediction_type == "v":
            target = self.get_velocity(x_start, noise, t)
        else:
            raise ValueError(f"Unknown prediction type: {self.config.prediction_type}")
        
        # Loss computation
        if self.config.loss_type == "mse":
            loss = F.mse_loss(model_output, target, reduction="none")
        elif self.config.loss_type == "mae":
            loss = F.l1_loss(model_output, target, reduction="none")
        elif self.config.loss_type == "huber":
            loss = F.huber_loss(model_output, target, reduction="none", delta=1.0)
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")
        
        # Advanced loss weighting
        if self.config.snr_weighting:
            # Min-SNR weighting (Hang et al.)
            snr = self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])
            snr_weight = torch.minimum(snr, torch.tensor(self.config.min_snr_gamma, device=device))
            while len(snr_weight.shape) < len(loss.shape):
                snr_weight = snr_weight.unsqueeze(-1)
            loss = loss * snr_weight
        
        return loss.mean()
    
    @torch.no_grad()
    def ddim_step(self, x_t: torch.Tensor, t: int, t_prev: int, 
                  context: Optional[torch.Tensor] = None, 
                  eta: float = 0.0) -> torch.Tensor:
        """DDIM sampling step with classifier-free guidance"""
        
        timestep = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        
        # Model prediction with guidance
        if self.config.classifier_free_guidance and context is not None:
            # Unconditional
            model_output_uncond = self.model(x_t.transpose(-1, -2), timestep, None)
            model_output_uncond = model_output_uncond.transpose(-1, -2)
            
            # Conditional
            model_output_cond = self.model(x_t.transpose(-1, -2), timestep, context)
            model_output_cond = model_output_cond.transpose(-1, -2)
            
            # Apply guidance
            guidance_scale = self.config.guidance_scale
            model_output = model_output_uncond + guidance_scale * (model_output_cond - model_output_uncond)
            
            # Guidance rescaling
            if self.config.guidance_rescale > 0:
                std_pos = model_output_cond.std()
                std_cfg = model_output.std()
                factor = std_pos / std_cfg
                factor = self.config.guidance_rescale * factor + (1 - self.config.guidance_rescale) * 1.0
                model_output = model_output * factor
        else:
            model_output = self.model(x_t.transpose(-1, -2), timestep, context)
            model_output = model_output.transpose(-1, -2)
        
        # Convert model output to x0 prediction
        if self.config.prediction_type == "epsilon":
            alpha_t = self.alphas_cumprod[t]
            pred_x0 = (x_t - torch.sqrt(1 - alpha_t) * model_output) / torch.sqrt(alpha_t)
        elif self.config.prediction_type == "x0":
            pred_x0 = model_output
        elif self.config.prediction_type == "v":
            pred_x0 = self.predict_start_from_v(x_t, model_output, timestep)
        
        # DDIM step
        if t_prev < 0:
            return pred_x0
        
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t_prev]
        
        sigma_t = eta * torch.sqrt((1 - alpha_t_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_t_prev))
        
        pred_dir = torch.sqrt(1 - alpha_t_prev - sigma_t**2) * model_output
        x_prev = torch.sqrt(alpha_t_prev) * pred_x0 + pred_dir
        
        if eta > 0:
            noise = torch.randn_like(x_t)
            x_prev = x_prev + sigma_t * noise
        
        return x_prev
    
    @torch.no_grad()
    def sample_ddim(self, shape: Tuple[int, ...], 
                   context: Optional[torch.Tensor] = None,
                   num_steps: int = 50,
                   eta: float = 0.0,
                   progress: bool = True) -> torch.Tensor:
        """DDIM sampling with optional conditioning"""
        
        device = next(self.model.parameters()).device
        
        # Create sampling schedule
        step_size = self.config.timesteps // num_steps
        timesteps = list(range(self.config.timesteps - 1, -1, -step_size))
        timesteps = timesteps[:num_steps]
        
        # Initialize with noise
        x = torch.randn(shape, device=device)
        
        iterator = tqdm(enumerate(timesteps), total=len(timesteps), disable=not progress)
        iterator.set_description("DDIM Sampling")
        
        for i, t in iterator:
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else -1
            x = self.ddim_step(x, t, t_prev, context, eta)
        
        return x
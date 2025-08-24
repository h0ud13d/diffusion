# enhanced_model.py - Advanced UNet with attention mechanisms and multi-scale features
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        t = t.float()
        freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=t.device) / (half - 1))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        assert self.head_dim * num_heads == channels
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv1d(channels, 3 * channels, 1)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        residual = x
        x = self.norm(x)
        
        qkv = self.qkv(x).view(B, 3, self.num_heads, self.head_dim, L)
        q, k, v = qkv.unbind(1)  # (B, H, D, L)
        
        # Compute attention
        scale = self.head_dim ** -0.5
        attn = (q.transpose(-2, -1) @ k) * scale  # (B, H, L, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v.transpose(-2, -1)).transpose(-2, -1)  # (B, H, D, L)
        out = out.contiguous().view(B, C, L)
        out = self.proj(out)
        
        return residual + out

class CrossAttention(nn.Module):
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8, dropout: float = 0.0):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.norm_context = nn.LayerNorm(context_dim)
        
        self.to_q = nn.Conv1d(channels, channels, 1)
        self.to_kv = nn.Linear(context_dim, 2 * channels)
        self.proj = nn.Conv1d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        residual = x
        
        x = self.norm(x)
        context = self.norm_context(context)  # (B, context_dim)
        
        q = self.to_q(x).view(B, self.num_heads, self.head_dim, L)  # (B, H, D, L)
        kv = self.to_kv(context).view(B, 2, self.num_heads, self.head_dim)  # (B, 2, H, D)
        k, v = kv.unbind(1)  # Each is (B, H, D)
        
        scale = self.head_dim ** -0.5
        
        # Compute attention weights: (B, H, L) @ (B, H, D) -> (B, H, L)
        q_reshaped = q.transpose(-2, -1)  # (B, H, L, D)
        k_reshaped = k.unsqueeze(2)  # (B, H, 1, D)
        attn = (q_reshaped @ k_reshaped.transpose(-2, -1)).squeeze(-1) * scale  # (B, H, L)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values: (B, H, L) * (B, H, D) -> (B, H, D, L)
        v_expanded = v.unsqueeze(-1).expand(-1, -1, -1, L)  # (B, H, D, L)
        attn_expanded = attn.unsqueeze(2)  # (B, H, 1, L)
        out = (attn_expanded * v_expanded).sum(dim=1)  # Sum over heads: (B, D, L)
        
        # Ensure correct output shape
        if out.shape[1] != C:
            out = out[:, :C, :]  # Truncate if necessary
        
        out = self.proj(out)
        return residual + out

class EnhancedResBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, context_dim: Optional[int] = None, 
                 dropout: float = 0.0, use_attention: bool = False, num_heads: int = 8):
        super().__init__()
        self.use_attention = use_attention
        
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.SiLU()
        
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        
        self.time_proj = nn.Linear(time_dim, channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        if use_attention:
            self.self_attn = MultiHeadSelfAttention(channels, num_heads, dropout)
        
        if context_dim is not None:
            self.cross_attn = CrossAttention(channels, context_dim, num_heads, dropout)
        else:
            self.cross_attn = None
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        residual = x
        
        # First conv block
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        
        # Second conv block
        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)
        
        # Add residual
        h = h + residual
        
        # Self-attention
        if self.use_attention:
            h = self.self_attn(h)
        
        # Cross-attention
        if self.cross_attn is not None and context is not None:
            h = self.cross_attn(h, context)
        
        return h

class DownsampleBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, context_dim: Optional[int] = None,
                 dropout: float = 0.0, use_attention: bool = False, num_heads: int = 8):
        super().__init__()
        self.downsample = nn.Conv1d(channels, channels, 3, stride=2, padding=1)
        self.resblock = EnhancedResBlock(channels, time_dim, context_dim, dropout, 
                                        use_attention, num_heads)
    
    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.downsample(x)
        return self.resblock(x, time_emb, context)

class UpsampleBlock(nn.Module):
    def __init__(self, channels: int, time_dim: int, context_dim: Optional[int] = None,
                 dropout: float = 0.0, use_attention: bool = False, num_heads: int = 8):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)
        self.resblock = EnhancedResBlock(channels, time_dim, context_dim, dropout, 
                                        use_attention, num_heads)
    
    def forward(self, x: torch.Tensor, skip: torch.Tensor, time_emb: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-1] != skip.shape[-1]:
            min_len = min(x.shape[-1], skip.shape[-1])
            x = x[..., :min_len]
            skip = skip[..., :min_len]
        x = x + skip
        return self.resblock(x, time_emb, context)

class EnhancedUNet1D(nn.Module):
    def __init__(self, 
                 c_in: int, 
                 c_out: int, 
                 base_channels: int = 64,
                 channel_multipliers: List[int] = [1, 2, 4, 8],
                 time_dim: int = 256,
                 context_dim: Optional[int] = None,
                 dropout: float = 0.0,
                 attention_resolutions: List[int] = [1, 2],
                 num_heads: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        
        self.c_in = c_in
        self.c_out = c_out
        self.use_checkpoint = use_checkpoint
        
        # Time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_dim // 4)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim // 4, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Input projection
        self.input_conv = nn.Conv1d(c_in, base_channels, 3, padding=1)
        
        # Calculate channels for each level
        channels = [base_channels * mult for mult in channel_multipliers]
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        in_ch = base_channels
        for i, out_ch in enumerate(channels):
            use_attention = i in attention_resolutions
            
            # ResBlock at current resolution
            self.encoder_blocks.append(
                EnhancedResBlock(in_ch, time_dim, context_dim, dropout, use_attention, num_heads)
            )
            
            # Downsample to next resolution
            if i < len(channels) - 1:
                self.encoder_downs.append(
                    DownsampleBlock(in_ch, time_dim, context_dim, dropout, use_attention, num_heads)
                )
                in_ch = out_ch
        
        # Middle block
        mid_ch = channels[-1]
        self.middle_blocks = nn.ModuleList([
            EnhancedResBlock(mid_ch, time_dim, context_dim, dropout, True, num_heads),
            EnhancedResBlock(mid_ch, time_dim, context_dim, dropout, True, num_heads)
        ])
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        
        for i, out_ch in enumerate(reversed(channels[:-1])):
            use_attention = (len(channels) - 2 - i) in attention_resolutions
            
            # Upsample from previous resolution
            self.decoder_ups.append(
                UpsampleBlock(mid_ch, time_dim, context_dim, dropout, use_attention, num_heads)
            )
            
            # ResBlock at current resolution
            self.decoder_blocks.append(
                EnhancedResBlock(out_ch, time_dim, context_dim, dropout, use_attention, num_heads)
            )
            
            mid_ch = out_ch
        
        # Output projection
        self.output_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv1d(base_channels, c_out, 3, padding=1)
        )
    
    def forward(self, x: torch.Tensor, t: torch.Tensor, 
                context: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Time embedding
        time_emb = self.time_mlp(self.time_embedding(t))
        
        # Input projection
        h = self.input_conv(x)
        
        # Store skip connections
        skip_connections = []
        
        # Encoder
        for i, (block, down) in enumerate(zip(self.encoder_blocks, self.encoder_downs)):
            h = block(h, time_emb, context)
            skip_connections.append(h)
            h = down(h, time_emb, context)
        
        # Handle last encoder block (no downsampling)
        if len(self.encoder_blocks) > len(self.encoder_downs):
            h = self.encoder_blocks[-1](h, time_emb, context)
            skip_connections.append(h)
        
        # Middle blocks
        for block in self.middle_blocks:
            h = block(h, time_emb, context)
        
        # Decoder
        for up, block in zip(self.decoder_ups, self.decoder_blocks):
            skip = skip_connections.pop()
            h = up(h, skip, time_emb, context)
            h = block(h, time_emb, context)
        
        # Output projection
        return self.output_conv(h)

class ConditionEncoder(nn.Module):
    """Encodes additional conditioning information (volatility, momentum, etc.)"""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, conditions: torch.Tensor) -> torch.Tensor:
        return self.encoder(conditions)
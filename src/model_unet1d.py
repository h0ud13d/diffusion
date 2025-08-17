# model_unet1d.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        t = t.float()
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

def conv3(in_ch, out_ch):
    return nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)

class ResBlock(nn.Module):
    def __init__(self, ch, tdim, drop=0.0):
        super().__init__()
        self.n1 = nn.GroupNorm(8, ch)
        self.n2 = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()
        self.c1 = conv3(ch, ch)
        self.c2 = conv3(ch, ch)
        self.tproj = nn.Linear(tdim, ch)
        self.drop = nn.Dropout(drop) if drop > 0 else nn.Identity()
    def forward(self, x, temb):
        h = self.act(self.n1(x))
        h = self.c1(h)
        h = h + self.tproj(temb).unsqueeze(-1)
        h = self.act(self.n2(h))
        h = self.drop(h)
        h = self.c2(h)
        return x + h

class Down(nn.Module):
    def __init__(self, ch, tdim, drop=0.0):
        super().__init__()
        self.pool = nn.Conv1d(ch, ch, kernel_size=2, stride=2)
        self.res = ResBlock(ch, tdim, drop)
    def forward(self, x, temb):
        x = self.pool(x)
        return self.res(x, temb)

class Up(nn.Module):
    def __init__(self, ch, tdim, drop=0.0):
        super().__init__()
        self.up = nn.ConvTranspose1d(ch, ch, kernel_size=2, stride=2)
        self.res = ResBlock(ch, tdim, drop)
    def forward(self, x, skip, temb):
        x = self.up(x)
        if x.shape[-1] > skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        if skip.shape[-1] > x.shape[-1]:
            skip = skip[..., :x.shape[-1]]
        x = x + skip
        return self.res(x, temb)

class UNet1D(nn.Module):
    def __init__(self, c_in, c_out, base=64, tdim=128, drop=0.0):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.time_mlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(), nn.Linear(tdim, tdim))

        self.inp = conv3(c_in, base)
        self.b1 = ResBlock(base, tdim, drop)
        self.d1 = Down(base, tdim, drop)
        self.b2 = ResBlock(base, tdim, drop)
        self.d2 = Down(base, tdim, drop)
        self.b3 = ResBlock(base, tdim, drop)

        self.mid = ResBlock(base, tdim, drop)

        self.u2 = Up(base, tdim, drop)
        self.u1 = Up(base, tdim, drop)
        self.u0 = Up(base, tdim, drop)

        self.out = nn.Sequential(nn.GroupNorm(8, base), nn.SiLU(), conv3(base, c_out))

    def forward(self, x, t):
        temb = self.time_mlp(self.time(t))
        h0 = self.inp(x)
        h1 = self.b1(h0, temb)
        h2 = self.d1(h1, temb)
        h3 = self.b2(h2, temb)
        h4 = self.d2(h3, temb)
        h5 = self.b3(h4, temb)
        hm = self.mid(h5, temb)
        u2 = self.u2(hm, h5, temb)
        u1 = self.u1(u2, h3, temb)
        u0 = self.u0(u1, h1, temb)
        return self.out(u0)


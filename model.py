import torch
from torch import nn, einsum
from einops import rearrange, repeat
import copy

class ConvNeXt(nn.Module):
    def __init__(self, in_channels, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x

# PreNorm layer
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

# FeedForward layer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# KernelAttention layer
class KernelAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads > 0 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, kx, krd, clst, att_mask=None, l_debug_idx=0):
        c_qkv = self.to_qkv(x).chunk(3, dim=-1)
        k_kqv = self.to_qkv(kx).chunk(3, dim=-1)
        c_kqv = self.to_qkv(clst).chunk(3, dim=-1)

        t_q, t_k, t_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), c_qkv)
        k_q, k_k, k_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), k_kqv)
        c_q, _, _ = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), c_kqv)

        # Information summary flow (ISF) -- Eq.2
        dots = einsum('b h i d, b h j d -> b h i j', t_q, k_k) * self.scale
        if att_mask is not None:
            dots = dots.masked_fill(att_mask, torch.tensor(-1e9))
        attn = self.attend(dots) * krd.permute(0, 1, 3, 2)
        att_out = einsum('b h i j, b h j d -> b h i d', attn, k_v)
        att_out = rearrange(att_out, 'b h n d -> b n (h d)')

        # Information distribution flow (IDF) -- Eq.3
        k_dots = einsum('b h i d, b h j d -> b h i j', k_q, t_k) * self.scale
        if att_mask is not None:
            k_dots = k_dots.masked_fill(att_mask.permute(0, 1, 3, 2), torch.tensor(-1e9))
        k_attn = self.attend(k_dots) * krd
        k_out = einsum('b h i j, b h j d -> b h i d', k_attn, t_v)
        k_out = rearrange(k_out, 'b h n d -> b n (h d)')

        # Classification token -- Eq.4
        c_dots = einsum('b h i d, b h j d -> b h i j', c_q, k_k) * self.scale
        if att_mask is not None:
            c_dots = c_dots.masked_fill(att_mask[:, :, :1], torch.tensor(-1e9))
        c_attn = self.attend(c_dots)
        c_out = einsum('b h i j, b h j d -> b h i d', c_attn, k_v)
        c_out = rearrange(c_out, 'b h n d -> b n (h d)')

        return self.to_out(att_out), self.to_out(k_out), self.to_out(c_out)


def kat_inference(kat_model, data):
    feats = data[0].float().cuda(non_blocking=True)
    rd = data[1].float().cuda(non_blocking=True)
    masks = data[2].int().cuda(non_blocking=True)
    kmasks = data[3].int().cuda(non_blocking=True)

    return kat_model(feats, rd, masks, kmasks)


# KATBlocks layer
class KATBlocks(nn.Module):
    def __init__(self, npk, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ms = npk  # Initial scale factor of the Gaussian mask

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                KernelAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
            ]))
        self.h = heads
        self.dim = dim

    def forward(self, x, kx, rd, clst, mask=None, kmask=None):
        kernel_mask = repeat(kmask, 'b i ()  -> b i j', j=self.dim) < 0.5
        att_mask = einsum('b i d, b j d -> b i j', mask.float(), kmask.float())
        att_mask = repeat(att_mask.unsqueeze(1), 'b () i j -> b h i j', h=self.h) < 0.5

        rd = repeat(rd.unsqueeze(1), 'b () i j -> b h i j', h=self.h)
        rd2 = rd * rd

        k_reps = []
        for l_idx, (pn, attn, ff) in enumerate(self.layers):
            x, kx, clst = pn(x), pn(kx), pn(clst)

            soft_mask = torch.exp(-rd2 / (2 * self.ms * 2 ** l_idx))
            x_, kx_, clst_ = attn(x, kx, soft_mask, clst, att_mask, l_idx)
            x = x + x_
            clst = clst + clst_
            kx = kx + kx_

            x = ff(x) + x
            clst = ff(clst) + clst
            kx = ff(kx) + kx

            k_reps.append(kx.masked_fill(kernel_mask, 0))

        return k_reps, clst

# KAT model
class KAT(nn.Module):
    def __init__(self, num_pk, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_kernal=16, pool='cls',
                 dim_head=64, dropout=0.5, emb_dropout=0.):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Linear(patch_dim, dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.kernel_token = nn.Parameter(torch.randn(1, 1, dim))
        self.nk = num_kernal

        self.dropout = nn.Dropout(emb_dropout)

        self.kt = KATBlocks(num_pk, dim, depth, heads, dim_head, mlp_dim, dropout)
        self.convnext = ConvNeXt(in_channels=8, dim=dim) 

        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, node_features, krd, mask=None, kmask=None):
        x = self.to_patch_embedding(node_features)
        b = x.shape[0]

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        kernel_tokens = repeat(self.kernel_token, '() () d -> b k d', b=b, k=self.nk)

        x = self.dropout(x)

        # Example usage of ConvNeXt
        x = rearrange(x, 'b d n -> b n d')  # Assuming input is in the format (batch_size, channels, height, width)
        x = self.convnext(x)

        k_reps, clst = self.kt(x, kernel_tokens, krd, cls_tokens, mask, kmask)

        return k_reps

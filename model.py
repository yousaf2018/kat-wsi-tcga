import torch
from torch import nn
from einops import repeat
import copy

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        original_shape = x.shape
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        x = x.view(original_shape)  # Restore original shape
        return x

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

        dots = einsum('b h i d, b h j d -> b h i j', t_q, k_k) * self.scale
        if att_mask is not None:
            dots = dots.masked_fill(att_mask, torch.tensor(-1e9))
        attn = self.attend(dots) * krd.permute(0, 1, 3, 2)
        att_out = einsum('b h i j, b h j d -> b h i d', attn, k_v)
        att_out = rearrange(att_out, 'b h n d -> b n (h d)')

        k_dots = einsum('b h i d, b h j d -> b h i j', k_q, t_k) * self.scale
        if att_mask is not None:
            k_dots = k_dots.masked_fill(att_mask.permute(0, 1, 3, 2), torch.tensor(-1e9))
        k_attn = self.attend(k_dots) * krd
        k_out = einsum('b h i j, b h j d -> b h i d', k_attn, t_v)
        k_out = rearrange(k_out, 'b h n d -> b n (h d)')

        c_dots = einsum('b h i d, b h j d -> b h i j', c_q, k_k) * self.scale
        if att_mask is not None:
            c_dots = c_dots.masked_fill(att_mask[:, :, :1], torch.tensor(-1e9))
        c_attn = self.attend(c_dots)
        c_out = einsum('b h i j, b h j d -> b h i d', c_attn, k_v)
        c_out = rearrange(c_out, 'b h n d -> b n (h d)')

        return self.to_out(att_out), self.to_out(k_out), self.to_out(c_out)

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Dropout(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)  # Convert (B, C, H, W) to (B, HW, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2).reshape(input.shape)  # Convert (B, HW, C) back to (B, C, H, W)
        x = input + self.drop_path(x)
        return x

class KATBlocks(nn.Module):
    def __init__(self, npk, dim, depth, heads, dim_head, mlp_dim, dropout=0., use_convnext=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.ms = npk  # initial scale factor of the Gaussian mask
        self.use_convnext = use_convnext

        for _ in range(depth):
            convnext_block = ConvNeXtBlock(dim) if use_convnext else nn.Identity()
            self.layers.append(nn.ModuleList([
                convnext_block,
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
        for l_idx, (convnext, pn, attn, ff) in enumerate(self.layers):
            if self.use_convnext:
                # Reshape and permute to apply ConvNeXt
                x = convnext(x.permute(0, 2, 1).reshape(x.shape[0], self.dim, 1, -1)).reshape(x.shape[0], -1, self.dim).permute(0, 2, 1)
            # Apply LayerNorm correctly
            x, kx, clst = pn(x.permute(0, 2, 1)).permute(0, 2, 1), pn(kx.permute(0, 2, 1)).permute(0, 2, 1), pn(clst.permute(0, 2, 1)).permute(0, 2, 1)

            soft_mask = torch.exp(-rd2 / (2*self.ms * 2**l_idx))
            x_, kx_, clst_ = attn(x, kx, soft_mask, clst, att_mask, l_idx)
            x = x + x_
            clst = clst + clst_
            kx = kx + kx_

            x = ff(x) + x
            clst = ff(clst) + clst
            kx = ff(kx) + kx

            k_reps.append(kx.masked_fill(kernel_mask, 0))

        return k_reps, clst

class KAT(nn.Module):
    def __init__(self, num_pk, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_kernal=16, pool='cls', dim_head=64, dropout=0.5, emb_dropout=0., use_convnext=True):
        super().__init__()

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Example of correct input dimensions handling in KAT model
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=dim, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            # nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Flatten(start_dim=2)
        )



        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.kernel_token = nn.Parameter(torch.randn(1, 1, dim))
        self.nk = num_kernal

        self.dropout = nn.Dropout(emb_dropout)

        self.kt = KATBlocks(num_pk, dim, depth, heads, dim_head, mlp_dim, dropout, use_convnext)
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
        k_reps, clst = self.kt(x, kernel_tokens, krd, cls_tokens, mask, kmask)

        return k_reps, self.mlp_head(clst[:, 0])
def kat_inference(kat_model, data):
    feats = data[0].float().cuda(non_blocking=True)
    rd = data[1].float().cuda(non_blocking=True)
    masks = data[2].int().cuda(non_blocking=True)
    kmasks = data[3].int().cuda(non_blocking=True)

    return kat_model(feats, rd, masks, kmasks)

class KATCL(nn.Module):
    """
    Build a BYOL model for the kernels.
    """
    def __init__(self, num_pk, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_kernal=16, pool='cls', dim_head=64, dropout=0.5, emb_dropout=0.,
                 byol_hidden_dim=512, byol_pred_dim=256, momentum=0.99, use_convnext=True):

        super(KATCL, self).__init__()

        self.momentum = momentum
        # create the online encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.online_kat = KAT(num_pk, patch_dim, num_classes, dim, depth, heads, mlp_dim, num_kernal, pool, dim_head, dropout, emb_dropout, use_convnext)
        self.online_projector = nn.Sequential(nn.Linear(dim, byol_hidden_dim, bias=False),
                                              nn.LayerNorm(byol_hidden_dim),
                                              nn.GELU(),
                                              nn.Dropout(dropout),
                                              nn.Linear(byol_hidden_dim, byol_pred_dim))  # output layer

        # create the target encoder
        self.target_kat = copy.deepcopy(self.online_kat)
        self.target_projector = copy.deepcopy(self.online_projector)

        # freeze target encoder
        for params in self.target_kat.parameters():
            params.requires_grad = False
        for params in self.target_projector.parameters():
            params.requires_grad = False

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(byol_pred_dim, byol_hidden_dim, bias=False),
                                       nn.LayerNorm(byol_hidden_dim),
                                       nn.GELU(),
                                       nn.Dropout(dropout),
                                       nn.Linear(byol_hidden_dim, byol_pred_dim))  # output layer

    @torch.no_grad()
    def _update_moving_average(self):
        for online_params, target_params in zip(self.online_kat.parameters(), self.target_kat.parameters()):
            target_params.data = target_params.data * self.momentum + online_params.data * (1 - self.momentum)

        for online_params, target_params in zip(self.online_projector.parameters(), self.target_projector.parameters()):
            target_params.data = target_params.data * self.momentum + online_params.data * (1 - self.momentum)

    def forward(self, x1, x2):
        # compute features for one view
        online_k1, o1 = self.online_kat(x1)  # Assuming x1 is the node_features in your kat_inference function
        online_z1 = self.online_projector(torch.cat(online_k1, dim=1))
        p1 = self.predictor(online_z1)

        with torch.no_grad():
            self._update_moving_average()
            target_k2, _ = self.target_kat(x2)  # Assuming x2 is the node_features in your kat_inference function
            target_z2 = self.target_projector(torch.cat(target_k2, dim=1))

        return p1, o1, target_z2

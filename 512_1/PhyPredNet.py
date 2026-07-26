import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEmb1D(nn.Module):

    def __init__(self, dim: int, lambda_min: float, lambda_max: float):
        super().__init__()
        assert dim % 2 == 0
        half = dim // 2
        if half == 1:
            lambdas = torch.tensor([lambda_min], dtype=torch.float32)
        else:
            i = torch.arange(half, dtype=torch.float32)
            log_lmin = math.log(lambda_min)
            log_lmax = math.log(lambda_max)
            lambdas = torch.exp(log_lmin + i * (log_lmax - log_lmin) / (half - 1))
        inv_l = 2.0 * math.pi / lambdas
        self.register_buffer('inv_l', inv_l)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.amp.autocast(enabled=False):
            x32 = x.float()
            inv_l32 = self.inv_l.float()
            phase = x32.unsqueeze(-1) * inv_l32
            two_pi = 2.0 * math.pi
            phase = torch.remainder(phase + math.pi, two_pi) - math.pi
            emb = torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)
        return emb.to(dtype=x.dtype)

def downsample_levels(z_full: torch.Tensor, D_emb: int) -> torch.Tensor:
    N = z_full.numel()
    if D_emb == N:
        return z_full
    base = N // D_emb
    rem = N % D_emb
    zs = []
    idx = 0
    for k in range(D_emb):
        length = base + (1 if k < rem else 0)
        z_slice = z_full[idx:idx + length]
        zs.append(z_slice.mean())
        idx += length
    return torch.stack(zs)

class AttentionGate3D(nn.Module):

    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm3d(F_int))
        self.W_x = nn.Sequential(nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True), nn.BatchNorm3d(F_int))
        self.psi = nn.Sequential(nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class ResBlock3D(nn.Module):

    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(out_channels))

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class ComplexUnetBranch(nn.Module):

    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm3d(channels), nn.ReLU(inplace=False), ResBlock3D(channels))
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(nn.Conv3d(channels, channels, kernel_size=3, padding=1), nn.BatchNorm3d(channels), nn.ReLU(inplace=False), ResBlock3D(channels))
        self.up = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(nn.Conv3d(channels * 2, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=False), ResBlock3D(out_channels))

    def forward(self, x):
        x1 = self.conv1(x)
        x_down = self.down(x1)
        x2 = self.conv2(x_down)
        x_up = self.up(x2)
        x_cat = torch.cat([x1, x_up], dim=1)
        return self.conv3(x_cat)

class ComplexTransformerBranch(nn.Module):

    def __init__(self, channels, out_channels, num_heads=4, num_layers=4, dropout=0.1, pool_factor=2):
        super().__init__()
        assert channels % 2 == 0
        self.proj_in = nn.Conv3d(channels, channels, kernel_size=1)
        self.pool = nn.AvgPool3d(kernel_size=(pool_factor // 2, pool_factor, pool_factor), stride=(pool_factor // 2, pool_factor, pool_factor))
        self.upsample = nn.Upsample(scale_factor=(pool_factor // 2, pool_factor, pool_factor), mode='trilinear', align_corners=False)
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Conv3d(channels, out_channels, kernel_size=1)
        half_c = channels // 2
        self.space_lat_emb = FourierEmb1D(half_c, lambda_min=0.01, lambda_max=5.12)
        self.space_lon_emb = FourierEmb1D(half_c, lambda_min=0.01, lambda_max=5.12)
        self.height_emb = FourierEmb1D(channels, lambda_min=500.0, lambda_max=16000.0)
        self.time_emb = FourierEmb1D(channels, lambda_min=1.0, lambda_max=24.0 * 365.0)
        z_low = torch.arange(500.0, 8000.0 + 1e-06, 500.0)
        z_high = torch.arange(9000.0, 16000.0 + 1e-06, 1000.0)
        z_full = torch.cat([z_low, z_high], dim=0)
        assert z_full.numel() == 24
        self.register_buffer('z_full', z_full)

    def _build_physical_encoding(self, x_low: torch.Tensor, lat_range, lon_range, t_steps: torch.Tensor):
        B, C, Dp, Hp, Wp = x_low.shape
        device = x_low.device
        dtype = x_low.dtype
        lat_min, lat_max = (float(lat_range[0]), float(lat_range[1]))
        lon_min, lon_max = (float(lon_range[0]), float(lon_range[1]))
        lat = torch.linspace(lat_min, lat_max, Hp, dtype=dtype, device=device)
        lon = torch.linspace(lon_min, lon_max, Wp, dtype=dtype, device=device)
        emb_lat = self.space_lat_emb(lat)
        emb_lon = self.space_lon_emb(lon)
        e_lat = emb_lat[:, None, :].expand(Hp, Wp, C // 2)
        e_lon = emb_lon[None, :, :].expand(Hp, Wp, C // 2)
        e_space_hw = torch.cat([e_lat, e_lon], dim=-1)
        e_space = e_space_hw[None, :, :, :].expand(Dp, Hp, Wp, C)
        z_full = self.z_full.to(device=device, dtype=dtype)
        z_emb = downsample_levels(z_full, Dp)
        emb_z = self.height_emb(z_emb)
        e_height = emb_z[:, None, None, :].expand(Dp, Hp, Wp, C)
        if not torch.is_tensor(t_steps):
            t_steps = torch.tensor([t_steps], dtype=dtype, device=device)
        else:
            t_steps = t_steps.to(device=device, dtype=dtype)
        if t_steps.dim() == 0:
            t_steps = t_steps[None]
        assert t_steps.shape[0] == B, 't_steps 的 batch 维必须等于 B'
        emb_t = self.time_emb(t_steps)
        e_time = emb_t[:, None, None, None, :].expand(B, Dp, Hp, Wp, C)
        e_space_b = e_space[None, :, :, :, :].expand(B, Dp, Hp, Wp, C)
        e_height_b = e_height[None, :, :, :, :].expand(B, Dp, Hp, Wp, C)
        pos_total = e_space_b + e_height_b + e_time
        pos_flat = pos_total.reshape(B, Dp * Hp * Wp, C)
        return pos_flat

    def forward(self, x, lat_range, lon_range, t_steps: torch.Tensor):
        B, C, D, H, W = x.shape
        x_proj = self.proj_in(x)
        x_low = self.pool(x_proj)
        B, C, Dp, Hp, Wp = x_low.shape
        x_flat = x_low.flatten(2).transpose(1, 2)
        pos_flat = self._build_physical_encoding(x_low, lat_range, lon_range, t_steps)
        tokens = x_flat + pos_flat
        x_trans = self.transformer(tokens)
        x_trans = x_trans.transpose(1, 2).reshape(B, C, Dp, Hp, Wp)
        x_up = self.upsample(x_trans)
        if x_up.shape[-3:] != (D, H, W):
            x_up = F.interpolate(x_up, size=(D, H, W), mode='trilinear', align_corners=False)
        return self.proj_out(x_up)

class ParallelLatentMapperComplex(nn.Module):

    def __init__(self, channels, out_channels, num_heads=4, num_transformer_layers=4, dropout=0.1):
        super().__init__()
        self.unet_branch = ComplexUnetBranch(channels, out_channels)
        self.transformer_branch = ComplexTransformerBranch(channels, out_channels, num_heads=num_heads, num_layers=num_transformer_layers, dropout=dropout, pool_factor=2)
        self.fusion_conv = nn.Sequential(nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1), nn.BatchNorm3d(out_channels), nn.ReLU(inplace=False))

    def forward(self, x, lat_range, lon_range, t_steps):
        out_unet = self.unet_branch(x)
        out_trans = self.transformer_branch(x, lat_range, lon_range, t_steps)
        fused = out_unet + out_trans
        return self.fusion_conv(fused)

class ComplexEncoder3D(nn.Module):

    def __init__(self, in_ch, base_ch=128, depths=[2, 2], dims=[128, 128]):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1, bias=False), nn.BatchNorm3d(base_ch), nn.ReLU(inplace=False))
        self.layers = nn.ModuleList()
        prev_ch = base_ch
        for i, (num, dim) in enumerate(zip(depths, dims)):
            blocks = []
            for j in range(num):
                stride = 2 if j == 0 and i > 0 else 1
                blocks.append(ResBlock3D(prev_ch, dim, stride))
                prev_ch = dim
            self.layers.append(nn.Sequential(*blocks))

    def forward(self, x):
        x = self.stem(x)
        feats = []
        for l in self.layers:
            x = l(x)
            feats.append(x)
        return feats

class ComplexDecoder3D(nn.Module):

    def __init__(self, dims=[128, 64, 60], depths=[2, 2, 2]):
        super().__init__()
        self.up_convs = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_ch, out_ch = (dims[i], dims[i + 1])
            self.up_convs.append(nn.ConvTranspose3d(in_ch, out_ch, 2, 2))
            self.attn_gates.append(AttentionGate3D(out_ch, out_ch, out_ch // 2))
            blocks = [ResBlock3D(2 * out_ch, out_ch)]
            self.decode_blocks.append(nn.Sequential(*blocks))

    def forward(self, feats):
        x = feats[-1]
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            skip = self.attn_gates[i](x, feats[-2 - i])
            x = self.decode_blocks[i](torch.cat([skip, x], dim=1))
        return x

class ResBlock3D_out(nn.Module):

    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm3d(out_channels))
        self.conv_out = nn.Conv3d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        out = self.conv_out(out)
        return out

class MutiPhyPreNET3D(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ComplexEncoder3D(in_ch=10, base_ch=64, depths=[2, 2], dims=[128, 256])
        self.mapper = ParallelLatentMapperComplex(channels=256, out_channels=256)
        self.decoder_shared = ComplexDecoder3D(dims=[256, 128], depths=[2, 2])
        self.conv_outP = ResBlock3D_out(128, 30)
        self.conv_outA = ResBlock3D_out(128, 90)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.cached_grid = None

    def _encode_and_decode(self, x, lat_range, lon_range, t_steps):
        feats = self.encoder(x)
        latent = self.mapper(feats[-1], lat_range, lon_range, t_steps)
        shared_feat = self.decoder_shared(feats[:-1] + [latent])
        return shared_feat

    def get_Phi(self, x, lat_range, lon_range, t_steps):
        B = x.size(0)
        shared_feat = self._encode_and_decode(x, lat_range, lon_range, t_steps)
        P = self.conv_outP(shared_feat)
        A_raw = self.conv_outA(shared_feat)
        A = A_raw.view(B, 30, 3, *shared_feat.shape[-3:])
        return (P, A)

    def compute_velocity_from_phi_psi(self, phi, psi, spacing=(1.0, 1.0, 1.0)):
        dz, dy, dx = spacing
        B, T, _, D, H, W = psi.shape
        grad_phi_z = torch.gradient(phi, dim=2)[0]
        grad_phi_y = torch.gradient(phi, dim=3)[0]
        grad_phi_x = torch.gradient(phi, dim=4)[0]
        grad_phi = torch.stack([grad_phi_x, grad_phi_y, grad_phi_z], dim=2)
        psi_x, psi_y, psi_z = (psi[:, :, 0], psi[:, :, 1], psi[:, :, 2])
        d_psi_z_dy = torch.gradient(psi_z, dim=3)[0]
        d_psi_y_dz = torch.gradient(psi_y, dim=2)[0]
        curl_x = d_psi_z_dy - d_psi_y_dz
        d_psi_x_dz = torch.gradient(psi_x, dim=2)[0]
        d_psi_z_dx = torch.gradient(psi_z, dim=4)[0]
        curl_y = d_psi_x_dz - d_psi_z_dx
        d_psi_y_dx = torch.gradient(psi_y, dim=4)[0]
        d_psi_x_dy = torch.gradient(psi_x, dim=3)[0]
        curl_z = d_psi_y_dx - d_psi_x_dy
        curl_psi = torch.stack([curl_x, curl_y, curl_z], dim=2)
        velocity = grad_phi + curl_psi
        return velocity

    def make_grid(self, input):
        B, _, D, H, W = input.size()
        if self.cached_grid is not None and self.cached_grid.shape[0] == B and (self.cached_grid.shape[-3:] == (D, H, W)) and (self.cached_grid.device == input.device):
            return self.cached_grid
        z = torch.arange(D, device=input.device).view(1, 1, D, 1, 1).expand(B, 1, D, H, W)
        y = torch.arange(H, device=input.device).view(1, 1, 1, H, 1).expand(B, 1, D, H, W)
        x = torch.arange(W, device=input.device).view(1, 1, 1, 1, W).expand(B, 1, D, H, W)
        grid = torch.cat((x, y, z), 1).float()
        self.cached_grid = grid
        return grid

    def warp(self, input, flow, grid, mode='nearest', padding_mode='border'):
        B, L, D, H, W = input.size()
        vgrid = grid + flow
        vgrid[:, 0] = 2.0 * vgrid[:, 0].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1].clone() / max(H - 1, 1) - 1.0
        vgrid[:, 2] = 2.0 * vgrid[:, 2].clone() / max(D - 1, 1) - 1.0
        vgrid = vgrid.permute(0, 2, 3, 4, 1)
        output = F.grid_sample(input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True)
        return output

    def brownian_diffuse(self, x, D_coef, grid, mode, n_samples=4):
        B, C, D_, H, W = x.shape
        x_rep = x.repeat(n_samples, 1, 1, 1, 1)
        D_rep = D_coef.repeat(n_samples, 1, 1, 1, 1)
        grid_rep = grid.repeat(n_samples, 1, 1, 1, 1)
        noise = torch.randn_like(D_rep)
        delta = D_rep * noise
        new_grid = grid_rep + delta
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
        new_grid[:, 2] = 2.0 * new_grid[:, 2] / (D_ - 1) - 1.0
        new_grid = new_grid.permute(0, 2, 3, 4, 1)
        warped = F.grid_sample(x_rep, new_grid, align_corners=True, padding_mode='border', mode=mode)
        warped = warped.view(n_samples, B, C, D_, H, W).mean(dim=0)
        return warped

    def forward(self, x, mode, x_mask, y_mask, lat_range, lon_range, t_steps):
        P, A = self.get_Phi(x, lat_range, lon_range, t_steps)
        grid = self.make_grid(x[:, 0:1])
        vf = self.compute_velocity_from_phi_psi(P, A)
        outs_v = []
        x_pre = x[:, -1:]
        T_out = vf.shape[1]
        for i in range(T_out):
            x_in = x_pre.detach()
            x_warp = self.warp(x_in, vf[:, i], grid, mode)
            outs_v.append(x_warp)
            x_pre = x_warp
        out_v = torch.cat(outs_v, dim=1)
        return out_v * y_mask

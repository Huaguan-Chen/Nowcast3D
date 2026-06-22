import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate3D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

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
        self.bn1   = nn.BatchNorm3d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ComplexUnetBranch(nn.Module):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            ResBlock3D(channels)
        )
        self.down = nn.MaxPool3d(kernel_size=2, stride=2)
        self.conv2 = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            ResBlock3D(channels)
        )
        self.up = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2)
        self.conv3 = nn.Sequential(
            nn.Conv3d(channels*2, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            ResBlock3D(out_channels)
        )
    def forward(self, x):
        x1 = self.conv1(x)
        x_down = self.down(x1)
        x2 = self.conv2(x_down)
        x_up = self.up(x2)
        x_cat = torch.cat([x1, x_up], dim=1)
        return self.conv3(x_cat)

class FourierEmb(nn.Module):
    def __init__(self, embedding_dim, wavelength_min, wavelength_max):
        super().__init__()
        if embedding_dim % 2 != 0:
            raise ValueError("embedding_dim must be even")
        wavelengths = torch.logspace(
            math.log10(wavelength_min),
            math.log10(wavelength_max),
            steps=embedding_dim // 2,
            dtype=torch.float32,
        )
        self.register_buffer("angular_frequencies", 2.0 * math.pi / wavelengths)

    def forward(self, coordinates):
        coordinates = coordinates.to(
            device=self.angular_frequencies.device,
            dtype=self.angular_frequencies.dtype,
        )
        angles = coordinates.unsqueeze(-1) * self.angular_frequencies
        return torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(-2)

class PhysicalSpatiotemporalEmbedding3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.height_emb = FourierEmb(channels, 500.0, 16000.0)
        self.row_emb = FourierEmb(channels, 1.0, 512.0)
        self.col_emb = FourierEmb(channels, 1.0, 512.0)
        self.time_emb = FourierEmb(channels, 1.0, 10.0 * 24.0 * 30.0)
        self.register_buffer(
            "native_heights_m",
            torch.tensor(
                [
                    500, 1000, 1500, 2000, 2500, 3000, 3500, 4000,
                    4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000,
                    9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000,
                ],
                dtype=torch.float32,
            ),
        )

    def representative_heights(self, embedded_depth):
        if embedded_depth > self.native_heights_m.numel():
            raise ValueError(
                f"embedded depth {embedded_depth} exceeds the 24 native levels"
            )
        bins = torch.tensor_split(self.native_heights_m, embedded_depth)
        return torch.stack([height_bin.mean() for height_bin in bins])

    def forward(
        self,
        batch_size,
        embedded_depth,
        embedded_height,
        embedded_width,
        *,
        absolute_time_steps=None,
        horizontal_scale=1.0,
        device=None,
        dtype=None,
    ):
        device = device or self.native_heights_m.device
        dtype = dtype or self.native_heights_m.dtype
        height_coordinates = self.representative_heights(embedded_depth)
        row_coordinates = (
            torch.arange(embedded_height, device=device, dtype=torch.float32)
            * float(horizontal_scale)
        )
        col_coordinates = (
            torch.arange(embedded_width, device=device, dtype=torch.float32)
            * float(horizontal_scale)
        )
        height = self.height_emb(height_coordinates).transpose(0, 1)
        row = self.row_emb(row_coordinates).transpose(0, 1)
        col = self.col_emb(col_coordinates).transpose(0, 1)
        space = (
            height[:, :, None, None]
            + row[:, None, :, None]
            + col[:, None, None, :]
        )
        if absolute_time_steps is None:
            absolute_time_steps = torch.zeros(batch_size, device=device)
        else:
            absolute_time_steps = torch.as_tensor(
                absolute_time_steps, device=device, dtype=torch.float32
            )
            if absolute_time_steps.ndim == 0:
                absolute_time_steps = absolute_time_steps.repeat(batch_size)
            if absolute_time_steps.shape != (batch_size,):
                raise ValueError(
                    "absolute_time_steps must be a scalar or have shape (batch_size,)"
                )
        time = self.time_emb(absolute_time_steps)[:, :, None, None, None]
        return (space[None] + time).to(device=device, dtype=dtype)


class ComplexTransformerBranch(nn.Module):
    def __init__(self, channels, out_channels, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Conv3d(channels, channels, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Conv3d(channels, out_channels, kernel_size=1)
        self.physical_embedding = PhysicalSpatiotemporalEmbedding3D(channels)

    def forward(self, x, absolute_time_steps=None, horizontal_scale=1.0):
        B, C, D, H, W = x.shape
        x_proj = self.proj_in(x)
        x_proj = x_proj + self.physical_embedding(
            B,
            D,
            H,
            W,
            absolute_time_steps=absolute_time_steps,
            horizontal_scale=horizontal_scale,
            device=x.device,
            dtype=x.dtype,
        )
        x_flat = x_proj.reshape(B, C, -1).transpose(1, 2)
        x_trans = self.transformer(x_flat)
        x_trans = x_trans.transpose(1, 2).reshape(B, C, D, H, W)
        return self.proj_out(x_trans)


class ParallelLatentMapperComplex(nn.Module):
    def __init__(self, channels, out_channels, num_heads=4, num_transformer_layers=4, dropout=0.1):
        super().__init__()
        self.unet_branch = ComplexUnetBranch(channels, out_channels)
        self.transformer_branch = ComplexTransformerBranch(channels, out_channels, num_heads, num_transformer_layers, dropout)
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x, absolute_time_steps=None, horizontal_scale=1.0):
        out_unet = self.unet_branch(x)
        out_trans = self.transformer_branch(
            x,
            absolute_time_steps=absolute_time_steps,
            horizontal_scale=horizontal_scale,
        )
        fused = out_unet + out_trans
        return self.fusion_conv(fused)


class ComplexEncoder3D(nn.Module):
    def __init__(self, in_ch, base_ch=64, depths=[2,2], dims=[64,128]):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(in_ch, base_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(base_ch),
            nn.ReLU(inplace=True)
        )
        self.layers = nn.ModuleList()
        prev_ch = base_ch
        for i,(num,dim) in enumerate(zip(depths,dims)):
            blocks=[]
            for j in range(num):
                stride=2 if j==0 and i>0 else 1
                blocks.append(ResBlock3D(prev_ch, dim, stride))
                prev_ch=dim
            self.layers.append(nn.Sequential(*blocks))
    def forward(self,x):
        x=self.stem(x)
        feats=[]
        for l in self.layers:
            x=l(x); feats.append(x)
        return feats

class ComplexDecoder3D(nn.Module):
    def __init__(self, dims=[128,64,60], depths=[2,2,2]):
        super().__init__()
        self.up_convs=nn.ModuleList(); self.attn_gates=nn.ModuleList(); self.decode_blocks=nn.ModuleList()
        for i in range(len(dims)-1):
            in_ch, out_ch = dims[i], dims[i+1]
            self.up_convs.append(nn.ConvTranspose3d(in_ch,out_ch,2,2))
            self.attn_gates.append(AttentionGate3D(out_ch,out_ch,out_ch//2))
            blocks=[ResBlock3D(2*out_ch,out_ch)]
            self.decode_blocks.append(nn.Sequential(*blocks))
    def forward(self,feats):
        x=feats[-1]
        for i in range(len(self.up_convs)):
            x=self.up_convs[i](x)
            skip=self.attn_gates[i](x, feats[-2-i])
            x=self.decode_blocks[i](torch.cat([skip,x],1))
        return x

class BrownResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels=None, stride=1):
        super().__init__()
        out_channels = out_channels or in_channels
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        self.out_conv = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        out = self.out_conv(out)
        return out

class BrownComplexDecoder3D(nn.Module):
    def __init__(self, dims=[128,64,60], depths=[2,2,2]):
        super().__init__()
        self.up_convs = nn.ModuleList()
        self.attn_gates = nn.ModuleList()
        self.decode_blocks = nn.ModuleList()
        for i in range(len(dims) - 1):
            in_ch, out_ch = dims[i], dims[i + 1]
            self.up_convs.append(nn.ConvTranspose3d(in_ch, out_ch, 2, 2))
            self.attn_gates.append(AttentionGate3D(out_ch, out_ch, out_ch // 2))
            self.decode_blocks.append(nn.Sequential(BrownResBlock3D(2 * out_ch, out_ch)))

    def forward(self, feats):
        x = feats[-1]
        for i in range(len(self.up_convs)):
            x = self.up_convs[i](x)
            skip = self.attn_gates[i](x, feats[-2 - i])
            x = self.decode_blocks[i](torch.cat([skip, x], 1))
        return x


class MutiPhyPreNET3D(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder   = ComplexEncoder3D(in_ch=10,base_ch=128,depths=[2,2],dims=[128,256])
        self.mapper    = ParallelLatentMapperComplex(channels=256, out_channels = 256)


        self.decoder_P = ComplexDecoder3D(dims=[256,128],depths=[2,2])
        self.conv_outP = ResBlock3D(128,30)

        self.decoder_A = ComplexDecoder3D(dims=[256,128],depths=[2,2])
        self.conv_outA = ResBlock3D(128,90)

        self.decoder_S = ComplexDecoder3D(dims=[256,128],depths=[2,2])
        self.conv_outS = ResBlock3D(128,30)

        self.decoder_sigma = BrownComplexDecoder3D(dims=[256,128],depths=[2,2])
        self.conv_out_sigma = BrownResBlock3D(128,90)

    def get_Phi(self, x, absolute_time_steps=None, horizontal_scale=1.0):
        B = x.size(0)
        feats = self.encoder(x)
        latent=feats[-1]
        latent=self.mapper(
            latent,
            absolute_time_steps=absolute_time_steps,
            horizontal_scale=horizontal_scale,
        )

        p_feat=self.decoder_P(feats[:-1]+[latent])
        P = self.conv_outP(p_feat)

        a_feat=self.decoder_A(feats[:-1]+[latent])
        A = self.conv_outA(a_feat).reshape(B,30,3,*a_feat.shape[-3:])

        s_feat=self.decoder_S(feats[:-1]+[latent])
        S = self.conv_outS(s_feat)

        sigma_feat = self.decoder_sigma(feats[:-1]+[latent])
        sigma = self.conv_out_sigma(sigma_feat).reshape(B,30,3,*sigma_feat.shape[-3:])

        return P, A, S, sigma


    def compute_velocity_from_phi_psi(self, phi, psi, spacing=(1.0, 1.0, 1.0)):
        dz, dy, dx = spacing
        B, T, _, D, H, W = psi.shape


        grad_phi_z = torch.gradient(phi, dim=2)[0]
        grad_phi_y = torch.gradient(phi, dim=3)[0]
        grad_phi_x = torch.gradient(phi, dim=4)[0]

        grad_phi = torch.stack([grad_phi_x, grad_phi_y, grad_phi_z], dim=2)


        psi_x, psi_y, psi_z = psi[:, :, 0], psi[:, :, 1], psi[:, :, 2]


        d_psi_z_dy = torch.gradient(psi_z,  dim=3)[0]
        d_psi_y_dz = torch.gradient(psi_y,  dim=2)[0]
        curl_x = d_psi_z_dy - d_psi_y_dz

        d_psi_x_dz = torch.gradient(psi_x,  dim=2)[0]
        d_psi_z_dx = torch.gradient(psi_z,  dim=4)[0]
        curl_y = d_psi_x_dz - d_psi_z_dx

        d_psi_y_dx = torch.gradient(psi_y,  dim=4)[0]
        d_psi_x_dy = torch.gradient(psi_x,  dim=3)[0]
        curl_z = d_psi_y_dx - d_psi_x_dy

        curl_psi = torch.stack([curl_x, curl_y, curl_z], dim=2)


        velocity = grad_phi + curl_psi


        return velocity

    def make_grid(self,input):
        B,_,D,H,W=input.size()
        z=torch.arange(D,device=input.device).view(1,1,D,1,1).expand(B,1,D,H,W)
        y=torch.arange(H,device=input.device).view(1,1,1,H,1).expand(B,1,D,H,W)
        x=torch.arange(W,device=input.device).view(1,1,1,1,W).expand(B,1,D,H,W)
        return torch.cat((x,y,z),1).float()

    def warp(self, input, flow, grid, mode="nearest", padding_mode="border"):
        B, L, D, H, W = input.size()

        vgrid = grid + flow


        vgrid[:, 0] = 2.0 * vgrid[:, 0].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1] = 2.0 * vgrid[:, 1].clone() / max(H - 1, 1) - 1.0
        vgrid[:, 2] = 2.0 * vgrid[:, 2].clone() / max(D - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 4, 1)

        output = torch.nn.functional.grid_sample(
            input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True
        )

        return output

    def brownian_diffuse(self, x, v, D, grid, mode, n_samples=8):
        B, C, D_, H, W = x.shape
        x_rep = x.repeat(n_samples, 1, 1, 1, 1)
        v_rep = v.repeat(n_samples, 1, 1, 1, 1)
        D_rep = D.repeat(n_samples, 1, 1, 1, 1)
        grid_rep = grid.repeat(n_samples, 1, 1, 1, 1)

        noise = torch.randn_like(v_rep)
        new_grid = grid_rep + v_rep + D_rep * noise
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
        new_grid[:, 2] = 2.0 * new_grid[:, 2] / (D_ - 1) - 1.0
        new_grid = new_grid.permute(0, 2, 3, 4, 1)

        warped = F.grid_sample(
            x_rep,
            new_grid,
            align_corners=True,
            padding_mode="border",
            mode=mode,
        ).view(n_samples, B, C, D_, H, W)
        reflectivity_factor = 10.0 ** ((warped * 80.0) / 10.0)
        mean_factor = reflectivity_factor.mean(dim=0)
        return 10.0 * torch.log10(torch.clamp(mean_factor, min=1e-12)) / 80.0

    def forward(
        self,
        x,
        mode,
        x_mask,
        y_mask,
        absolute_time_steps=None,
        horizontal_scale=1.0,
    ):
        P, A, S, sigma = self.get_Phi(
            x,
            absolute_time_steps=absolute_time_steps,
            horizontal_scale=horizontal_scale,
        )
        grid = self.make_grid(x[:, 0:1])
        velocity = self.compute_velocity_from_phi_psi(P, A).float()
        forecasts = []
        x_pre = x[:, -1:]
        for index in range(velocity.shape[1]):
            x_pre = self.brownian_diffuse(
                x_pre.detach(),
                velocity[:, index],
                sigma[:, index],
                grid,
                mode,
            )
            x_pre = x_pre + S[:, index:index + 1]
            forecasts.append(x_pre)
        return torch.cat(forecasts, dim=1) * y_mask

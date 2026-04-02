import torch
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class AttentionGate3D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        """
        F_g: features from gating signal (decoder)
        F_l: features from encoder (skip connection)
        F_int: intermediate feature channels
        """
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
        """
        g: gating signal from decoder
        x: skip connection from encoder
        """
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



class ComplexTransformerBranch(nn.Module):
    def __init__(self, channels, out_channels, num_heads=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.proj_in = nn.Conv3d(channels, channels, kernel_size=1)
        encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=num_heads, dropout=dropout, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj_out = nn.Conv3d(channels, out_channels, kernel_size=1)
        self.pos_embed = nn.Parameter(torch.zeros(1, 12*128*128, channels))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_proj = self.proj_in(x)
        x_flat = x_proj.reshape(B, C, -1).transpose(1, 2)
        x_flat = x_flat + self.pos_embed.to(x_flat.device)
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
    def forward(self, x):
        out_unet = self.unet_branch(x)
        out_trans = self.transformer_branch(x)
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



class MutiPhyPreNET3D(nn.Module):
    def __init__(self):
        super().__init__()
        # encoder outputs 256 channels at 18×128×128
        self.encoder   = ComplexEncoder3D(in_ch=10,base_ch=128,depths=[2,2],dims=[128,256])
        self.mapper    = ParallelLatentMapperComplex(channels=256, out_channels = 256)
        # decoder dims: [60 -> 64 -> (upsample steps...) -> original]

        self.decoder_P = ComplexDecoder3D(dims=[256,128],depths=[2,2])
        self.conv_outP = ResBlock3D(128,30)

        self.decoder_A = ComplexDecoder3D(dims=[256,128],depths=[2,2])
        self.conv_outA = ResBlock3D(128,90)

        self.decoder_S = ComplexDecoder3D(dims=[256,128],depths=[2,2])
        self.conv_outS = ResBlock3D(128,30)

        self.conv_outK = ResBlock3D(128,90)



    def get_Phi(self,x):
        B = x.size(0)
        feats = self.encoder(x)
        latent=feats[-1]
        latent=self.mapper(latent)

        p_feat=self.decoder_P(feats[:-1]+[latent])
        P = self.conv_outP(p_feat)

        a_feat=self.decoder_A(feats[:-1]+[latent])
        A = self.conv_outA(a_feat).reshape(B,30,3,*a_feat.shape[-3:])

        s_feat=self.decoder_S(feats[:-1]+[latent])
        S = self.conv_outS(s_feat)

        K = self.conv_outK(s_feat)

        return P, A, S,K
    


    def compute_velocity_from_phi_psi(self, phi, psi, spacing=(1.0, 1.0, 1.0)):
        dz, dy, dx = spacing
        B, T, _, D, H, W = psi.shape

        # compute grad(phi): shape = (B, T, 3, D, H, W)
        grad_phi_z = torch.gradient(phi, dim=2)[0]
        grad_phi_y = torch.gradient(phi, dim=3)[0]
        grad_phi_x = torch.gradient(phi, dim=4)[0]

        grad_phi = torch.stack([grad_phi_x, grad_phi_y, grad_phi_z], dim=2)

        # split psi into components
        psi_x, psi_y, psi_z = psi[:, :, 0], psi[:, :, 1], psi[:, :, 2]

        # compute curl(psi)
        d_psi_z_dy = torch.gradient(psi_z,  dim=3)[0]
        d_psi_y_dz = torch.gradient(psi_y,  dim=2)[0]
        curl_x = d_psi_z_dy - d_psi_y_dz

        d_psi_x_dz = torch.gradient(psi_x,  dim=2)[0]
        d_psi_z_dx = torch.gradient(psi_z,  dim=4)[0]
        curl_y = d_psi_x_dz - d_psi_z_dx

        d_psi_y_dx = torch.gradient(psi_y,  dim=4)[0]
        d_psi_x_dy = torch.gradient(psi_x,  dim=3)[0]
        curl_z = d_psi_y_dx - d_psi_x_dy

        curl_psi = torch.stack([curl_x, curl_y, curl_z], dim=2)  # shape (B, T, 3, D, H, W)


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

        vgrid = grid + flow  # B, 3, D, H, W


        vgrid[:, 0] = 2.0 * vgrid[:, 0].clone() / max(W - 1, 1) - 1.0  # x
        vgrid[:, 1] = 2.0 * vgrid[:, 1].clone() / max(H - 1, 1) - 1.0  # y
        vgrid[:, 2] = 2.0 * vgrid[:, 2].clone() / max(D - 1, 1) - 1.0  # z

        vgrid = vgrid.permute(0, 2, 3, 4, 1)  # B, D, H, W, 3

        output = torch.nn.functional.grid_sample(
            input, vgrid, padding_mode=padding_mode, mode=mode, align_corners=True
        )

        return output

    def brownian_diffuse(self,x, v, D, grid, mode,n_samples=8):
        B, C, D_, H, W = x.shape
        device = x.device

        x_rep     = x.repeat(n_samples, 1, 1, 1, 1)       # (B*n,1,D,H,W)
        v_rep     = v.repeat(n_samples, 1, 1, 1, 1)       # (B*n,3,D,H,W)
        D_rep     = D.repeat(n_samples, 1, 1, 1, 1)       # (B*n,3,D,H,W)
        grid_rep  = grid.repeat(n_samples, 1, 1, 1, 1)    # (B*n,3,D,H,W)

        noise = torch.randn_like(v_rep)  # (B*n,3,D,H,W)
        delta = v_rep + D_rep * noise    # (B*n,3,D,H,W)

        new_grid = grid_rep + delta

        # normalize to [-1,1]
        new_grid[:, 0] = 2.0 * new_grid[:, 0] / (W - 1) - 1.0
        new_grid[:, 1] = 2.0 * new_grid[:, 1] / (H - 1) - 1.0
        new_grid[:, 2] = 2.0 * new_grid[:, 2] / (D_ - 1) - 1.0
        new_grid = new_grid.permute(0, 2, 3, 4, 1)  # (B*n,D,H,W,3)

        warped = F.grid_sample(x_rep, new_grid, align_corners=True, padding_mode="border", mode = mode)  # (B*n,1,D,H,W)

        warped = warped.view(n_samples, B, C, D_, H, W).mean(dim=0)  # (B,1,D,H,W)
        # warped = warped.view(n_samples, B, C, D_, H, W).max(dim=0)[0]  # (B,1,D,H,W)

        # warped_norm = self.energy_norm(x,warped)

        return warped


    def forward(self,x,mode,x_mask,y_mask):

        P,A,S,K = self.get_Phi(x)

       
        grid=self.make_grid(x[:,0:1])

        vf = self.compute_velocity_from_phi_psi(P,A) 

        outs=[]

        x_pre=x[:,-1:]

        for i in range(vf.shape[1]):
            x_pre=self.brownian_diffuse(x_pre,vf[:,i],K[:,i],grid,mode)
            x_pre=x_pre+S[:,i:i+1]
            outs.append(x_pre)

        return torch.cat(outs,1) * y_mask
    




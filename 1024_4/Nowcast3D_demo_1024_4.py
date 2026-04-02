from glob import glob
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import imageio
from scipy import ndimage

from Diff_Structure import get_model as get_model_S
from Diff_Intensity import get_model as get_model_I
from PhyPredNet import MutiPhyPreNET3D

root_dir = ""
folder_name = "20250728_120000_fill"
folder = os.path.join(root_dir, folder_name)

save_dir = "vis_results"
os.makedirs(save_dir, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


model = MutiPhyPreNET3D().to(device)
model.load_state_dict(torch.load("PhyPredNet_1024_4.pth", map_location=device))

diffusion_S = get_model_S().to(device)
diffusion_S.load_state_dict(torch.load("Diff_Structure_1024_4.pth", map_location=device))
diffusion_S.eval()

diffusion_I = get_model_I().to(device)
diffusion_I.load_state_dict(torch.load("Diff_Intensity_1024_4.pth", map_location=device))
diffusion_I.eval()


def remove_small_connected_regions(tensor, min_size=4):
    B, T, H, W = tensor.shape
    out = torch.zeros_like(tensor)
    arr = tensor.detach().cpu().numpy()

    for b in range(B):
        for t in range(T):
            layer = arr[b, t]
            mask = layer > 0
            labeled, num = ndimage.label(mask)
            sizes = ndimage.sum(mask, labeled, range(num + 1))
            keep = sizes >= min_size
            out[b, t] = torch.from_numpy(layer * keep[labeled])

    return out.to(tensor.device)


def save_video_side_by_side(pred_tensor, gt_tensor, save_path, fps=5):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as patches

    pred = pred_tensor[0].detach().cpu().numpy() * 80
    gt = gt_tensor[0].detach().cpu().numpy() * 80

    levels = [0, 5, 15, 25, 35, 45, 50, 55, 60, 65, 70, 80]
    colors = [
        [1, 1, 1, 0],
        [0, 0.25, 0.6, 0.95],
        [0, 0.5, 0.5, 0.95],
        [0, 0.5, 0.25, 0.95],
        [0.7, 0.7, 0, 0.95],
        [0.8, 0.5, 0, 0.95],
        [0.8, 0.3, 0, 0.95],
        [0.7, 0, 0, 0.95],
        [0.6, 0, 0.3, 0.95],
        [0.5, 0, 0.5, 0.95],
        [0.35, 0, 0.6, 0.95],
    ]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)

    with imageio.get_writer(save_path, fps=fps, codec="libx264") as writer:
        for t in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            for i, data in enumerate([gt[t], pred[t]]):
                ax[i].imshow(data, cmap=cmap, norm=norm)
                ax[i].axis("off")
                rect = patches.Rectangle(
                    (0, 0), data.shape[1] - 1, data.shape[0] - 1,
                    linewidth=2, edgecolor="black", facecolor="none"
                )
                ax[i].add_patch(rect)

            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            writer.append_data(image[:, :, :3])
            plt.close(fig)


def load_frame(path):
    arr = np.load(path)

    if arr.shape == (24, 256, 256):
        return arr
    if arr.shape == (6144, 256):
        return arr.reshape(24, 256, 256)

    raise ValueError(f"Unexpected shape for {path}: {arr.shape}")


def load_one_folder(folder):
    data_files = sorted(glob(os.path.join(folder, "*_data.npy")))
    mask_files = sorted(glob(os.path.join(folder, "*_mask.npy")))

    x_data = np.stack([load_frame(f) for f in data_files[:10]], axis=0) / 800.0
    x_mask = np.stack([load_frame(f) for f in mask_files[:10]], axis=0)
    y_data = np.stack([load_frame(f) for f in data_files[10:40]], axis=0) / 800.0

    x_data = torch.from_numpy(x_data).float().unsqueeze(0)
    x_mask = torch.from_numpy(x_mask).float().unsqueeze(0)
    y_data = torch.from_numpy(y_data).float().unsqueeze(0)
    y_mask = torch.max(x_mask, dim=1, keepdim=True)[0].repeat(1, 30, 1, 1, 1)

    print("x_data:", x_data.shape)
    print("x_mask:", x_mask.shape)
    print("y_data:", y_data.shape)
    print("y_mask:", y_mask.shape)

    return x_data, x_mask, y_data, y_mask






with torch.no_grad():
    x_data, x_mask, y_data, y_mask = load_one_folder(folder)
    x_data, x_mask = x_data.to(device), x_mask.to(device)
    y_data, y_mask = y_data.to(device), y_mask.to(device)

    x_pre = model(x_data, "nearest", x_mask, y_mask)

    x_zmax = torch.max(x_data, dim=2)[0]
    x_pre_zmax = torch.max(x_pre, dim=2)[0]
    y_zmax = torch.max(y_data, dim=2)[0]
    y_mask_zmax = torch.max(y_mask, dim=2)[0]

    x_zmax_norm = diffusion_S.normalize(x_zmax)
    x_pre_zmax_norm = diffusion_S.normalize(x_pre_zmax)

    pred_y_0 = diffusion_I.sample(x_zmax_norm, x_pre_zmax_norm)
    pred_y_0 = torch.where(pred_y_0 < 0.1, torch.tensor(0.0, device=device), pred_y_0) * y_mask_zmax
    pred_y_0 = remove_small_connected_regions(pred_y_0, min_size=4)

    pred_y_new = diffusion_S.sample(x_zmax_norm, x_pre_zmax_norm)
    pred_y_new = torch.where(pred_y_new < 0.1, torch.tensor(0.0, device=device), pred_y_new) * y_mask_zmax
    pred_y_new = remove_small_connected_regions(pred_y_new, min_size=4)

    pred_y_new_dieta = pred_y_new + 0.5 * (x_pre_zmax - pred_y_0)

    Z_0 = 10.0 ** ((pred_y_0 * 80.0) / 10.0)
    Z_1 = 10.0 ** ((pred_y_new * 80.0) / 10.0)
    Z_2 = 10.0 ** ((pred_y_new_dieta * 80.0) / 10.0)

    pred_y_ens = (10.0 * torch.log10(torch.clamp((Z_0 + Z_1 + Z_2) / 3.0, min=1e-12))) / 80.0
    pred_y = (pred_y_ens + x_pre_zmax) / 2.0
    pred_y = remove_small_connected_regions(pred_y, min_size=4) 

    save_video_side_by_side(
        pred_y,
        y_zmax,
        os.path.join(save_dir, f"{folder_name}.mp4"),
        fps=5
    )

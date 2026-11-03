from datetime import datetime
from glob import glob
from pathlib import Path
import os

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
from scipy import ndimage
import torch

from Diff_Intensity import get_model as get_model_I
from Diff_Structure import get_model as get_model_S
from PhyPredNet import MutiPhyPreNET3D


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / "ckpt-1024-4"
FOLDER_NAME = "20250728_120000_fill"
FOLDER = HERE / FOLDER_NAME
GT_FOLDER = HERE / "20250728_120000"
SAVE_DIR = HERE / "vis_results"
REFERENCE_TIME = datetime(2024, 1, 1)
HORIZONTAL_SCALE = 4.0
INTERPOLATION = "nearest"


def load_frame(path):
    arr = np.load(path)
    if arr.shape == (24, 256, 256):
        return arr
    if arr.shape == (6144, 256):
        return arr.reshape(24, 256, 256)
    raise ValueError(f"Unexpected shape for {path}: {arr.shape}")


def load_one_folder(folder, gt_folder):
    data_files = sorted(glob(str(folder / "*_data.npy")))
    mask_files = sorted(glob(str(folder / "*_mask.npy")))
    gt_data_files = sorted(glob(str(gt_folder / "*_data.npy")))
    if len(data_files) < 10 or len(mask_files) < 10:
        raise ValueError(f"{folder} requires 10 data files and 10 mask files")
    if len(gt_data_files) < 40:
        raise ValueError(f"{gt_folder} requires 40 raw data files")

    x_files = data_files[:10]
    x_mask_files = mask_files[:10]
    y_files = gt_data_files[10:40]

    x_data = np.stack([load_frame(f) for f in x_files], axis=0) / 800.0
    x_mask = np.stack([load_frame(f) for f in x_mask_files], axis=0)
    y_data = np.stack([load_frame(f) for f in y_files], axis=0) / 800.0

    x_data = torch.from_numpy(x_data).float().unsqueeze(0)
    x_mask = torch.from_numpy(x_mask).float().unsqueeze(0)
    y_data = torch.from_numpy(y_data).float().unsqueeze(0)
    y_mask = torch.max(x_mask, dim=1, keepdim=True)[0].repeat(1, 30, 1, 1, 1)

    last_time = datetime.strptime(
        Path(x_files[-1]).name.split("_data.npy")[0],
        "%Y%m%d_%H%M%S",
    )
    absolute_time = int((last_time - REFERENCE_TIME).total_seconds() // 360)

    print("x_data:", x_data.shape)
    print("x_mask:", x_mask.shape)
    print("y_data:", y_data.shape)
    print("y_mask:", y_mask.shape)

    return x_data, x_mask, y_data, y_mask, absolute_time


def load_state(path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
        state = state["model"]
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    return {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in state.items()
    }


def physical_reflectivity_mean(*members):
    stack = torch.stack(members, dim=0)
    stack = torch.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=-10.0).clamp(max=1.0)
    factor = 10.0 ** ((stack * 80.0) / 10.0)
    return 10.0 * torch.log10(torch.clamp(factor.mean(dim=0), min=1e-12)) / 80.0


def remove_small_connected_regions(tensor, min_size=4):
    b, t, h, w = tensor.shape
    out = torch.zeros_like(tensor)
    arr = tensor.detach().cpu().numpy()
    for bi in range(b):
        for ti in range(t):
            layer = arr[bi, ti]
            mask = layer > 0
            labeled, num = ndimage.label(mask)
            sizes = ndimage.sum(mask, labeled, range(num + 1))
            keep = sizes >= min_size
            out[bi, ti] = torch.from_numpy(layer * keep[labeled])
    return out.to(tensor.device)


def save_video_side_by_side(pred_tensor, gt_tensor, save_path, fps=5):
    pred = pred_tensor[0].detach().cpu().numpy() * 80.0
    gt = gt_tensor[0].detach().cpu().numpy() * 80.0

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
        for time_index in range(pred.shape[0]):
            fig, ax = plt.subplots(1, 2, figsize=(8, 4), constrained_layout=True)
            for panel, data in enumerate([gt[time_index], pred[time_index]]):
                ax[panel].imshow(data, cmap=cmap, norm=norm)
                ax[panel].axis("off")
            fig.canvas.draw()
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            writer.append_data(image[:, :, :3])
            plt.close(fig)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.mha.set_fastpath_enabled(False)

model = MutiPhyPreNET3D().to(device)
model.load_state_dict(load_state(CKPT_DIR / "PhyPredNet_1024_4.pth", device), strict=True)

diffusion_I = get_model_I().to(device)
diffusion_I.load_state_dict(load_state(CKPT_DIR / "Diff_Intensity_1024_4.pth", device), strict=True)

diffusion_S = get_model_S().to(device)
diffusion_S.load_state_dict(load_state(CKPT_DIR / "Diff_Structure_1024_4.pth", device), strict=True)

with torch.no_grad():
    x_data, x_mask, y_data, y_mask, absolute_time = load_one_folder(FOLDER, GT_FOLDER)
    x_data = x_data.to(device)
    x_mask = x_mask.to(device)
    y_data = y_data.to(device)
    y_mask = y_mask.to(device)

    x_pre = model(
        x_data,
        INTERPOLATION,
        x_mask,
        y_mask,
        absolute_time_steps=torch.tensor([absolute_time], device=device),
        horizontal_scale=HORIZONTAL_SCALE,
    )

    x_zmax = torch.max(x_data, dim=2)[0]
    x_pre_zmax = torch.max(x_pre, dim=2)[0]
    y_zmax = torch.max(y_data, dim=2)[0]
    y_mask_zmax = torch.max(y_mask, dim=2)[0]

    x_zmax_norm = diffusion_S.normalize(x_zmax)
    x_pre_zmax_norm = diffusion_S.normalize(x_pre_zmax)

    pred_y_0 = diffusion_I.sample(
        x_zmax_norm,
        x_pre_zmax_norm,
        physical_mean_with=x_pre_zmax,
        mask=y_mask_zmax,
    )
    pred_y_0 = remove_small_connected_regions(pred_y_0, min_size=4)

    pred_y_new = diffusion_S.sample(
        x_zmax_norm,
        x_pre_zmax_norm,
        physical_mean_with=x_pre_zmax,
        mask=y_mask_zmax,
    )
    pred_y_new = remove_small_connected_regions(pred_y_new, min_size=4)

    pred_y_new_dieta = pred_y_new + 0.5 * (x_pre_zmax - pred_y_0)

    pred_y_ens = physical_reflectivity_mean(pred_y_0, pred_y_new, pred_y_new_dieta)
    pred_y = remove_small_connected_regions(pred_y_ens * y_mask_zmax, min_size=4)

SAVE_DIR.mkdir(parents=True, exist_ok=True)
save_path = SAVE_DIR / f"{FOLDER_NAME}.mp4"
save_video_side_by_side(pred_y, y_zmax, save_path, fps=5)

print(f"device: {device}")
print(f"saved: {save_path}")

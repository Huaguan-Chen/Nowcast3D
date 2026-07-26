from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
from Diff_Intensity import get_model as get_model_I
from Diff_Structure import get_model as get_model_S
from PhyPredNet import MutiPhyPreNET3D
HERE = Path(__file__).resolve().parent
CKPT_DIR = HERE / 'ckpt-512-1'
DEFAULT_INPUT_DIR = HERE / '2025062905_fill'
DEFAULT_OUTPUT_DIR = HERE / 'npz_results'
LAT_RANGE = (37.34, 42.46)
LON_RANGE = (113.96, 119.08)
T_IN = 10
T_OUT = 30
DIFFUSION_THRESHOLD = 0.1
MIN_COMPONENT_SIZE = 4

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--case-name')
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()

def load_state(path: Path) -> dict[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        state = torch.load(path, map_location='cpu')
    if isinstance(state, dict) and isinstance(state.get('model'), dict):
        state = state['model']
    if isinstance(state, dict) and isinstance(state.get('state_dict'), dict):
        state = state['state_dict']
    if not isinstance(state, dict):
        raise TypeError(f'Checkpoint is not a state dictionary: {path}')
    return {key[7:] if key.startswith('module.') else key: value for key, value in state.items()}

def raw_data_files(folder: Path) -> list[Path]:
    return sorted((path for path in folder.rglob('*_data.npy') if not path.name.endswith('_data2D.npy') and (not path.name.endswith('_data_pre2D.npy'))))

def load_volume(path: Path) -> np.ndarray:
    value = np.load(path)
    if value.ndim == 3 and value.shape[:2] == (24, 512):
        if value.shape[2] < 512:
            raise ValueError(f'Unexpected input shape: {path}: {value.shape}')
        return value[..., :512].astype(np.float32)
    if value.ndim == 2 and value.shape[0] == 24 * 512:
        if value.shape[1] < 512:
            raise ValueError(f'Unexpected input shape: {path}: {value.shape}')
        return value[:, :512].reshape(24, 512, 512).astype(np.float32)
    raise ValueError(f'Unexpected input shape: {path}: {value.shape}')

def load_case(input_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data_files = raw_data_files(input_dir)
    mask_files = sorted(input_dir.rglob('*_mask.npy'))
    if len(data_files) != T_IN or len(mask_files) != T_IN:
        raise ValueError(f'{input_dir} requires exactly {T_IN} filled data/mask pairs; found {len(data_files)} data and {len(mask_files)} masks')
    data_stamps = np.asarray([path.name.removesuffix('_data.npy') for path in data_files])
    mask_stamps = np.asarray([path.name.removesuffix('_mask.npy') for path in mask_files])
    if not np.array_equal(data_stamps, mask_stamps):
        raise ValueError('Filled data and mask timestamps do not match')
    x_values = [load_volume(path) / np.float32(800.0) for path in data_files]
    x_masks = [load_volume(path) for path in mask_files]
    x = torch.from_numpy(np.stack(x_values)).unsqueeze(0)
    x_mask = torch.from_numpy(np.stack(x_masks)).unsqueeze(0)
    return (x, x_mask)

def physical_reflectivity_mean(*members: torch.Tensor) -> torch.Tensor:
    stack = torch.stack(members, dim=0)
    stack = torch.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=-10.0).clamp(max=1.0)
    linear_z = 10.0 ** (stack * 80.0 / 10.0)
    return 10.0 * torch.log10(torch.clamp(linear_z.mean(dim=0), min=1e-12)) / 80.0

def remove_small_connected_regions(tensor: torch.Tensor, min_size: int=MIN_COMPONENT_SIZE) -> torch.Tensor:
    values = tensor.detach().float().cpu().numpy()
    output = np.zeros_like(values)
    for batch_index in range(values.shape[0]):
        for time_index in range(values.shape[1]):
            frame = values[batch_index, time_index]
            labels, count = ndimage.label(frame > 0)
            sizes = ndimage.sum(frame > 0, labels, index=np.arange(1, count + 1))
            keep = np.zeros(count + 1, dtype=bool)
            keep[1:] = sizes >= min_size
            output[batch_index, time_index] = frame * keep[labels]
    return torch.from_numpy(output).to(device=tensor.device, dtype=tensor.dtype)

def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor[0].detach().float().cpu().numpy()

def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise FileNotFoundError(args.input_dir)
    for filename in ('PhyPredNet_512_1.pth', 'Diff_Intensity_512_1.pth', 'Diff_Structure_512_1.pth'):
        if not (CKPT_DIR / filename).is_file():
            raise FileNotFoundError(CKPT_DIR / filename)
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError(f'CUDA is required; requested={args.device}, cuda_available={torch.cuda.is_available()}')
    torch.backends.mha.set_fastpath_enabled(False)
    case_name = args.case_name or args.input_dir.name.removesuffix('_fill')
    x_cpu, x_mask_cpu = load_case(args.input_dir)
    x = x_cpu.to(device)
    x_mask = x_mask_cpu.to(device)
    y_mask = x_mask.float().mean(dim=1, keepdim=True).repeat(1, T_OUT, 1, 1, 1)
    y_mask_zmax = y_mask.max(dim=2).values
    model = MutiPhyPreNET3D().to(device)
    model.load_state_dict(load_state(CKPT_DIR / 'PhyPredNet_512_1.pth'), strict=True)
    with torch.no_grad():
        forecast_3d = model(x, mode='nearest', x_mask=torch.ones_like(x_mask), y_mask=y_mask, lat_range=LAT_RANGE, lon_range=LON_RANGE, t_steps=torch.tensor([0.0], device=device))
        input_zmax = x.max(dim=2).values
        physical = forecast_3d.max(dim=2).values
    diffusion_I = get_model_I().to(device)
    diffusion_I.load_state_dict(load_state(CKPT_DIR / 'Diff_Intensity_512_1.pth'), strict=True)
    diffusion_I.eval()
    diffusion_S = get_model_S().to(device)
    diffusion_S.load_state_dict(load_state(CKPT_DIR / 'Diff_Structure_512_1.pth'), strict=True)
    diffusion_S.eval()
    with torch.no_grad():
        input_norm = diffusion_S.normalize(input_zmax)
        physical_norm = diffusion_S.normalize(physical)
        sample_I = diffusion_I.sample(input_norm, physical_norm)
        sample_I = torch.where(sample_I < DIFFUSION_THRESHOLD, torch.zeros_like(sample_I), sample_I) * y_mask_zmax
        sample_S = diffusion_S.sample(input_norm, physical_norm)
        sample_S = torch.where(sample_S < DIFFUSION_THRESHOLD, torch.zeros_like(sample_S), sample_S) * y_mask_zmax
        member_I = remove_small_connected_regions(physical_reflectivity_mean(sample_I, physical) * y_mask_zmax)
        member_S = remove_small_connected_regions(physical_reflectivity_mean(sample_S, physical) * y_mask_zmax)
        member_R = member_S + 0.5 * (physical - member_I)
        ensemble = remove_small_connected_regions(physical_reflectivity_mean(member_I, member_S, member_R) * y_mask_zmax)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f'tianjin_{case_name}_release_I_S_R_physical_mean.npz'
    np.savez_compressed(output, pred=to_numpy(ensemble))
    print(f'device: {device}')
    print(f'case: {case_name}')
    print(f'saved: {output}')
if __name__ == '__main__':
    main()

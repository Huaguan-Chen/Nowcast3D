# Nowcast3D: Reliable Precipitation Nowcasting via Gray-Box Learning

## Environment Setup

```bash
conda create -n nowcast3d python=3.12 -y
conda activate nowcast3d
```

Install PyTorch according to your CUDA version from https://pytorch.org. For example, for CUDA 12.6:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Install the remaining packages:

```bash
pip install matplotlib numpy tqdm scipy imageio einops
```

## Repository structure

```text
Nowcast3D/
|-- README.md
`-- 1024_4/
    |-- Nowcast3D_demo_1024_4.py
    |-- PhyPredNet.py
    |-- Diff_Structure.py
    |-- Diff_Intensity.py
    `-- fill_data_3D.py
```

## Checkpoints and test sample

The repository code and the model weights / sample folder are stored separately.

Suggested layout:

```text
Nowcast3D/
|-- README.md
`-- 1024_4/
    |-- Nowcast3D_demo_1024_4.py
    |-- PhyPredNet.py
    |-- Diff_Structure.py
    |-- Diff_Intensity.py
    |-- fill_data_3D.py
    |-- ckpt-1024-4/
    |   |-- PhyPredNet_1024_4.pth
    |   |-- Diff_Structure_1024_4.pth
    |   `-- Diff_Intensity_1024_4.pth
    `-- 20250728_120000/
        |-- *_data.npy
        `-- *_mask.npy
```

You can download the sample data and model checkpoints from https://drive.google.com/drive/folders/19utD5oIJ4x-mevyG5vmJgilteYWZDlrd?usp=sharing.

## Typical usage

### 1. Fill masked low-level frames

Edit `root_dir` in `1024_4/fill_data_3D.py`, then run:

```bash
python 1024_4/fill_data_3D.py
```

Default behavior:

- input folder: `20250728_120000`
- output folder: `20250728_120000_fill`

### 2. Run nowcasting demo

Edit `root_dir`, `folder_name`, and checkpoint paths in `1024_4/Nowcast3D_demo_1024_4.py`, then run:

```bash
python 1024_4/Nowcast3D_demo_1024_4.py
```

Default behavior:

- input: first 10 frames
- target for visualization: next 30 frames
- output: an `.mp4` file in `vis_results/`

## Important Note

> With authorization from the participating institutions, the **development and training of model configurations** were carried out across the **China Meteorological Administration Earth System Modeling and Prediction Center** and the **Tianjin Meteorological Observatory**.
>
> We also incorporated **additional training data**.
>
> To extend the radar reflectivity input from the original **24 x 256 x 256** to **24 x 512 x 512** while minimizing performance degradation, we **simplified the model architecture** to meet operational deployment requirements.
>
> The system has been **deployed and is currently in operational use** in **Tianjin, Hebei, Guangxi, and other regions of China**, as well as in **Pakistan**.
>
> Please note that the **best-performing checkpoints are used in operational deployments**. They are **confidential** and **will not be made public**.

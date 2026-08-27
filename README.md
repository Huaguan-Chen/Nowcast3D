# Nowcast3D: Reliable Precipitation Nowcasting via Gray-Box Learning


## Environment setup

```bash
git clone https://github.com/Huaguan-Chen/Nowcast3D.git
cd Nowcast3D

conda create -n nowcast3d python=3.12 -y
conda activate nowcast3d
```

Install PyTorch for your CUDA version from
[pytorch.org](https://pytorch.org/). For example, for CUDA 12.6:

```bash
pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
```

Install the remaining dependencies:

```bash
pip install numpy scipy matplotlib "imageio[ffmpeg]" einops tqdm
```

A CUDA-capable GPU is recommended. The `512_1` demo requires CUDA.

## Checkpoints and sample data

Download the public checkpoints and samples from:

[Google Drive: Nowcast3D checkpoints and sample data](https://drive.google.com/drive/folders/1af1wFGhWh-tnYq6wjDp-z3lI8pZZ9sUF?usp=drive_link)

The Google Drive folders map to the repository variants as follows:

| Google Drive folder | Repository directory |
|---|---|
| `10.24-0.04 (Sample Data & ckpts)` | `1024_4` |
| `5.12-0.01 (Sample Data & ckpts)` | `512_1` |
| `Original Radar Data` | nationwide 3-D radar mosaic BIN files |

Download the three checkpoint files from each `ckpts` folder and extract the
sample ZIP files into the corresponding repository directory. Keep the
checkpoint filenames unchanged.

The resulting layout should be:

```text
Nowcast3D/
|-- README.md
|-- Radardata2npy.py
|-- 1024_4/
|   |-- Nowcast3D_demo_1024_4.py
|   |-- fill_data_3D.py
|   |-- PhyPredNet.py
|   |-- Diff_Intensity.py
|   |-- Diff_Structure.py
|   |-- ckpt-1024-4/
|   |   |-- PhyPredNet_1024_4.pth
|   |   |-- Diff_Intensity_1024_4.pth
|   |   `-- Diff_Structure_1024_4.pth
|   `-- 20250728_120000/
|       |-- *_data.npy
|       `-- *_mask.npy
`-- 512_1/
    |-- Nowcast3D_demo_512_1.py
    |-- fill_data_3D.py
    |-- PhyPredNet.py
    |-- Diff_Intensity.py
    |-- Diff_Structure.py
    |-- ckpt-512-1/
    |   |-- PhyPredNet_512_1.pth
    |   |-- Diff_Intensity_512_1.pth
    |   `-- Diff_Structure_512_1.pth
    |-- 2025062905/
    |-- 2025082712/
    `-- 2025090911/
```

Each timestamp must have a matching pair:

```text
<timestamp>_data.npy
<timestamp>_mask.npy
```

## Run the 1024_4 demo (10.24° × 10.24°, 0.04°)

The default sample is `20250728_120000`. Run the commands from the `1024_4`
directory so that the default empty `root_dir` in `fill_data_3D.py` resolves
correctly:

```bash
cd 1024_4
python fill_data_3D.py
python Nowcast3D_demo_1024_4.py
cd ..
```

The fill script reads the first 10 data/mask pairs and creates
`20250728_120000_fill`. The demo uses those 10 filled volumes as input and the
next 30 raw volumes as the visualization reference.

The video is written to:

```text
1024_4/vis_results/20250728_120000_fill.mp4
```

To use another location or case, update `root_dir`, `input_name`, and
`output_name` in `fill_data_3D.py`, then update `FOLDER_NAME`, `FOLDER`,
`GT_FOLDER`, and `SAVE_DIR` in `Nowcast3D_demo_1024_4.py`.

## Run the 512_1 demo (5.12° × 5.12°, 0.01°)

The filling and forecasting steps are separate. For the default case:

```bash
python 512_1/fill_data_3D.py
python 512_1/Nowcast3D_demo_512_1.py
```

For another released case:

```bash
python 512_1/fill_data_3D.py \
  --input-dir 512_1/2025082712 \
  --output-dir 512_1/2025082712_fill

python 512_1/Nowcast3D_demo_512_1.py \
  --input-dir 512_1/2025082712_fill \
  --case-name 2025082712
```

The final forecast is saved as `512_1/npy_results/<case>_pred.npy`.

## Convert original radar files

`Radardata2npy.py` reads the original nationwide networked 3-D radar mosaic
BIN files. Each complete timestamp contains 24 vertical-level BIN files. The
script crops the nationwide radar grid to a specified geographic region and
saves the regional volume as paired `*_data.npy` and `*_mask.npy` files.

Before running it, edit `folder_in`, `folder_out`, the date range, and the
geographic crop near the bottom of the script:

```bash
python Radardata2npy.py
```

## Important note

With authorization from the participating institutions, **data preparation
and model training for model configurations** were carried out at
the **China Meteorological Administration Earth System Modeling and Prediction
Center** and the **Tianjin Meteorological Observatory**.

We also **expanded the training dataset** to support operational deployment.

To extend the radar reflectivity input from the original **`24 x 256 x 256`**
to **`24 x 512 x 512`** while minimizing performance degradation, the **model
architecture was simplified** to meet operational deployment requirements.
The system has been **deployed and is currently in operational use** in
**Tianjin, Hebei, Guangxi, and other regions of China**, as well as in
**Pakistan**.

Please note that the **released checkpoints is not the best-performing checkpoints**. The **best-performing checkpoints are used in operational
deployments**. They are **confidential** and **will not be made public**.

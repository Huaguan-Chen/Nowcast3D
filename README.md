## Nowcast3D: Reliable precipitation nowcasting viagray-box learning

### Repository structure

```text
Nowcast3D/
├── README.md
└── 1024_4/
    ├── Nowcast3D_demo_1024_4.py
    ├── PhyPredNet.py
    ├── Diff_Structure.py
    ├── Diff_Intensity.py
    └── fill_data_3D.py
```

### Checkpoints and test sample

The repository code and the model weights / sample folder are stored separately.

Suggested layout:

```text
Nowcast3D/
├── README.md
└── 1024_4/
    ├── Nowcast3D_demo_1024_4.py
    ├── PhyPredNet.py
    ├── Diff_Structure.py
    ├── Diff_Intensity.py
    ├── fill_data_3D.py
    ├── ckpt-1024-4/
    │   ├── PhyPredNet_1024_4.pth
    │   ├── Diff_Structure_1024_4.pth
    │   └── Diff_Intensity_1024_4.pth
    └── 20250728_120000/
        ├── *_data.npy
        └── *_mask.npy
```

You can download the sample data and model checkpoints from https://drive.google.com/drive/folders/19utD5oIJ4x-mevyG5vmJgilteYWZDlrd?usp=sharing.

### Typical usage

#### 1. Fill masked low-level frames

Edit `root_dir` in `1024_4/fill_data_3D.py`, then run:

```bash
python 1024_4/fill_data_3D.py
```

Default behavior:

- input folder: `xx`
- output folder: `xx_fill`

#### 2. Run nowcasting demo

Edit `root_dir`, `folder_name`, and checkpoint paths in `1024_4/Nowcast3D_demo_1024_4.py`, then run:

```bash
python 1024_4/Nowcast3D_demo_1024_4.py
```

Default behavior:

- input: first 10 frames
- target for visualization: next 30 frames
- output: an `.mp4` file in `vis_results/`




### Important Note

> **Our model has already been deployed at the Tianjin Meteorological Bureau, and the data preparation and model training were completed at the Tianjin Meteorological Observatory.**
>
> To extend the radar reflectivity input from the original **24 × 256 × 256** to **24 × 512 × 512** while minimizing performance degradation, we simplified the model architecture.
>
> **The related code and training checkpoints will be released soon.** Please note that the **best-performing checkpoints used in actual deployment are confidential** and will **not** be made public.

import os
import shutil
from glob import glob

import numpy as np

root_dir = ""
input_name = "Point3"
output_name = "Point3_fill"
input_dir = os.path.join(root_dir, input_name)
output_dir = os.path.join(root_dir, output_name)

LOW_LEVELS = [0, 1, 2, 3, 4]
HIGH_IDS = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]

RIDGE_ALPHA = 1e-2

THR_HI = 10.0
LOCAL_WIN = 5
LOCAL_FRAC = 0.30

GF_RADIUS_MAIN = 8
GF_EPS_MAIN = 1e-3

DETAIL_GAIN_MAIN = 0.25
DETAIL_GAIN_HIGH = 0.15
GF_RADIUS_HIGH = 3
GF_EPS_HIGH = 5e-4

DBZ_NZ_THR = 5


def load_frame(path):
    arr = np.load(path)
    original_shape = arr.shape
    if arr.shape == (24, 256, 256):
        vol = arr.astype(np.float32)
    elif arr.shape == (6144, 256):
        vol = arr.reshape(24, 256, 256).astype(np.float32)
    else:
        raise ValueError(f"Unexpected shape for {path}: {arr.shape}")
    return vol, original_shape


def save_frame(path, vol, original_shape):
    if original_shape == (24, 256, 256):
        out = vol
    elif original_shape == (6144, 256):
        out = vol.reshape(6144, 256)
    else:
        raise ValueError(f"Unexpected target shape: {original_shape}")
    np.save(path, out.astype(np.float32))


def to_dbz(vol_raw):
    return vol_raw.astype(np.float32) / 10.0


def from_dbz(vol_dbz):
    return vol_dbz.astype(np.float32) * 10.0


def build_predictor_matrix(vol_dbz, high_ids):
    cols = []
    for k in high_ids:
        x = vol_dbz[k].reshape(-1)
        mu = np.nanmean(x)
        sd = np.nanstd(x) + 1e-6
        x = (x - mu) / sd
        cols.append(x)
    return np.stack(cols, axis=1)


def ridge_fit_predict(X_obs, y_obs, X_all, alpha):
    X1 = np.hstack([X_obs, np.ones((X_obs.shape[0], 1), dtype=X_obs.dtype)])
    A = X1.T @ X1
    A[np.arange(A.shape[0]), np.arange(A.shape[0])] += alpha
    b = X1.T @ y_obs
    beta = np.linalg.solve(A, b)
    Xall1 = np.hstack([X_all, np.ones((X_all.shape[0], 1), dtype=X_all.dtype)])
    return Xall1 @ beta


def local_fraction(mask, win=5):
    H, W = mask.shape
    pad = win // 2
    m = mask.astype(np.uint8)
    m = np.pad(m, ((pad, pad), (pad, pad)), mode="constant", constant_values=0)
    s = np.zeros((H, W), dtype=np.int32)
    for dr in range(win):
        for dc in range(win):
            s += m[dr:dr + H, dc:dc + W]
    return s / float(win * win)


def box_filter(img, r):
    img = img.astype(np.float32, copy=False)
    k = 2 * r + 1
    im = np.pad(img, ((r, r), (r, r)), mode="reflect")
    ii = np.zeros((im.shape[0] + 1, im.shape[1] + 1), dtype=np.float32)
    ii[1:, 1:] = np.cumsum(np.cumsum(im, axis=0), axis=1)
    S = ii[k:, k:] - ii[:-k, k:] - ii[k:, :-k] + ii[:-k, :-k]
    return S / float(k * k)


def guided_filter_joint(I, p, r, eps):
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    mean_I = box_filter(I, r)
    mean_p = box_filter(p, r)
    corr_I = box_filter(I * I, r)
    corr_Ip = box_filter(I * p, r)

    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = box_filter(a, r)
    mean_b = box_filter(b, r)
    q = mean_a * I + mean_b
    return q


def guided_detail(guide, r, eps):
    base = guided_filter_joint(guide, guide, r, eps)
    return guide - base


def complete_volume_with_mask(vol_raw, mask_raw):
    vol_dbz = to_dbz(vol_raw)
    mask = mask_raw > 0

    if mask.shape != vol_dbz.shape:
        raise ValueError(f"Shape mismatch: vol={vol_dbz.shape}, mask={mask.shape}")

    upper_mean = np.nanmean(vol_dbz[HIGH_IDS, ...], axis=0)
    hi_pixel = upper_mean >= THR_HI
    hi_local = local_fraction(hi_pixel, win=LOCAL_WIN) >= LOCAL_FRAC
    hi_any = hi_pixel | hi_local

    X_all = build_predictor_matrix(vol_dbz, HIGH_IDS)

    guide_main = upper_mean
    detail_main = guided_detail(guide_main, r=GF_RADIUS_MAIN, eps=GF_EPS_MAIN)
    detail_high = guided_detail(guide_main, r=GF_RADIUS_HIGH, eps=GF_EPS_HIGH)

    vol_out = vol_dbz.copy()

    for Lk in LOW_LEVELS:
        low = vol_dbz[Lk]
        obs = mask[Lk]

        unknown_mask = (~obs) & hi_any
        known_mask = obs

        if unknown_mask.sum() == 0:
            continue

        y = low.reshape(-1)
        keep_idx = np.where(known_mask.reshape(-1))[0]

        if keep_idx.size == 0:
            continue

        Xobs = X_all[keep_idx]
        yobs = y[keep_idx]

        y_pred_all = ridge_fit_predict(Xobs, yobs, X_all, alpha=RIDGE_ALPHA).reshape(low.shape)

        u = low.copy()
        u[unknown_mask] = y_pred_all[unknown_mask]

        u_guided = guided_filter_joint(guide_main, u, r=GF_RADIUS_MAIN, eps=GF_EPS_MAIN)
        u[unknown_mask] = u_guided[unknown_mask]

        u[unknown_mask] += DETAIL_GAIN_MAIN * detail_main[unknown_mask]
        u[unknown_mask] += DETAIL_GAIN_HIGH * detail_high[unknown_mask]
        u[unknown_mask] = np.clip(u[unknown_mask], -10.0, 80.0)

        obs_nonzero = known_mask & (low > DBZ_NZ_THR)
        filled_nonzero = unknown_mask & (u > DBZ_NZ_THR)

        obs_count = int(np.count_nonzero(obs_nonzero))
        filled_count = int(np.count_nonzero(filled_nonzero))

        if obs_count > 0 and filled_count > 0:
            obs_mean = float(np.mean(low[obs_nonzero]))
            filled_mean = float(np.mean(u[filled_nonzero]))
            if filled_mean != 0.0:
                scalar = obs_mean / max(filled_mean, 1e-6)
                u[filled_nonzero] *= scalar
                u[filled_nonzero] = np.clip(u[filled_nonzero], -10.0, 80.0)

        u[known_mask] = low[known_mask]
        vol_out[Lk] = u

    return from_dbz(vol_out)


def main():
    os.makedirs(output_dir, exist_ok=True)

    data_files = sorted(glob(os.path.join(input_dir, "*_data.npy")))[:10]
    mask_files = sorted(glob(os.path.join(input_dir, "*_mask.npy")))[:10]

    if len(data_files) < 10 or len(mask_files) < 10:
        raise ValueError("Point3 does not contain 10 data files and 10 mask files")

    filled_list = []

    for data_path, mask_path in zip(data_files, mask_files):
        vol_raw, data_shape = load_frame(data_path)
        mask_raw, mask_shape = load_frame(mask_path)

        filled_raw = complete_volume_with_mask(vol_raw, mask_raw)

        data_name = os.path.basename(data_path)
        mask_name = os.path.basename(mask_path)

        save_frame(os.path.join(output_dir, data_name), filled_raw, data_shape)
        save_frame(os.path.join(output_dir, mask_name), mask_raw, mask_shape)

        filled_list.append(filled_raw)

        print(f"saved: {os.path.join(output_dir, data_name)}")

    point3_fill = np.stack(filled_list, axis=0)
    np.save(os.path.join(root_dir, "point3_fill.npy"), point3_fill.astype(np.float32))
    print(f"saved: {os.path.join(root_dir, 'point3_fill.npy')}")
    print(f"output_dir: {output_dir}")


if __name__ == "__main__":
    main()
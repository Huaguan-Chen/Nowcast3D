# -*- coding: utf-8 -*-
import bz2
import struct
import os
from datetime import datetime, timedelta
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm


class MosaicParser(object):
    """Parse gridded mosaic radar files with a 256-byte header."""

    def __init__(self):
        super(MosaicParser, self).__init__()

    def parse(self, buf):
        """Parse one reflectivity layer from a radar binary buffer."""
        compress_flag = struct.unpack('h', buf[166:168])[0]
        cols = struct.unpack('i', buf[148:152])[0]
        rows = struct.unpack('i', buf[152:156])[0]
        compress_data = buf[256:]

        if compress_flag == 1:
            try:
                data = bz2.decompress(compress_data)
            except Exception as exc:
                print("BZip2 decompression failed:", exc)
                return None
            if not data:
                return None
        else:
            data = compress_data

        try:
            data_arr = np.frombuffer(data, dtype=np.int16)
            data_arr = np.reshape(data_arr, (rows, cols))
        except Exception as exc:
            print("Radar data parsing or reshaping failed:", exc)
            return None

        llat = struct.unpack('i', buf[124:128])[0]
        llon = struct.unpack('i', buf[128:132])[0]
        ulat = struct.unpack('i', buf[132:136])[0]
        ulon = struct.unpack('i', buf[136:140])[0]

        return (
            data_arr,
            llat / 1000.0,
            llon / 1000.0,
            ulat / 1000.0,
            ulon / 1000.0,
            buf[0:256],
        )

    def parseUV(self, buf):
        """Parse interleaved U and V wind components from a binary buffer."""
        compress_flag = struct.unpack('h', buf[166:168])[0]
        cols = struct.unpack('i', buf[148:152])[0]
        rows = struct.unpack('i', buf[152:156])[0]
        compress_data = buf[256:]

        if compress_flag == 1:
            try:
                data = bz2.decompress(compress_data)
            except Exception as exc:
                print("BZip2 decompression failed:", exc)
                return None
            if not data:
                return None
        else:
            data = compress_data

        data_arr = np.frombuffer(data, dtype=np.int16)
        u_arr = data_arr[::2]
        v_arr = data_arr[1::2]
        try:
            u_arr = np.reshape(u_arr, (rows, cols))
            v_arr = np.reshape(v_arr, (rows, cols))
        except Exception as exc:
            print("Wind data reshaping failed:", exc)
            return None

        llat = struct.unpack('i', buf[124:128])[0]
        llon = struct.unpack('i', buf[128:132])[0]
        ulat = struct.unpack('i', buf[132:136])[0]
        ulon = struct.unpack('i', buf[136:140])[0]
        cx = struct.unpack('i', buf[140:144])[0]
        cy = struct.unpack('i', buf[144:148])[0]

        return (
            u_arr,
            v_arr,
            cx,
            cy,
            llat / 10000.0,
            llon / 10000.0,
            ulat / 10000.0,
            ulon / 10000.0,
        )


# Global radar grid metadata.
LAT_MAX_GLOBAL = 54.2
LAT_MIN_GLOBAL = 12.2
LON_MIN_GLOBAL = 73.0
LON_MAX_GLOBAL = 135.0
ROWS_GLOBAL = 4200
COLS_GLOBAL = 6200


def latlon_to_indices(
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    lat_min_glob=LAT_MIN_GLOBAL,
    lat_max_glob=LAT_MAX_GLOBAL,
    lon_min_glob=LON_MIN_GLOBAL,
    lon_max_glob=LON_MAX_GLOBAL,
    rows=ROWS_GLOBAL,
    cols=COLS_GLOBAL,
):
    """
    Convert a latitude-longitude box to NumPy slice boundaries.

    Rows run north to south. Columns run west to east. The returned upper
    boundaries are exclusive.
    """
    lat_res = (lat_max_glob - lat_min_glob) / rows
    lon_res = (lon_max_glob - lon_min_glob) / cols

    if lat_max < lat_min:
        lat_min, lat_max = lat_max, lat_min
    if lon_max < lon_min:
        lon_min, lon_max = lon_max, lon_min

    row_start = int(np.floor((lat_max_glob - lat_max) / lat_res))
    row_end = int(np.ceil((lat_max_glob - lat_min) / lat_res))
    col_start = int(np.floor((lon_min - lon_min_glob) / lon_res))
    col_end = int(np.ceil((lon_max - lon_min_glob) / lon_res))

    row_start = max(0, min(rows, row_start))
    row_end = max(0, min(rows, row_end))
    col_start = max(0, min(cols, col_start))
    col_end = max(0, min(cols, col_end))

    return row_start, row_end, col_start, col_end


def read_Radar(filename_base, row_start, row_end, col_start, col_end):
    """
    Read and crop all 24 height layers for one radar time step.

    Return None if any layer is missing or cannot be parsed. Otherwise return
    arrays with shapes (24, height, width) for reflectivity and validity mask.
    """
    data_Radar = []
    mask_Radar = []

    for i in range(24):
        filename_i = filename_base + '_' + str(i).zfill(2) + '.bin'
        try:
            with open(filename_i, 'rb') as file:
                buf = file.read()
        except Exception as exc:
            print(f"Failed to read {filename_i}: {exc}")
            return None

        radar_parser = MosaicParser()
        parsed = radar_parser.parse(buf)
        if parsed is None:
            print(f"Parser returned None for {filename_i}")
            return None

        data_full = parsed[0]

        try:
            data0 = data_full[row_start:row_end, col_start:col_end]
        except Exception as exc:
            print(f"Failed to crop {filename_i}: {exc}")
            return None

        # Preserve the original preprocessing: clamp negative values to zero.
        data_r_i = np.maximum(data0, 0).astype(np.float32)
        data_r_i = np.expand_dims(data_r_i, axis=0)

        # The original int16 value -32768 marks missing data.
        mask_r_i = np.where(data0 > -32768, 1, 0)
        mask_r_i = np.expand_dims(mask_r_i, axis=0)

        data_Radar.append(data_r_i)
        mask_Radar.append(mask_r_i)

    if len(data_Radar) != 24 or len(mask_Radar) != 24:
        return None

    return np.concatenate(data_Radar, axis=0), np.concatenate(mask_Radar, axis=0)


def find_complete_datetimes(folder_path, start_date=None, end_date=None):
    """
    Find time steps that contain exactly 24 radar layer files.

    Dates are optionally restricted to the inclusive YYYYMMDD interval defined
    by start_date and end_date.
    """
    complete = {}
    with os.scandir(folder_path) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            if not entry.name.endswith('.bin'):
                continue

            parts = entry.name.split('_')
            if len(parts) < 3:
                continue

            date_str = parts[0]
            time_str = parts[1]

            if start_date and date_str < start_date:
                continue
            if end_date and date_str > end_date:
                continue

            dt = date_str + '_' + time_str
            complete.setdefault(dt, 0)
            complete[dt] += 1

    return [dt for dt, count in complete.items() if count == 24]


def process_group(
    dt,
    folder_in,
    folder_out,
    row_start,
    row_end,
    col_start,
    col_end,
):
    """Convert one complete 24-layer radar time step to data and mask files."""
    filename_base = os.path.join(folder_in, dt)
    result = read_Radar(filename_base, row_start, row_end, col_start, col_end)
    if result is None:
        print(f"Skipped {dt}: one or more radar layers are missing or unreadable.")
        return None

    data, mask = result
    np.save(os.path.join(folder_out, dt + '_data.npy'), data)
    np.save(os.path.join(folder_out, dt + '_mask.npy'), mask)
    print(f"Saved {dt}")
    return dt


if __name__ == "__main__":
    # Input and output directories.
    folder_in = '/home/qqxt/data/3Dradar/3Dradar'
    folder_out = '/home/qqxt/data/3Dradar/3Dradar_new'

    # Inclusive date range in YYYYMMDD format. Use None for no limit.
    start_date = '20240501'
    end_date = '20260531'

    # Geographic crop boundaries.
    lat_min_crop = 20.00
    lat_max_crop = 25.12
    lon_min_crop = 105.01
    lon_max_crop = 110.12

    row_start, row_end, col_start, col_end = latlon_to_indices(
        lat_min_crop,
        lat_max_crop,
        lon_min_crop,
        lon_max_crop,
        rows=ROWS_GLOBAL,
        cols=COLS_GLOBAL,
    )
    print(
        "Crop coordinates:",
        f"lat [{lat_min_crop}, {lat_max_crop}], "
        f"lon [{lon_min_crop}, {lon_max_crop}]",
    )
    print(
        "Crop indices:",
        f"row [{row_start}:{row_end}], col [{col_start}:{col_end}]",
    )

    max_workers = 4

    if not os.path.exists(folder_out):
        os.makedirs(folder_out)

    complete_list = find_complete_datetimes(folder_in, start_date, end_date)
    print(
        f"Found {len(complete_list)} complete time steps "
        f"from {start_date or 'unrestricted'} to {end_date or 'unrestricted'}."
    )

    if not complete_list:
        print("No matching complete radar time steps were found.")
        exit(0)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                process_group,
                dt,
                folder_in,
                folder_out,
                row_start,
                row_end,
                col_start,
                col_end,
            )
            for dt in complete_list
        ]
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Processing",
        ):
            try:
                future.result()
            except Exception as exc:
                print(f"Worker failed: {exc}")
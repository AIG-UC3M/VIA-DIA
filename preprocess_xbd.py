"""
preprocess_xbd.py  —  ONE-TIME preprocessing script
=====================================================
Run ONCE before training:

    python preprocess_xbd.py --data_dir /path/to/xBD --patch_size 64

Output layout
-------------
cache_train/
    metadata.npz                 global index: labels, centers, disaster per patch
    disasters/
        hurricane-harvey/
            patches_pre.npy      float32  (N_d, 3, P, P)  mmap
            patches_post.npy     float32  (N_d, 3, P, P)  mmap
            masks.npy            uint8    (N_d, P, P)      mmap
            climate.npy          float16  (C, T, H, W)     fully in RAM at train time
            event_labels.npy     float32  (T,)
        midwest-flooding/
            ...
cache_val/   (same layout)
cache_test/  (same layout)

Benefits of per-disaster files
-------------------------------
- Resumable: crash on disaster N leaves N-1 complete and skipped on rerun.
- Bounded peak RAM: one disaster processed at a time, freed before the next.
- Largest single file ~1.3 GB vs ~14 GB monolith.
- OS page-cache works efficiently on smaller files.
- Climate files are the same across splits (just symlinked conceptually).
"""

import os
import gc
import json
import argparse
import numpy as np
import cv2
import tifffile
from datetime import timedelta, datetime
from natsort import natsorted
from itertools import chain
from shapely.wkt import loads
from shapely.geometry import Polygon
from tqdm import tqdm

eps = 1e-7

RAW_EVENT_DATES = {
    "guatemala-volcano":   ("2018-06-03", "2018-06-03"),
    "hurricane-michael":   ("2018-10-07", "2018-10-16"),
    "santa-rosa-wildfire": ("2017-10-08", "2017-10-31"),
    "hurricane-florence":  ("2018-09-10", "2018-09-19"),
    "midwest-flooding":    ("2019-01-03", "2019-05-31"),
    "palu-tsunami":        ("2018-09-18", "2018-09-18"),
    "socal-fire":          ("2018-07-23", "2018-08-30"),
    "hurricane-harvey":    ("2017-08-17", "2017-09-02"),
    "mexico-earthquake":   ("2017-09-19", "2017-09-19"),
    "hurricane-matthew":   ("2016-09-28", "2016-10-10"),
    "nepal-flooding":      ("2017-07-01", "2017-09-30"),
    "moore-tornado":       ("2013-05-20", "2013-05-20"),
    "tuscaloosa-tornado":  ("2011-04-27", "2011-04-27"),
    "sunda-tsunami":       ("2018-12-22", "2018-12-22"),
    "lower-puna-volcano":  ("2018-05-23", "2018-08-14"),
    "joplin-tornado":      ("2011-05-22", "2011-05-22"),
    "woolsey-fire":        ("2018-11-09", "2018-11-28"),
    "pinery-bushfire":     ("2018-11-25", "2018-12-02"),
    "portugal-wildfire":   ("2017-06-17", "2017-06-24"),
}
DAMAGE_CLASSES = {'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3}
CLIMATE_VARS   = [
    '2m_temperature', 'total_precipitation',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    'volumetric_soil_water_layer_1', 'surface_solar_radiation_downwards',
    'surface_latent_heat_flux', 'leaf_area_index_high_vegetation',
    'surface_pressure', 'potential_evaporation',
]
PADDING_DAYS = 90


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _event_info():
    result, max_t = {}, 0
    for e, (s, en) in RAW_EVENT_DATES.items():
        start = datetime.strptime(s,  "%Y-%m-%d") - timedelta(days=PADDING_DAYS)
        end   = datetime.strptime(en, "%Y-%m-%d") + timedelta(days=PADDING_DAYS)
        n = (end - start).days + 1
        result[e] = n
        max_t = max(max_t, n)
    return result, max_t


def _disaster_name(path):
    for event in RAW_EVENT_DATES:
        if event in path:
            return event
    raise ValueError(f'No disaster name found in: {path}')


def _collect_file_lists(data_dir, splits):
    pre, post, lpre, lpost, clim = [], [], [], [], []
    for s in splits:
        for d in os.listdir(os.path.join(data_dir, s)):
            base = os.path.join(data_dir, s, d)
            imgs = os.listdir(os.path.join(base, 'images'))
            lbls = os.listdir(os.path.join(base, 'labels'))
            era5 = os.listdir(os.path.join(base, 'era5'))
            pre  .append(natsorted([os.path.join(base,'images',f) for f in imgs if 'pre'  in f]))
            post .append(natsorted([os.path.join(base,'images',f) for f in imgs if 'post' in f]))
            lpre .append(natsorted([os.path.join(base,'labels',f) for f in lbls if 'pre'  in f]))
            lpost.append(natsorted([os.path.join(base,'labels',f) for f in lbls if 'post' in f]))
            clim .append(natsorted([os.path.join(base,'era5',  f) for f in era5 if f.endswith('.npz')]))
    flat = lambda ll: natsorted(list(chain.from_iterable(ll)))
    return flat(pre), flat(post), flat(lpre), flat(lpost), flat(clim)


def _read_image(path):
    img = tifffile.imread(path)
    if img.shape[:2] != (1024, 1024):
        img = cv2.resize(img, (1024, 1024))
    return img


def _read_mask(path):
    mask = np.zeros((1024, 1024), dtype=np.uint8)
    try:
        with open(path) as f:
            data = json.load(f)
        if 'features' in data and 'xy' in data['features']:
            for feat in data['features']['xy']:
                if 'wkt' in feat:
                    geom = loads(feat['wkt'])
                    if isinstance(geom, Polygon) and geom.is_valid:
                        coords = np.array(geom.exterior.coords, dtype=np.int32)
                        cv2.fillPoly(mask, [coords], 1)
    except Exception:
        pass
    return mask


def _extract_patch(arr, center, patch_size):
    cx, cy = center
    h = patch_size // 2
    x1 = max(0, cx - h);  x2 = min(arr.shape[1], cx + h)
    y1 = max(0, cy - h);  y2 = min(arr.shape[0], cy + h)
    p = arr[y1:y2, x1:x2]
    if p.shape[0] != patch_size or p.shape[1] != patch_size:
        p = cv2.resize(p, (patch_size, patch_size))
    return p


# ---------------------------------------------------------------------------
# Label parsing  →  records grouped by disaster
# ---------------------------------------------------------------------------

def _build_records_by_disaster(img_pre_files, img_post_files,
                                lbl_pre_files, lbl_post_files, clim_files):
    """Returns dict: disaster_name → list of patch dicts."""
    by_disaster = {}
    for n in tqdm(range(len(img_pre_files)), desc='Parsing labels'):
        with open(lbl_pre_files[n])  as fp: data_pre  = json.load(fp)
        with open(lbl_post_files[n]) as fp: data_post = json.load(fp)
        if 'features' not in data_post or 'xy' not in data_post['features']:
            continue
        meta_pre  = data_pre['metadata']
        meta_post = data_post['metadata']
        disaster  = _disaster_name(img_pre_files[n])

        for feat in data_post['features']['xy']:
            if 'wkt' not in feat or 'properties' not in feat:
                continue
            props = feat['properties']
            if props.get('feature_type') != 'building':
                continue
            dmg = props.get('subtype', 'no-damage')
            if dmg not in DAMAGE_CLASSES:
                continue
            try:
                geom = loads(feat['wkt'])
                if not (isinstance(geom, Polygon) and geom.is_valid):
                    continue
                coords = np.array(geom.exterior.coords, dtype=np.int32)
                xmin, ymin = coords.min(axis=0)
                xmax, ymax = coords.max(axis=0)
            except Exception:
                continue

            rec = dict(
                img_pre    = img_pre_files[n],
                img_post   = img_post_files[n],
                lbl_post   = lbl_post_files[n],
                clim_file  = clim_files[n],
                pre_date   = meta_pre['capture_date'],
                post_date  = meta_post['capture_date'],
                label_pre  = 0,
                label_post = DAMAGE_CLASSES[dmg],
                center     = ((xmin + xmax) // 2, (ymin + ymax) // 2),
            )
            by_disaster.setdefault(disaster, []).append(rec)

    return by_disaster


# ---------------------------------------------------------------------------
# Statistics — vectorised parallel Welford, one pass per data type
# ---------------------------------------------------------------------------

def _compute_stats(records_by_disaster, patch_size):
    print('Computing normalisation statistics (vectorised Welford)...')

    n_p  = np.zeros(3,             dtype=np.float64)
    mu_p = np.zeros(3,             dtype=np.float64)
    M2_p = np.zeros(3,             dtype=np.float64)
    n_c  = np.zeros(len(CLIMATE_VARS), dtype=np.float64)
    mu_c = np.zeros(len(CLIMATE_VARS), dtype=np.float64)
    M2_c = np.zeros(len(CLIMATE_VARS), dtype=np.float64)

    all_records = [r for recs in records_by_disaster.values() for r in recs]

    # Image patch stats
    prev_img = None
    img_pre_arr = img_post_arr = None
    for rec in tqdm(all_records, desc='Image stats'):
        if rec['img_pre'] != prev_img:
            img_pre_arr  = _read_image(rec['img_pre']).astype(np.float64)
            img_post_arr = _read_image(rec['img_post']).astype(np.float64)
            prev_img = rec['img_pre']
        for arr in (img_pre_arr, img_post_arr):
            patch = _extract_patch(arr, rec['center'], patch_size)
            flat  = patch.reshape(-1, 3)
            valid = flat[~np.isnan(flat).any(axis=1)]
            nb    = valid.shape[0]
            if nb == 0:
                continue
            mu_b  = valid.mean(axis=0)
            M2_b  = ((valid - mu_b)**2).sum(axis=0)
            delta = mu_b - mu_p
            new_n = n_p + nb
            mu_p  = (n_p * mu_p + nb * mu_b) / new_n
            M2_p += M2_b + delta**2 * (n_p * nb / new_n)
            n_p   = new_n

    # Climate stats — one load per unique file
    seen = set()
    for rec in tqdm(all_records, desc='Climate stats'):
        cf = rec['clim_file']
        if cf in seen:
            continue
        seen.add(cf)
        with np.load(cf, allow_pickle=True) as f:
            cs = f['data'].astype(np.float64)
        for v in range(cs.shape[0]):
            flat  = cs[v].ravel()
            valid = flat[~np.isnan(flat)]
            nb    = valid.size
            if nb == 0:
                continue
            mu_b  = valid.mean()
            M2_b  = ((valid - mu_b)**2).sum()
            delta = mu_b - mu_c[v]
            new_n = n_c[v] + nb
            mu_c[v]  = (n_c[v] * mu_c[v] + nb * mu_b) / new_n
            M2_c[v] += M2_b + delta**2 * (n_c[v] * nb / new_n)
            n_c[v]   = new_n
        del cs

    return (mu_p.astype(np.float32),
            np.sqrt(M2_p / np.maximum(n_p - 1, 1)).astype(np.float32),
            mu_c.astype(np.float32),
            np.sqrt(M2_c / np.maximum(n_c - 1, 1)).astype(np.float32))


# ---------------------------------------------------------------------------
# Per-disaster writer
# ---------------------------------------------------------------------------

def _write_disaster(disaster, records, out_dir, patch_size,
                    p_mean, p_std, climate_mean, climate_std,
                    event_num_ts, max_cH, max_cW):
    """
    Write all files for one disaster into out_dir.
    Peak RAM: current_image (~12 MB) + one climate volume (~5 MB).
    All arrays freed before returning.
    """
    os.makedirs(out_dir, exist_ok=True)
    N_d = len(records)

    # ---- allocate per-disaster memmaps ------------------------------------
    def mmap(name, dtype, shape):
        return np.memmap(os.path.join(out_dir, name),
                         dtype=dtype, mode='w+', shape=shape)

    mm_pre  = mmap('patches_pre.npy',  np.float32, (N_d, 3, patch_size, patch_size))
    mm_post = mmap('patches_post.npy', np.float32, (N_d, 3, patch_size, patch_size))
    mm_mask = mmap('masks.npy',        np.uint8,   (N_d, patch_size, patch_size))

    # ---- extract patches --------------------------------------------------
    prev_img_pre = prev_img_post = prev_mask_path = None
    img_pre_arr  = img_post_arr  = mask_arr        = None

    for i, rec in enumerate(tqdm(records, desc=f'  Patches {disaster}', leave=False)):
        if rec['img_pre'] != prev_img_pre:
            img_pre_arr  = _read_image(rec['img_pre'])
            prev_img_pre = rec['img_pre']
        if rec['img_post'] != prev_img_post:
            img_post_arr  = _read_image(rec['img_post'])
            prev_img_post = rec['img_post']
        if rec['lbl_post'] != prev_mask_path:
            mask_arr       = _read_mask(rec['lbl_post'])
            prev_mask_path = rec['lbl_post']

        center = rec['center']
        pp = _extract_patch(img_pre_arr,  center, patch_size).astype(np.float32)
        pq = _extract_patch(img_post_arr, center, patch_size).astype(np.float32)
        mk = _extract_patch(mask_arr,     center, patch_size)

        pp = np.clip((pp - p_mean) / (3 * p_std + eps), -1.0, 1.0)
        pq = np.clip((pq - p_mean) / (3 * p_std + eps), -1.0, 1.0)

        mm_pre [i] = pp.transpose(2, 0, 1)
        mm_post[i] = pq.transpose(2, 0, 1)
        mm_mask[i] = mk

    del mm_pre, mm_post, mm_mask   # flush + release
    del img_pre_arr, img_post_arr, mask_arr
    gc.collect()

    # ---- climate ----------------------------------------------------------
    # Use the first record's climate file (all records in same disaster
    # share the same file; representative dates from first record).
    rep = records[0]
    with np.load(rep['clim_file'], allow_pickle=True) as f:
        cs_raw      = f['data']
        valid_times = f['times']

    C, T, raw_H, raw_W = cs_raw.shape
    c_mean = climate_mean.reshape(-1, 1, 1, 1)
    c_std  = climate_std .reshape(-1, 1, 1, 1)

    cs_norm = np.clip((cs_raw.astype(np.float32) - c_mean)
                      / (3 * c_std + eps), -1.0, 1.0)
    del cs_raw

    event_T = event_num_ts[disaster]
    cs_out  = np.zeros((C, event_T, max_cH, max_cW), dtype=np.float16)
    cs_out[:, :min(T, event_T), :raw_H, :raw_W] = \
        cs_norm[:, :min(T, event_T)].astype(np.float16)
    np.nan_to_num(cs_out, copy=False)
    del cs_norm

    ev_start, ev_end = RAW_EVENT_DATES[disaster]
    ev_labels = np.zeros(event_T, dtype=np.float32)
    ev_labels[np.logical_and(valid_times >= np.datetime64(ev_start),
                              valid_times <= np.datetime64(ev_end))] = 1.0
    vtd = valid_times[:event_T].astype('datetime64[D]')

    # Use representative pre/post dates from first record
    pre_idx  = np.where(vtd == np.datetime64(rep['pre_date']) .astype('datetime64[D]'))[0]
    post_idx = np.where(vtd == np.datetime64(rep['post_date']).astype('datetime64[D]'))[0]
    ev_labels[pre_idx [0] if len(pre_idx)  else 0]  = 2.0
    ev_labels[post_idx[0] if len(post_idx) else -1] = 3.0
    ev_labels[-1] = 4.0

    np.save(os.path.join(out_dir, 'climate.npy'),      cs_out)
    np.save(os.path.join(out_dir, 'event_labels.npy'), ev_labels)
    del cs_out, ev_labels
    gc.collect()

    # ---- per-disaster metadata --------------------------------------------
    np.savez(os.path.join(out_dir, 'metadata.npz'),
             label_pre       = np.array([r['label_pre']  for r in records], dtype=np.int64),
             label_post      = np.array([r['label_post'] for r in records], dtype=np.int64),
             centers         = np.array([r['center']     for r in records], dtype=np.int32),
             pre_dates       = np.array([r['pre_date']   for r in records]),
             post_dates      = np.array([r['post_date']  for r in records]),
             img_pre_paths   = np.array([r['img_pre']    for r in records]),
             img_post_paths  = np.array([r['img_post']   for r in records]),
             lbl_post_paths  = np.array([r['lbl_post']   for r in records]),
             clim_file_paths = np.array([r['clim_file']  for r in records]),
             n_patches       = N_d,
             patch_size      = patch_size,
             )


# ---------------------------------------------------------------------------
# Top-level coordinator
# ---------------------------------------------------------------------------

def extract_and_save(data_dir, splits, patch_size, cache_dir, stats_path):
    os.makedirs(cache_dir, exist_ok=True)

    img_pre_f, img_post_f, lbl_pre_f, lbl_post_f, clim_f = \
        _collect_file_lists(data_dir, splits)

    records_by_disaster = _build_records_by_disaster(
        img_pre_f, img_post_f, lbl_pre_f, lbl_post_f, clim_f)

    N_total = sum(len(v) for v in records_by_disaster.values())
    print(f'Total patches: {N_total} across {len(records_by_disaster)} disasters')

    if N_total == 0:
        print('Nothing to process.')
        return

    # ---------------------------------------------------------------- stats
    if os.path.isfile(stats_path):
        print(f'Loading existing stats from {stats_path}')
        s = np.load(stats_path)
        patch_mean, patch_std     = s['patch_mean'],   s['patch_std']
        climate_mean, climate_std = s['climate_mean'], s['climate_std']
    else:
        patch_mean, patch_std, climate_mean, climate_std = \
            _compute_stats(records_by_disaster, patch_size)
        np.savez(stats_path,
                 patch_mean=patch_mean, patch_std=patch_std,
                 climate_mean=climate_mean, climate_std=climate_std)

    p_mean = patch_mean.reshape(1, 1, -1).astype(np.float32)
    p_std  = patch_std .reshape(1, 1, -1).astype(np.float32)
    event_num_ts, _ = _event_info()

    # ------------------------------------------ probe max spatial dims once
    print('Scanning climate files for spatial dimensions...')
    cH, cW = 0, 0
    seen_cf = set()
    for recs in records_by_disaster.values():
        cf = recs[0]['clim_file']
        if cf in seen_cf:
            continue
        seen_cf.add(cf)
        with np.load(cf, allow_pickle=True) as f:
            _, _, h, w = f['data'].shape
        cH = max(cH, h);  cW = max(cW, w)
    print(f'Climate spatial dims (max): H={cH}, W={cW}')

    # --------------------------------- process one disaster at a time
    disasters_dir = os.path.join(cache_dir, 'disasters')
    disaster_sizes = {}   # disaster → N_d  (for global metadata)

    for disaster, records in sorted(records_by_disaster.items()):
        out_dir   = os.path.join(disasters_dir, disaster)
        done_flag = os.path.join(out_dir, 'metadata.npz')

        if os.path.isfile(done_flag):
            print(f'[skip] {disaster} already done')
            meta = np.load(done_flag)
            disaster_sizes[disaster] = int(meta['n_patches'])
            continue

        print(f'\n[{disaster}]  N={len(records)}')
        _write_disaster(
            disaster     = disaster,
            records      = records,
            out_dir      = out_dir,
            patch_size   = patch_size,
            p_mean       = p_mean,
            p_std        = p_std,
            climate_mean = climate_mean,
            climate_std  = climate_std,
            event_num_ts = event_num_ts,
            max_cH       = cH,
            max_cW       = cW,
        )
        disaster_sizes[disaster] = len(records)

    # --------------------------------- global metadata (index into disasters)
    # Stores only lightweight arrays: one entry per patch pointing to its
    # disaster and local index within that disaster's files.
    global_disaster = []
    global_local_idx = []
    global_label_post = []

    for disaster in sorted(disaster_sizes.keys()):
        out_dir = os.path.join(disasters_dir, disaster)
        meta    = np.load(os.path.join(out_dir, 'metadata.npz'))
        n       = int(meta['n_patches'])
        global_disaster  .extend([disaster] * n)
        global_local_idx .extend(range(n))
        global_label_post.extend(meta['label_post'].tolist())

    np.savez(os.path.join(cache_dir, 'metadata.npz'),
             disaster      = np.array(global_disaster),
             local_idx     = np.array(global_local_idx,  dtype=np.int32),
             label_post    = np.array(global_label_post, dtype=np.int64),
             patch_mean    = patch_mean,
             patch_std     = patch_std,
             climate_mean  = climate_mean,
             climate_std   = climate_std,
             patch_size    = patch_size,
             max_cH        = cH,
             max_cW        = cW,
             max_T         = max(event_num_ts.values()),
             )
    print(f'\nDone. Cache written to {cache_dir}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',   required=True)
    parser.add_argument('--patch_size', type=int, default=64)
    args = parser.parse_args()

    stats_path = os.path.join(args.data_dir, 'data_py', 'stats.npz')
    os.makedirs(os.path.join(args.data_dir, 'data_py'), exist_ok=True)

    for split_group, tag in [
        (['tier1', 'tier3'], 'train'),
        (['hold'],           'val'),
        (['test'],           'test'),
    ]:
        valid = [s for s in split_group
                 if os.path.isdir(os.path.join(args.data_dir, s))]
        if not valid:
            continue
        cache_dir = os.path.join(args.data_dir, f'cache_{tag}')
        print(f'\n=== Processing {tag} ({valid}) → {cache_dir} ===')
        extract_and_save(args.data_dir, valid, args.patch_size,
                         cache_dir, stats_path)

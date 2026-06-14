"""
xBDClimate_database.py
======================
Memory-mapped xBD + ERA-5 dataset.  Requires preprocess_xbd.py to have run.

Cache layout:
    cache_train/
        metadata.npz              global index: disaster + local_idx per patch
        disasters/
            hurricane-harvey/
                patches_pre.npy   float32 (N_d, 3, P, P)  mmap
                patches_post.npy  float32 (N_d, 3, P, P)  mmap
                masks.npy         uint8   (N_d, P, P)      mmap
                climate.npy       float16 (C, T, H, W)     in RAM
                event_labels.npy  float32 (T,)             in RAM
                metadata.npz      per-disaster labels/centers
            ...

__getitem__ cost (per sample):
    disaster lookup     O(1) dict
    memmap[local_idx]   OS page-cache hit for hot disasters
    .copy()             small buffer copy (64×64 patch)
    augmentation        optional
    torch.from_numpy    zero-copy
"""

import os
import random
import warnings
import numpy as np
import cv2
import torch
from collections import defaultdict, deque
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.utils.data.dataloader import default_collate

warnings.filterwarnings('ignore')

# Module-level climate cache: loaded once in the main process before DataLoader
# spawns workers. Forked workers inherit it via copy-on-write → each disaster's
# (C,T,H,W) array is physically shared across all workers at OS level.
# __getitem__ returns a VIEW (not a copy) so the collate function receives
# 32 pointers to the same array and can use expand() instead of stack().
_CLIMATE_CACHE: dict = {}   # disaster_name → (C, T, H, W) float32 numpy array
_EV_CACHE:      dict = {}   # disaster_name → (T,) float32 numpy array


DAMAGE_CLASSES = {
    'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3
}
CLIMATE_VARIABLE_NAMES = [
    '2m_temperature', 'total_precipitation',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
    'volumetric_soil_water_layer_1', 'surface_solar_radiation_downwards',
    'surface_latent_heat_flux', 'leaf_area_index_high_vegetation',
    'surface_pressure', 'potential_evaporation',
]


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class ContinuousLocalityBatchSampler(Sampler):
    """
    Infinite locality-aware batch sampler.
    Groups patches from the same source image so the OS page-cache for
    per-disaster memmaps stays warm.  O(1) deque pops throughout.
    """

    def __init__(self, dataset, batch_size=32, patches_per_image=8, shuffle=True):
        self.batch_size        = batch_size
        self.patches_per_image = patches_per_image
        self.shuffle           = shuffle

        # Group global indices by their source image path
        self.img_to_indices = defaultdict(list)
        for global_idx in range(len(dataset)):
            img_path = dataset.img_pre_paths[global_idx]
            self.img_to_indices[img_path].append(global_idx)
        self.img_paths = list(self.img_to_indices.keys())

    def _fresh_pools(self):
        pools = {}
        for img, idxs in self.img_to_indices.items():
            lst = list(idxs)
            if self.shuffle:
                random.shuffle(lst)
            pools[img] = deque(lst)
        return pools

    def __iter__(self):
        pools         = self._fresh_pools()
        img_list      = list(self.img_paths)
        if self.shuffle:
            random.shuffle(img_list)
        active        = deque(img_list)
        batch         = []

        while True:
            if not active:
                pools    = self._fresh_pools()
                img_list = list(self.img_paths)
                if self.shuffle:
                    random.shuffle(img_list)
                active   = deque(img_list)

            img  = active.popleft()
            pool = pools[img]
            for _ in range(min(self.patches_per_image, len(pool))):
                batch.append(pool.popleft())

            if pool:
                if self.shuffle:
                    k = random.randint(0, len(active))
                    active.rotate(-k)
                    active.appendleft(img)
                    active.rotate(k)
                else:
                    active.append(img)

            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]

    def __len__(self):
        return int(1e12)


# ---------------------------------------------------------------------------
# Per-disaster shard
# ---------------------------------------------------------------------------

class _DisasterShard:
    """
    Holds open mmap handles and in-RAM climate arrays for one disaster.
    Opened lazily on first access so unused disasters cost nothing.
    """
    
    def __init__(self, shard_dir: str, patch_size: int, max_T: int):
        self._dir        = shard_dir
        self._ready      = False
        self._patch_size = patch_size
        self._max_T      = max_T

        # Lightweight metadata loaded eagerly (tiny arrays)
        meta = np.load(os.path.join(shard_dir, 'metadata.npz'), allow_pickle=True)
        self.label_pre     = meta['label_pre'].astype(np.int64)
        self.label_post    = meta['label_post'].astype(np.int64)
        self.centers       = meta['centers']
        self.pre_dates     = meta['pre_dates']
        self.post_dates    = meta['post_dates']
        self.img_pre_paths   = meta['img_pre_paths'].tolist()
        self.img_post_paths  = meta['img_post_paths'].tolist() \
                               if 'img_post_paths' in meta \
                               else [''] * int(meta['n_patches'])
        self.lbl_post_paths  = meta['lbl_post_paths'].tolist()
        # One representative climate file path per patch in this disaster
        _cp = meta['clim_file_paths'].tolist() if 'clim_file_paths' in meta else ['']
        self.clim_file_paths = (_cp * int(meta['n_patches']))[:int(meta['n_patches'])] \
                               if len(_cp) == 1 else _cp
        self.pre_dates       = meta['pre_dates']
        self.post_dates      = meta['post_dates']
        self.n_patches       = int(meta['n_patches'])

        # Heavy arrays initialised to None; loaded on first __getitem__ call
        self._mm_pre   = None
        self._mm_post  = None
        self._mm_mask  = None
        self._climate  = None
        self._ev_labels= None

    def _ensure_loaded(self):
        if self._ready:
            return
        N = self.n_patches
        P = self._patch_size
        
        # --- REVERTED: Remove np.array() to prevent OOM hanging ---
        self._mm_pre  = np.memmap(os.path.join(self._dir, 'patches_pre.npy'),
                                  dtype='float32', mode='r', shape=(N, 3, P, P))
        self._mm_post = np.memmap(os.path.join(self._dir, 'patches_post.npy'),
                                  dtype='float32', mode='r', shape=(N, 3, P, P))
        self._mm_mask = np.memmap(os.path.join(self._dir, 'masks.npy'),
                                  dtype='uint8',   mode='r', shape=(N, P, P))
        
        # Populate module-level cache (no-op if already loaded by this process).
        # In the main process this runs at dataset.__init__ time, before
        # DataLoader forks workers — so workers inherit fully-populated dicts
        # via copy-on-write without re-reading any files.
        disaster = os.path.basename(self._dir)
        if disaster not in _CLIMATE_CACHE:
            clim_raw = np.load(os.path.join(self._dir, 'climate.npy')).astype(np.float32)
            ev_raw   = np.load(os.path.join(self._dir, 'event_labels.npy')).astype(np.float32)
            C, T, H, W = clim_raw.shape
            if T < self._max_T:
                clim_pad = np.zeros((C, self._max_T, H, W), dtype=np.float32)
                clim_pad[:, :T] = clim_raw
                ev_pad = np.zeros(self._max_T, dtype=np.float32)
                ev_pad[:T] = ev_raw
            else:
                clim_pad = clim_raw
                ev_pad   = ev_raw
            _CLIMATE_CACHE[disaster] = clim_pad   # (C, max_T, H, W) float32
            _EV_CACHE[disaster]      = ev_pad      # (max_T,) float32
        self._disaster   = disaster
        self._ready = True

    def get(self, local_idx: int):
        """Return (pre, post, mask, climate, ev_labels, label_pre, label_post)."""
        self._ensure_loaded()
        
        # It is okay to .copy() the small 64x64 image patches to give PyTorch owned storage
        pre  = self._mm_pre[local_idx].copy()
        post = self._mm_post[local_idx].copy()
        mask = self._mm_mask[local_idx].copy()
        
        # Return VIEWS into the module-level cache — zero copy.
        # The collate function detects same-object and uses expand().
        clim = _CLIMATE_CACHE[self._disaster]   # (C, max_T, H, W) — shared
        ev   = _EV_CACHE     [self._disaster]   # (max_T,) — shared
        return (pre, post, mask, clim, ev,
                int(self.label_pre[local_idx]),
                int(self.label_post[local_idx]))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class xBDClimateDataset(Dataset):
    """
    Parameters
    ----------
    cache_dir : str
        Directory produced by preprocess_xbd.py for this split.
    augment : bool
        Apply random flips / rotations / colour jitter.
    """

    def __init__(self, cache_dir: str, augment: bool = False):
        self.cache_dir = cache_dir
        self.augment   = augment

        # Global index
        meta = np.load(os.path.join(cache_dir, 'metadata.npz'), allow_pickle=True)
        self._global_disaster  = meta['disaster']             # (N,) str
        self._global_local_idx = meta['local_idx']            # (N,) int32
        self.label_post        = meta['label_post']           # (N,) for sampler
        self.patch_mean        = meta['patch_mean'].astype(np.float32)
        self.patch_std         = meta['patch_std'].astype(np.float32)
        self.climate_mean      = meta['climate_mean'].astype(np.float32)
        self.climate_std       = meta['climate_std'].astype(np.float32)

        patch_size = int(meta['patch_size'])
        max_T      = int(meta['max_T']) if 'max_T' in meta else self._infer_max_T(cache_dir)

        # Load per-disaster shards (metadata only — arrays lazily loaded)
        disasters_dir = os.path.join(cache_dir, 'disasters')
        self._shards: dict[str, _DisasterShard] = {}
        for name in os.listdir(disasters_dir):
            d = os.path.join(disasters_dir, name)
            if os.path.isdir(d):
                self._shards[name] = _DisasterShard(d, patch_size=patch_size, max_T=max_T)

        # Eagerly load all climate data into the module-level cache NOW,
        # before DataLoader forks workers.  Each shard's _ensure_loaded()
        # populates _CLIMATE_CACHE and _EV_CACHE; forked workers inherit
        # the fully-populated dicts via OS copy-on-write.
        for shard in self._shards.values():
            shard._ensure_loaded()

        # Build img_pre_paths for the locality sampler (one entry per global idx)
        self.img_pre_paths = []
        for i in range(len(self._global_disaster)):
            disaster   = str(self._global_disaster[i])
            local_idx  = int(self._global_local_idx[i])
            self.img_pre_paths.append(
                self._shards[disaster].img_pre_paths[local_idx])

        # ── Attributes required by model.py ─────────────────────────────────
        self.damage_classes         = DAMAGE_CLASSES
        self.climate_variable_names = CLIMATE_VARIABLE_NAMES
        # Flat per-patch arrays for interpretation() filtering in model.py
        self.label_pre       = self._build_flat('label_pre')
        self.label_post_path = self._build_flat_list('lbl_post_paths')
        self.patch_image_pre = self._build_flat_list('img_pre_paths')
        self.patch_image_post= self._build_flat_list('img_post_paths')
        self.patch_pre_date  = self._build_flat('pre_dates')
        self.patch_post_date = self._build_flat('post_dates')
        self.patch_climate   = self._build_flat_list('clim_file_paths')
        # patch_pre/patch_post not held in RAM (too large); model.py only
        # reassigns these to filtered subsets before iterating via DataLoader
        self.patch_pre  = None
        self.patch_post = None

        self._print_class_distribution()

    # ---------------------------------------------------------------- augment

    def _augment(self, pre, post, mask):
        """
        pre/post : (3, P, P) float32 CHW
        mask     : (P, P) uint8

        cv2.warpAffine (0.08 ms/patch, 70% fire rate → 3.4 ms/batch) replaced
        with np.rot90 on CHW axes (0.006 ms/patch → 0.24 ms/batch, 14× faster).
        All ops stay in CHW — no HWC transpose needed.
        """
        # Discrete 90°/180°/270° rotation (50% probability)
        if random.random() > 0.5:
            k = random.randint(1, 3)
            pre  = np.rot90(pre,  k, axes=(1, 2))   # CHW: rotate H,W
            post = np.rot90(post, k, axes=(1, 2))
            mask = np.rot90(mask, k, axes=(0, 1))   # HW: rotate H,W
        # Horizontal flip
        if random.random() > 0.5:
            pre  = pre [:, :, ::-1]
            post = post[:, :, ::-1]
            mask = mask[:,    ::-1]
        # Vertical flip
        if random.random() > 0.5:
            pre  = pre [:, ::-1]
            post = post[:, ::-1]
            mask = mask[   ::-1]
        # Brightness × contrast
        if random.random() > 0.4:
            scale = random.uniform(0.7, 1.3) * random.uniform(0.8, 1.2)
            pre  = pre  * scale
            post = post * scale
        # Additive Gaussian noise
        if random.random() > 0.6:
            noise = np.random.normal(0, 5/255.0, pre.shape).astype(np.float32)
            pre  = pre  + noise
            post = post + noise
        # Per-channel shift (vectorised, no Python loop)
        if random.random() > 0.5:
            shifts = np.random.uniform(-10/255.0, 10/255.0, (3,1,1)).astype(np.float32)
            pre  = pre  + shifts
            post = post + shifts
        pre  = np.clip(pre,  -1.0, 1.0)
        post = np.clip(post, -1.0, 1.0)
        # ascontiguousarray needed because ::-1 slices produce non-contiguous views
        return (np.ascontiguousarray(pre),
                np.ascontiguousarray(post),
                np.ascontiguousarray(mask))

    # --------------------------------------------------------------- helpers

    @staticmethod
    def _infer_max_T(cache_dir: str) -> int:
        """Fallback: scan disaster dirs to find the largest T dimension."""
        disasters_dir = os.path.join(cache_dir, 'disasters')
        max_T = 0
        for name in os.listdir(disasters_dir):
            cf = os.path.join(disasters_dir, name, 'climate.npy')
            if os.path.isfile(cf):
                arr = np.load(cf, mmap_mode='r')
                max_T = max(max_T, arr.shape[1])
        return max_T

    def _build_flat(self, attr: str) -> np.ndarray:
        """Concatenate a scalar-per-patch shard attribute in global index order."""
        out = []
        for i in range(len(self._global_disaster)):
            shard = self._shards[str(self._global_disaster[i])]
            out.append(getattr(shard, attr)[int(self._global_local_idx[i])])
        return np.array(out)

    def _build_flat_list(self, attr: str) -> np.ndarray:
        """Concatenate a list-per-patch shard attribute in global index order."""
        out = []
        for i in range(len(self._global_disaster)):
            shard = self._shards[str(self._global_disaster[i])]
            out.append(getattr(shard, attr)[int(self._global_local_idx[i])])
        return np.array(out)

    def _print_class_distribution(self):
        counts = {}
        for l in self.label_post:
            counts[int(l)] = counts.get(int(l), 0) + 1
        for i, name in enumerate(['no-damage','minor-damage','major-damage','destroyed']):
            print(f'{name}: {counts.get(i, 0)}')

    def __len__(self):
        return len(self._global_disaster)

    # ----------------------------------------------------------- __getitem__

    def __getitem__(self, idx):
        disaster  = str(self._global_disaster [idx])
        local_idx = int(self._global_local_idx[idx])

        pre, post, mask, clim, ev, lbl_pre, lbl_post = \
            self._shards[disaster].get(local_idx)

        if self.augment:
            pre, post, mask = self._augment(pre, post, mask)

        img_pre_path = self._shards[disaster].img_pre_paths[local_idx]
        
        return (
            torch.from_numpy(pre),
            torch.from_numpy(post),
            torch.from_numpy(np.ascontiguousarray(mask.astype(np.float32))),
            torch.from_numpy(clim),   # view of module-level cache
            torch.from_numpy(ev),     # view of module-level cache
            torch.tensor(lbl_pre,  dtype=torch.long),
            torch.tensor(lbl_post, dtype=torch.long),
            img_pre_path,                          # event_name (str)
            torch.tensor(idx, dtype=torch.long),  # patches_idx
        )

# ---------------------------------------------------------------------------
# Custom Zero-Copy Collate Function
# ---------------------------------------------------------------------------
def zero_copy_collate(batch):
    """
    Collate function for xBD + ERA-5 batches.

    Climate tensors come from the module-level _CLIMATE_CACHE, so all items
    from the same disaster share the same underlying numpy array.  For
    same-disaster batches (>95% with the locality sampler) we avoid the
    redundant stack by using expand() + contiguous(), which copies one
    (C,T,H,W) slab instead of B separate ones.

    Rules that must hold for correctness with PyTorch DataLoader:
      - Collate runs inside worker processes: no CUDA calls (.pin_memory()
        on a tensor requires CUDA and will raise "initialization error").
      - pin_memory=True on the DataLoader is the only safe way to pin
        tensors; it runs in a dedicated main-process thread after workers
        return the batch.
      - Non-contiguous (strided) tensors passed to that thread raise
        "more than one element refers to a single memory location".
        All tensors returned here must be contiguous.
    """
    pre      = default_collate([b[0] for b in batch])   # (B,3,P,P)
    post     = default_collate([b[1] for b in batch])   # (B,3,P,P)
    mask     = default_collate([b[2] for b in batch])   # (B,P,P)
    lbl_pre  = default_collate([b[5] for b in batch])
    lbl_post = default_collate([b[6] for b in batch])
    img_path = [b[7] for b in batch]
    p_idx    = default_collate([b[8] for b in batch])

    clim_0 = batch[0][3]   # (C,T,H,W) — view into _CLIMATE_CACHE
    ev_0   = batch[0][4]   # (T,)
    B      = len(batch)

    # Same-disaster batch: all items share data_ptr() → copy one slab.
    # expand() + contiguous() copies exactly C×T×H×W×4 bytes (one disaster),
    # vs default_collate which would copy that B times.
    if all(b[3].data_ptr() == clim_0.data_ptr() for b in batch):
        clim_batch = clim_0.unsqueeze(0).expand(B, *clim_0.shape).contiguous()
    else:
        # Mixed-disaster (rare): stack copies each unique slab once
        clim_batch = torch.stack([b[3] for b in batch])

    if all(b[4].data_ptr() == ev_0.data_ptr() for b in batch):
        ev_batch = ev_0.unsqueeze(0).expand(B, *ev_0.shape).contiguous()
    else:
        ev_batch = torch.stack([b[4] for b in batch])

    return (pre, post, mask, clim_batch, ev_batch, lbl_pre, lbl_post, img_path, p_idx)

# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def get_damage_data_loaders(
    data_dir,
    batch_size      = 32,
    patch_size      = 64,
    num_workers     = 8,   
    prefetch_factor = 4,
):
    def _ds(tag, augment):
        cache = os.path.join(data_dir, f'cache_{tag}')
        if not os.path.isdir(cache):
            raise FileNotFoundError(
                f'Cache not found: {cache}\n'
                f'Run:  python preprocess_xbd.py --data_dir {data_dir} '
                f'--patch_size {patch_size}')
        return xBDClimateDataset(cache, augment=augment)

    train_dataset = _ds('train', augment=True)
    val_dataset   = _ds('val',   augment=False)
    test_dataset  = _ds('test',  augment=False)

    if len(train_dataset) == 0:
        print('No training patches found!')
        return None, None, None, None, None, None

    base_kw = {
        'pin_memory': True,
        'collate_fn': zero_copy_collate,
    }
    
    if num_workers > 0:
        base_kw.update({
            'num_workers':        num_workers,
            'prefetch_factor':    prefetch_factor,
            'persistent_workers': True,
        })
    else:
        base_kw['num_workers'] = 0

    train_sampler = ContinuousLocalityBatchSampler(
        train_dataset, batch_size=batch_size, patches_per_image=8, shuffle=True)
    val_sampler   = ContinuousLocalityBatchSampler(
        val_dataset,   batch_size=batch_size, patches_per_image=8, shuffle=False)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, **base_kw)
    val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,   **base_kw)
    
    # Test loader doesn't use a custom sampler, but can still use the zero-copy collator
    test_loader  = DataLoader(test_dataset,
                              batch_size=batch_size, shuffle=False,
                              drop_last=False, **base_kw)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Simplified interface for Lightning module
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Filtered subset for interpretation
# ---------------------------------------------------------------------------

class FilteredSubset:
    """
    Wrapper that filters a dataset to only samples matching label_post == target_cat.
    
    Used by interpretation() method to visualize a specific damage class.
    Delegates all attribute access to the underlying dataset and filters __getitem__
    indices on the fly.
    """
    
    def __init__(self, dataset, target_cat):
        self.dataset = dataset
        self.target_cat = target_cat
        # Find all indices where label_post == target_cat
        self._filtered_indices = np.where(dataset.label_post == target_cat)[0]
        
        # Build filtered flat arrays for interpretation
        self.augment           = False   # interpretation disables augmentation
        self.damage_classes    = dataset.damage_classes
        self.climate_variable_names = dataset.climate_variable_names
        self.patch_mean        = dataset.patch_mean
        self.patch_std         = dataset.patch_std
        
        # Filtered flat arrays
        mask = (dataset.label_post == target_cat)
        self.label_pre        = dataset.label_pre[mask]
        self.label_post       = dataset.label_post[mask]
        self.label_post_path  = dataset.label_post_path[mask]
        self.patch_image_pre  = dataset.patch_image_pre[mask]
        self.patch_image_post = dataset.patch_image_post[mask]
        self.patch_pre_date   = dataset.patch_pre_date[mask]
        self.patch_post_date  = dataset.patch_post_date[mask]
        self.patch_climate    = dataset.patch_climate[mask]
        
        # patch_pre and patch_post remain None (not held in RAM)
        self.patch_pre  = None
        self.patch_post = None
    
    def __len__(self):
        return len(self._filtered_indices)
    
    def __getitem__(self, idx):
        # Map filtered idx -> original dataset idx
        real_idx = self._filtered_indices[idx]
        return self.dataset[real_idx]


# ---------------------------------------------------------------------------
# Simplified interface for Lightning module
# ---------------------------------------------------------------------------

def get_dataloaders(
    cache_dir: str,
    batch_size: int = 32,
    num_workers: int = 8,
    augment: bool = True,
    patch_size: int = 64,
):
    """
    Simplified dataloader interface for Lightning module.
    
    Parameters
    ----------
    cache_dir : str
        Path to preprocessed cache directory
    batch_size : int
        Batch size for training
    num_workers : int
        Number of dataloader workers
    augment : bool
        Enable augmentation for training set
    patch_size : int
        Image patch size (must match preprocessing)
    
    Returns
    -------
    train_loader, val_loader, test_loader : DataLoader
        PyTorch dataloaders for each split
        
    Note
    ----
    The returned dataloaders have a .dataset attribute to access
    dataset properties like damage_classes, climate_variable_names, etc.
    """
    # Call the full function which handles cache loading
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = get_damage_data_loaders(
        data_dir=cache_dir,
        batch_size=batch_size,
        patch_size=patch_size,
        num_workers=num_workers,
        prefetch_factor=4,
    )
    
    return train_loader, val_loader, test_loader
# utils/lmdb_dataset.py
# LMDB-backed dataset for CDD-11 and general image restoration benchmarks.

import os
import pickle
import lmdb
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import random
import re
from utils.factor_utils import (
    parse_factors, build_name, get_leave_one_out_name,
    factors_to_present, FACTORS, FACTOR2IDX
)


class LMDBAllWeatherDataset(Dataset):
    """LMDB dataset loader for composite-degradation image restoration.

    Supports CDD-11 (11 composite degradation configurations from 4 atomic
    types: low-light, haze, rain, snow) and generic LQ/GT LMDB datasets.

    Args:
        lmdb_path: Path to the LMDB database.
        patch_size: Random crop size during training (None = full resolution).
        is_train: Training mode (enables augmentation).
        readahead: Enable LMDB read-ahead (may improve sequential throughput).
        use_precomputed_depth: Load pre-computed depth maps if available.
        use_counterfactual_supervision: Build an index for leave-one-out
            counterfactual images y_{S\\i}.
        depth_extractor: Deprecated; not used.
    """

    def __init__(
        self,
        lmdb_path,
        patch_size=256,
        is_train=True,
        readahead=False,
        use_precomputed_depth=False,
        use_counterfactual_supervision=False,
        depth_extractor=None,
    ):
        self.lmdb_path = lmdb_path
        self.patch_size = patch_size
        self.is_train = is_train
        self.readahead = readahead
        self.use_precomputed_depth = use_precomputed_depth
        self.use_counterfactual_supervision = use_counterfactual_supervision

        if not LMDB_AVAILABLE:
            raise ImportError("lmdb not installed. Install with: pip install lmdb")
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB database not found: {lmdb_path}")

        self.env = lmdb.open(
            lmdb_path,
            readonly=True,
            lock=False,
            readahead=bool(readahead),
            meminit=False,
        )

        with self.env.begin(write=False) as txn:
            self.keys = [key.decode() for key, _ in txn.cursor()]

        print(f"Loaded LMDB dataset: {len(self.keys)} samples from {lmdb_path}")

        # Index for counterfactual supervision (scene_id, deg_name) → key
        if self.use_counterfactual_supervision:
            self.key_map = {}
            self.gt_key_map = {}
            self._build_index()
        else:
            self.key_map = None
            self.gt_key_map = None

        self.to_tensor = transforms.ToTensor()

    # ------------------------------------------------------------------
    # Counterfactual index construction
    # ------------------------------------------------------------------
    def _build_index(self):
        """Build a (scene_id, deg_name) → key mapping for leave-one-out lookups."""
        print("Building index for counterfactual supervision...")
        for key in self.keys:
            try:
                with self.env.begin(write=False) as txn:
                    data = pickle.loads(txn.get(key.encode()))
                    if isinstance(data, dict):
                        deg_name = data.get('deg_name', data.get('degradation', None))
                        scene_id = self._extract_scene_id(key, data)
                        if scene_id is not None:
                            if deg_name:
                                self.key_map[(scene_id, deg_name)] = key
                            if deg_name in [None, "clean", "gt", ""]:
                                self.gt_key_map[scene_id] = key
            except Exception:
                continue
        print(f"Index built: {len(self.key_map)} (scene, deg) pairs, "
              f"{len(self.gt_key_map)} clean images")

    def _extract_scene_id(self, key, data):
        """Heuristically extract a scene identifier from the key or data dict."""
        match = re.search(r'(\d+)(?:\.(?:png|jpg|jpeg))?$', key)
        if match:
            return match.group(1)
        if isinstance(data, dict):
            scene_id = data.get('scene_id', data.get('img_id', data.get('id', None)))
            if scene_id:
                return str(scene_id)
        return str(hash(key) % 1000000)

    # ------------------------------------------------------------------
    # Core data loading
    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        """Load a sample and return a dict with unified batch keys.

        Returns:
            dict with keys:
                LQ / y:    (3, H, W) degraded image, range [-1, 1].
                GT / x:    (3, H, W) clean image, range [-1, 1].
                deg_name:  Degradation label string (e.g. "low_haze_rain").
                present:   (4,) binary factor-presence vector.
                w:         (4,) factor weights (= present in the minimal version).
                m:         (4, H, W) spatial intensity maps (1/0 minimal).
                y_minus:   (4, 3, H, W) leave-one-out images.
                has_cf:    (4,) availability flags for y_minus.
                depth:     (1, H, W) depth map (optional).
        """
        key = self.keys[idx]

        with self.env.begin(write=False) as txn:
            data = pickle.loads(txn.get(key.encode()))

        if isinstance(data, dict):
            lq_array = data.get('LQ', data.get('input', None))
            gt_array = data.get('GT', data.get('clear', data.get('gt', None)))
            deg_name = data.get('deg_name', data.get('degradation', None))
            depth_array = data.get('depth', None)
        else:
            raise ValueError(f"Unknown data format in LMDB: {type(data)}")

        if lq_array is None:
            raise ValueError(f"LQ array is None for key: {key}")
        if gt_array is None:
            raise ValueError(f"GT array is None for key: {key}")

        scene_id = self._extract_scene_id(key, data) if self.use_counterfactual_supervision else None

        # Convert to PIL
        if isinstance(lq_array, np.ndarray):
            if lq_array.max() <= 1.0:
                lq_array = (lq_array * 255).astype(np.uint8)
            lq_img = Image.fromarray(lq_array)
        else:
            raise ValueError(f"Unknown LQ format: {type(lq_array)} for key: {key}")

        if isinstance(gt_array, np.ndarray):
            if gt_array.max() <= 1.0:
                gt_array = (gt_array * 255).astype(np.uint8)
            gt_img = Image.fromarray(gt_array)
        else:
            raise ValueError(f"Unknown GT format: {type(gt_array)} for key: {key}")

        # Training augmentation
        if self.is_train and self.patch_size is not None:
            lq_img, gt_img = self._random_crop(lq_img, gt_img)
            lq_img, gt_img = self._random_augment(lq_img, gt_img)

        # To tensor in [-1, 1]
        y = self.to_tensor(lq_img) * 2.0 - 1.0
        x = self.to_tensor(gt_img) * 2.0 - 1.0

        # Ensure LQ/GT have the same spatial size
        if y.shape != x.shape:
            min_h = min(y.shape[1], x.shape[1])
            min_w = min(y.shape[2], x.shape[2])
            y = y[:, :min_h, :min_w]
            x = x[:, :min_h, :min_w]

        C, H, W = y.shape

        # Parse degradation factors → presence vector / spatial maps
        factors = parse_factors(deg_name)
        present = factors_to_present(factors)
        w = present.clone()

        m = torch.zeros(4, H, W)
        for i in range(4):
            if present[i] > 0:
                m[i].fill_(1.0)

        # Counterfactual leave-one-out images
        y_minus = torch.zeros(4, C, H, W)
        has_cf = torch.zeros(4)

        if self.use_counterfactual_supervision and self.is_train and scene_id is not None:
            for factor in FACTORS:
                i = FACTOR2IDX[factor]
                if factor in factors:
                    factors_minus = [f for f in factors if f != factor]
                    deg_minus_name = build_name(factors_minus)

                    y_m = None
                    ok = False

                    if deg_minus_name == "clean":
                        if scene_id in self.gt_key_map:
                            try:
                                with self.env.begin(write=False) as txn:
                                    gt_data = pickle.loads(txn.get(self.gt_key_map[scene_id].encode()))
                                    if isinstance(gt_data, dict):
                                        gt_arr = gt_data.get('GT', gt_data.get('clear', gt_data.get('gt', None)))
                                        if gt_arr is not None and isinstance(gt_arr, np.ndarray):
                                            if gt_arr.max() <= 1.0:
                                                gt_arr = (gt_arr * 255).astype(np.uint8)
                                            if self.is_train and self.patch_size is not None:
                                                y_m = x.clone()
                                            else:
                                                y_m = self.to_tensor(Image.fromarray(gt_arr)) * 2.0 - 1.0
                                            ok = True
                            except Exception:
                                pass
                    else:
                        if (scene_id, deg_minus_name) in self.key_map:
                            try:
                                with self.env.begin(write=False) as txn:
                                    minus_data = pickle.loads(txn.get(self.key_map[(scene_id, deg_minus_name)].encode()))
                                    if isinstance(minus_data, dict):
                                        minus_arr = minus_data.get('LQ', minus_data.get('input', None))
                                        if minus_arr is not None and isinstance(minus_arr, np.ndarray):
                                            if minus_arr.max() <= 1.0:
                                                minus_arr = (minus_arr * 255).astype(np.uint8)
                                            minus_img = Image.fromarray(minus_arr)
                                            if self.is_train and self.patch_size is not None:
                                                minus_img, _ = self._random_crop(minus_img, gt_img)
                                                minus_img, _ = self._random_augment(minus_img, gt_img)
                                            y_m = self.to_tensor(minus_img) * 2.0 - 1.0
                                            if y_m.shape != y.shape:
                                                y_m = torch.nn.functional.interpolate(
                                                    y_m.unsqueeze(0), size=y.shape[1:],
                                                    mode='bilinear', align_corners=False
                                                ).squeeze(0)
                                            ok = True
                            except Exception:
                                pass

                    if ok and y_m is not None:
                        y_minus[i] = y_m
                        has_cf[i] = 1.0
                    else:
                        y_minus[i] = y
                        has_cf[i] = 0.0
                else:
                    y_minus[i] = y
                    has_cf[i] = 0.0

        if deg_name is None:
            deg_name = ""

        result = {
            'x': x,
            'y': y,
            'deg_name': deg_name,
            'present': present,
            'w': w,
            'm': m,
            'y_minus': y_minus,
            'has_cf': has_cf,
            'LQ': y,
            'GT': x,
        }

        if self.use_precomputed_depth and depth_array is not None:
            if isinstance(depth_array, np.ndarray):
                depth_tensor = torch.from_numpy(depth_array).float()
                if depth_tensor.dim() == 2:
                    depth_tensor = depth_tensor.unsqueeze(0)
                if depth_tensor.shape[1:] != y.shape[1:]:
                    depth_tensor = torch.nn.functional.interpolate(
                        depth_tensor.unsqueeze(0),
                        size=y.shape[1:],
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0)
                result['depth'] = depth_tensor

        return result

    # ------------------------------------------------------------------
    # Augmentation helpers
    # ------------------------------------------------------------------
    def _random_crop(self, lq_img, gt_img):
        """Synchronised random crop of LQ and GT to ``patch_size``."""
        w, h = lq_img.size
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            lq_img = lq_img.resize((new_w, new_h), Image.BILINEAR)
            gt_img = gt_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h

        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        lq_img = TF.crop(lq_img, i, j, self.patch_size, self.patch_size)
        gt_img = TF.crop(gt_img, i, j, self.patch_size, self.patch_size)
        return lq_img, gt_img

    def _random_augment(self, lq_img, gt_img):
        """Random horizontal/vertical flip (no 90°/270° rotation to avoid H/W swap)."""
        if random.random() > 0.5:
            lq_img = TF.hflip(lq_img)
            gt_img = TF.hflip(gt_img)
        if random.random() > 0.5:
            lq_img = TF.vflip(lq_img)
            gt_img = TF.vflip(gt_img)
        return lq_img, gt_img

    def __del__(self):
        if hasattr(self, 'env'):
            self.env.close()


# LMDB availability check
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    print("Warning: lmdb not installed. Install with: pip install lmdb")


# ============================================================================
# Image-folder fallback dataset
# ============================================================================
class ImageFolderDataset(Dataset):
    """Folder-based dataset (fallback when LMDB is unavailable).

    Expected directory layout::

        root_dir/
            input/   # degraded images
            clear/   # corresponding clean images

    CDD-11 naming convention: ``<degradation>_<id>.png``
    (e.g. ``low_haze_rain_001.png``).
    """

    def __init__(
        self,
        root_dir,
        patch_size=256,
        is_train=True,
        use_precomputed_depth=False,
    ):
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'input')
        self.clear_dir = os.path.join(root_dir, 'clear')
        self.patch_size = patch_size
        self.is_train = is_train
        self.use_precomputed_depth = use_precomputed_depth

        self.samples = self._build_samples()
        self.to_tensor = transforms.ToTensor()

    def _build_samples(self):
        samples = []
        if not os.path.exists(self.input_dir):
            return samples

        input_files = [f for f in os.listdir(self.input_dir)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        for input_file in input_files:
            input_path = os.path.join(self.input_dir, input_file)
            name_without_ext = os.path.splitext(input_file)[0]

            # Try exact filename match
            clear_path = None
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.clear_dir, name_without_ext + ext)
                if os.path.exists(candidate):
                    clear_path = candidate
                    break

            # Try numeric-ID match (CDD-11: degradation_id.png)
            if clear_path is None:
                match = re.search(r'(\d+)\.(png|jpg|jpeg)$', input_file, re.IGNORECASE)
                if match:
                    img_id = match.group(1)
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate = os.path.join(self.clear_dir, f"{img_id}{ext}")
                        if os.path.exists(candidate):
                            clear_path = candidate
                            break

            if clear_path and os.path.exists(clear_path):
                deg_name = self._extract_degradation_name(input_file)
                samples.append((input_path, clear_path, deg_name))

        return samples

    def _extract_degradation_name(self, filename):
        """Extract degradation label from filename (e.g. low_haze_rain_001.png)."""
        match = re.match(r'^(.+)_\d+\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, clear_path, deg_name = self.samples[idx]

        lq_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(clear_path).convert('RGB')

        if self.is_train and self.patch_size is not None:
            lq_img, gt_img = self._random_crop(lq_img, gt_img)
            lq_img, gt_img = self._random_augment(lq_img, gt_img)

        lq_tensor = self.to_tensor(lq_img) * 2.0 - 1.0
        gt_tensor = self.to_tensor(gt_img) * 2.0 - 1.0

        result = {
            'LQ': lq_tensor,
            'GT': gt_tensor,
            'deg_name': deg_name,
        }

        if self.use_precomputed_depth:
            depth_path = input_path.replace('/input/', '/depth/').replace('.png', '_depth.npy')
            if os.path.exists(depth_path):
                depth_array = np.load(depth_path)
                depth_tensor = torch.from_numpy(depth_array).float()
                if depth_tensor.dim() == 2:
                    depth_tensor = depth_tensor.unsqueeze(0)
                result['depth'] = depth_tensor
            else:
                result['depth'] = None

        return result

    def _random_crop(self, lq_img, gt_img):
        w, h = lq_img.size
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h)
            new_w, new_h = int(w * scale), int(h * scale)
            lq_img = lq_img.resize((new_w, new_h), Image.BILINEAR)
            gt_img = gt_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        lq_img = TF.crop(lq_img, i, j, self.patch_size, self.patch_size)
        gt_img = TF.crop(gt_img, i, j, self.patch_size, self.patch_size)
        return lq_img, gt_img

    def _random_augment(self, lq_img, gt_img):
        if random.random() > 0.5:
            lq_img = TF.hflip(lq_img)
            gt_img = TF.hflip(gt_img)
        if random.random() > 0.5:
            lq_img = TF.vflip(lq_img)
            gt_img = TF.vflip(gt_img)
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])
            lq_img = TF.rotate(lq_img, angle)
            gt_img = TF.rotate(gt_img, angle)
        return lq_img, gt_img

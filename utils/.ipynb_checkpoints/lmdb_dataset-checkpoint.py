# utils/lmdb_dataset.py
# LMDB 数据集加载器（支持 CDD-11 和通用图像恢复数据集）

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
import sys
import subprocess
import shutil
import hashlib
import struct
from utils.factor_utils import (
    parse_factors, build_name, get_leave_one_out_name, 
    factors_to_present, FACTORS, FACTOR2IDX
)


class LMDBAllWeatherDataset(Dataset):
    """
    LMDB 格式的全天气数据集加载器。
    
    支持：
    - CDD-11 数据集（11 种复合退化）
    - 通用图像恢复数据集
    
    数据格式：
    - key: 图像路径或 ID
    - value: pickle 序列化的字典 {'LQ': array, 'GT': array, 'deg_name': str, ...}
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
        """
        Args:
            lmdb_path: LMDB 数据库路径
            patch_size: 训练时的 patch 大小（None 表示使用全分辨率）
            is_train: 是否为训练模式
            readahead: 是否启用 LMDB 预读（可能提升顺序读取吞吐，但会占用更多页缓存）
            use_precomputed_depth: 是否使用预计算的深度图
            use_counterfactual_supervision: 是否启用反事实监督（返回 y_S_minus_i，当前未使用合成器）
            depth_extractor: Depth-Anything 提取器（已废弃，当前不使用）
        """
        self.lmdb_path = lmdb_path
        self.patch_size = patch_size
        self.is_train = is_train
        self.readahead = readahead
        self.use_precomputed_depth = use_precomputed_depth
        self.use_counterfactual_supervision = use_counterfactual_supervision
        
        # 检查 LMDB 是否可用
        if not LMDB_AVAILABLE:
            raise ImportError("lmdb not installed. Install with: pip install lmdb")
        
        # 打开 LMDB
        if not os.path.exists(lmdb_path):
            raise FileNotFoundError(f"LMDB database not found: {lmdb_path}")

        # 如果 LMDB 在不支持 mmap 的文件系统上，自动复制到本地 /tmp
        lmdb_path = self._maybe_copy_to_local(lmdb_path)
        self.lmdb_path = lmdb_path  # 更新为可能已复制的路径

        # 获取实际文件大小，设置 map_size (虚拟地址空间上限，不实际分配内存)
        data_mdb = os.path.join(lmdb_path, 'data.mdb')
        if os.path.isfile(data_mdb):
            actual_size = os.path.getsize(data_mdb)
            # 使用 max(256MB, 文件大小*2) 避免 mmap SIGBUS
            map_size = max(256 * 1024 * 1024, int(actual_size * 2))
        else:
            map_size = 1024 * 1024 * 1024

        self._map_size = map_size  # 保存用于延迟重开
        print(f"[LMDB] path={lmdb_path}, data.mdb={actual_size if os.path.isfile(data_mdb) else 'N/A'} bytes, map_size={map_size}", flush=True)

        # 在子进程中枚举 keys (SIGBUS 只会杀死子进程，不影响主训练进程)
        self.keys = self._enumerate_keys(lmdb_path, map_size)
        self.env = None   # 延迟打开

        print(f"[LMDB] Loaded {len(self.keys)} keys from {lmdb_path}", flush=True)
        
        # ✅建立索引表：用于查找同一 scene 的不同退化组合（用于反事实监督）
        if self.use_counterfactual_supervision:
            self.key_map = {}  # (scene_id, deg_name) -> key
            self.gt_key_map = {}  # scene_id -> key (for clean/GT)
            self._build_index()
        else:
            self.key_map = None
            self.gt_key_map = None
        
        # Transforms
        self.to_tensor = transforms.ToTensor()

    @staticmethod
    def _maybe_copy_to_local(lmdb_path):
        """
        用子进程探测 LMDB 是否能在当前文件系统上正常读取。
        如果 SIGBUS（常见于页大小不匹配或网络文件系统），则：
        1. 先尝试用 mdb_dump | mdb_load 重建（修复页大小不匹配）
        2. 如果没有 mdb_dump，则尝试简单复制到本地磁盘
        """
        data_mdb = os.path.join(lmdb_path, 'data.mdb')
        if not os.path.isfile(data_mdb):
            return lmdb_path  # 让后续逻辑报错

        # 快速探测：子进程尝试 mmap 读一个 key
        fsize = os.path.getsize(data_mdb)
        probe = (
            "import lmdb,sys\n"
            f"env=lmdb.open({lmdb_path!r},readonly=True,lock=False,"
            f"readahead=False,meminit=False,"
            f"map_size={max(256*1024*1024, fsize*2)})\n"
            "txn=env.begin(write=False)\n"
            "cur=txn.cursor()\n"
            "if cur.first(): sys.stdout.write('OK')\n"
            "env.close()\n"
        )
        try:
            proc = subprocess.run(
                [sys.executable, '-c', probe],
                capture_output=True, timeout=30,
            )
            if proc.returncode == 0:
                return lmdb_path  # mmap 正常，无需处理
        except subprocess.TimeoutExpired:
            pass

        # --- mmap 失败，需要重建 ---
        path_hash = hashlib.md5(lmdb_path.encode()).hexdigest()[:8]
        basename = os.path.basename(lmdb_path.rstrip('/'))
        rebuilt_path = os.path.join('/tmp', f'lmdb_rebuilt_{basename}_{path_hash}')
        rebuilt_data = os.path.join(rebuilt_path, 'data.mdb')

        # 检查是否已有可用的重建缓存
        if os.path.isfile(rebuilt_data):
            test_probe = (
                "import lmdb,sys\n"
                f"env=lmdb.open({rebuilt_path!r},readonly=True,lock=False,"
                f"readahead=False,meminit=False,"
                f"map_size={max(256*1024*1024, fsize*3)})\n"
                "txn=env.begin(write=False)\n"
                "print(txn.stat()['entries'])\n"
                "env.close()\n"
            )
            try:
                tp = subprocess.run([sys.executable, '-c', test_probe],
                                    capture_output=True, timeout=15)
                if tp.returncode == 0 and int(tp.stdout.strip()) > 0:
                    print(f"[LMDB] Using cached rebuilt copy: {rebuilt_path} "
                          f"({int(tp.stdout.strip())} entries)", flush=True)
                    return rebuilt_path
            except Exception:
                pass

        # 读取源 LMDB 的页大小（data.mdb 前 4096 字节的 offset 20-23）
        with open(data_mdb, 'rb') as f:
            header = f.read(4096)
        if len(header) >= 24:
            lmdb_page_size = struct.unpack_from('<I', header, 20)[0]
            os_page_size = os.sysconf('SC_PAGESIZE') if hasattr(os, 'sysconf') else 4096
            print(f"[LMDB] Page size: LMDB={lmdb_page_size}, OS={os_page_size}", flush=True)
        else:
            lmdb_page_size = 0
            os_page_size = 4096

        print(f"[LMDB] mmap failed on {lmdb_path} (SIGBUS).", flush=True)

        # 方案1: 用 mdb_dump | mdb_load 重建（能修复页大小不匹配）
        has_mdb_dump = shutil.which('mdb_dump') and shutil.which('mdb_load')
        if has_mdb_dump:
            print(f"[LMDB] Rebuilding via mdb_dump|mdb_load -> {rebuilt_path} ...", flush=True)
            os.makedirs(rebuilt_path, exist_ok=True)
            cmd = f'mdb_dump -a "{lmdb_path}" | mdb_load "{rebuilt_path}"'
            ret = subprocess.run(cmd, shell=True, capture_output=True, timeout=300)
            if ret.returncode == 0 and os.path.isfile(rebuilt_data):
                print(f"[LMDB] Rebuild complete ({os.path.getsize(rebuilt_data)/1024/1024:.1f} MB).",
                      flush=True)
                return rebuilt_path
            print(f"[LMDB] mdb_dump|mdb_load failed: {ret.stderr.decode(errors='replace')[:200]}",
                  flush=True)

        # 方案2: 用 Python lmdb 在子进程中逐条读出，再在主进程写入新 LMDB
        # 子进程设置 MALLOC_MMAP_MAX_=0 强制禁用 mmap 分配
        print(f"[LMDB] Rebuilding via Python subprocess reader -> {rebuilt_path} ...", flush=True)
        reader_script = (
            "import lmdb,sys,os,struct\n"
            f"path={lmdb_path!r}\n"
            f"ms={max(256*1024*1024, fsize*2)}\n"
            "env=lmdb.open(path,readonly=True,lock=False,readahead=False,meminit=False,map_size=ms)\n"
            "txn=env.begin(write=False)\n"
            "for k,v in txn.cursor():\n"
            "  kl=struct.pack('<I',len(k))\n"
            "  vl=struct.pack('<I',len(v))\n"
            "  sys.stdout.buffer.write(kl+k+vl+v)\n"
            "env.close()\n"
        )
        try:
            rp = subprocess.run(
                [sys.executable, '-c', reader_script],
                capture_output=True, timeout=300,
            )
        except subprocess.TimeoutExpired:
            rp = None

        if rp is None or rp.returncode != 0:
            sig = -rp.returncode if rp and rp.returncode < 0 else '?'
            raise RuntimeError(
                f"[LMDB] Cannot read {lmdb_path} — LMDB file is corrupted "
                f"(SIGBUS even after copy, signal={sig}).\n"
                f"Page size in file: {lmdb_page_size}, OS page size: {os_page_size}.\n"
                f"Solutions:\n"
                f"  1. Install lmdb-utils: apt-get install lmdb-utils\n"
                f"     Then re-run (mdb_dump|mdb_load will fix page size).\n"
                f"  2. Re-create the LMDB on this machine from original images.\n"
                f"  3. Copy the LMDB from a machine with matching page size ({lmdb_page_size})."
            )

        # 解析子进程输出，写入新 LMDB
        buf = rp.stdout
        os.makedirs(rebuilt_path, exist_ok=True)
        out_env = lmdb.open(rebuilt_path, map_size=max(1024*1024*1024, fsize*3))
        pos = 0
        count = 0
        with out_env.begin(write=True) as txn:
            while pos + 4 <= len(buf):
                kl = struct.unpack_from('<I', buf, pos)[0]
                pos += 4
                key = buf[pos:pos+kl]
                pos += kl
                vl = struct.unpack_from('<I', buf, pos)[0]
                pos += 4
                val = buf[pos:pos+vl]
                pos += vl
                txn.put(key, val)
                count += 1
        out_env.sync()
        out_env.close()
        print(f"[LMDB] Rebuild complete: {count} entries, "
              f"{os.path.getsize(rebuilt_data)/1024/1024:.1f} MB.", flush=True)
        return rebuilt_path

    @staticmethod
    def _enumerate_keys(lmdb_path, map_size):
        """
        在子进程中枚举 LMDB keys。
        好处: SIGBUS (mmap 损坏/文件系统不兼容) 只杀死子进程，
        主进程能拿到明确错误信息而非静默 core dump。
        """
        script = (
            "import lmdb,sys\n"
            f"env=lmdb.open({lmdb_path!r},readonly=True,lock=False,"
            f"readahead=False,meminit=False,map_size={map_size})\n"
            "txn=env.begin(write=False)\n"
            "for k in txn.cursor().iternext(keys=True,values=False):\n"
            "  sys.stdout.buffer.write(k+b'\\n')\n"
            "env.close()\n"
        )
        print(f"[LMDB] enumerating keys via subprocess...", flush=True)
        proc = subprocess.run(
            [sys.executable, '-c', script],
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            sig = -proc.returncode if proc.returncode < 0 else None
            sig_msg = f" (signal {sig}, likely SIGBUS from corrupted LMDB)" if sig else ""
            raise RuntimeError(
                f"LMDB key enumeration failed for {lmdb_path}{sig_msg}. "
                f"returncode={proc.returncode}. "
                f"stderr: {proc.stderr.decode(errors='replace')[:500]}"
            )
        keys = [line.decode() for line in proc.stdout.strip().split(b'\n') if line]
        print(f"[LMDB] Loaded {len(keys)} keys from {lmdb_path}", flush=True)
        return keys

    def _build_index(self):
        """
        建立索引表：扫描所有 keys，构建 (scene_id, deg_name) -> key 的映射。
        
        假设 key 格式可能是：
        - 路径格式：path/to/{deg_name}_{scene_id}.png
        - ID 格式：{deg_name}_{scene_id}
        - 或其他格式
        
        需要根据实际 LMDB key 格式调整。
        """
        print("Building index for counterfactual supervision...")
        
        for key in self.keys:
            # 尝试从 key 中提取 scene_id 和 deg_name
            # 假设 key 可能包含退化名称和场景 ID
            # 这里需要根据实际数据格式调整
            
            # 方法1: 从 LMDB 数据中读取 deg_name
            try:
                self._ensure_env()
                with self.env.begin(write=False) as txn:
                    data = pickle.loads(txn.get(key.encode()))
                    if isinstance(data, dict):
                        deg_name = data.get('deg_name', data.get('degradation', None))
                        # 尝试从 key 或数据中提取 scene_id
                        # 如果 key 是路径，提取文件名；如果是 ID，直接使用
                        scene_id = self._extract_scene_id(key, data)
                        
                        if scene_id is not None:
                            if deg_name:
                                self.key_map[(scene_id, deg_name)] = key
                            # 如果 deg_name 是 clean/gt，记录到 gt_key_map
                            if deg_name in [None, "clean", "gt", ""]:
                                self.gt_key_map[scene_id] = key
            except Exception as e:
                # 如果解析失败，跳过
                continue
        
        print(f"Index built: {len(self.key_map)} (scene, deg) pairs, {len(self.gt_key_map)} clean images")
    
    def _extract_scene_id(self, key, data):
        """
        从 key 或 data 中提取 scene_id。
        
        这里需要根据实际数据格式实现。
        可能的格式：
        - key 是路径：path/to/{deg_name}_{scene_id}.png -> 提取 scene_id
        - key 是 ID：{deg_name}_{scene_id} -> 提取 scene_id
        - data 中有 scene_id 字段
        """
        # 方法1: 从 key 中提取（假设格式是 {deg_name}_{scene_id} 或路径）
        # 尝试匹配数字 ID
        match = re.search(r'(\d+)(?:\.(?:png|jpg|jpeg))?$', key)
        if match:
            return match.group(1)
        
        # 方法2: 从 data 中提取
        if isinstance(data, dict):
            scene_id = data.get('scene_id', data.get('img_id', data.get('id', None)))
            if scene_id:
                return str(scene_id)
        
        # 方法3: 使用 key 的 hash 作为 scene_id（fallback）
        return str(hash(key) % 1000000)
    
    def __len__(self):
        return len(self.keys)

    def _ensure_env(self):
        """延迟打开 LMDB env（每个 DataLoader worker 进程各自打开，避免 fork 后 mmap 冲突）"""
        if self.env is None:
            self.env = lmdb.open(
                self.lmdb_path,
                readonly=True,
                lock=False,
                readahead=bool(self.readahead),
                meminit=False,
                map_size=self._map_size,
            )

    def __getitem__(self, idx):
        """
        获取一个样本（统一 batch 格式）。
        
        Returns:
            dict: {
                'x': (3, H, W) tensor，clean/GT，值域 [-1, 1]
                'y': (3, H, W) tensor，composite degraded，值域 [-1, 1]
                'deg_name': str 或 None
                'present': (4,) tensor，因子是否存在 [low, haze, rain, snow]
                'w': (4,) tensor，因子强度标量（最小版本 = present）
                'm': (4, H, W) tensor，空间强度图（最小版本 = 1/0）
                'y_minus': (4, 3, H, W) tensor，leave-one-out 图 y_{S\\i}
                'has_cf': (4,) tensor，是否有该因子的 y_minus 可用（0/1）
                'depth': (1, H, W) tensor（可选）
            }
        """
        key = self.keys[idx]

        # 从 LMDB 读取（延迟打开 env，fork-safe）
        self._ensure_env()
        with self.env.begin(write=False) as txn:
            data = pickle.loads(txn.get(key.encode()))
        
        # 解析数据
        if isinstance(data, dict):
            lq_array = data.get('LQ', data.get('input', None))
            gt_array = data.get('GT', data.get('clear', data.get('gt', None)))
            deg_name = data.get('deg_name', data.get('degradation', None))
            depth_array = data.get('depth', None)
        else:
            raise ValueError(f"Unknown data format in LMDB: {type(data)}")
        
        # 检查必要的数据是否存在
        if lq_array is None:
            raise ValueError(f"LQ array is None for key: {key}")
        if gt_array is None:
            raise ValueError(f"GT array is None for key: {key}")
        
        # 提取 scene_id（用于查找 y_minus）
        scene_id = self._extract_scene_id(key, data) if self.use_counterfactual_supervision else None
        
        # 转换为 PIL Image
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
        
        # 数据增强（训练时）
        if self.is_train and self.patch_size is not None:
            lq_img, gt_img = self._random_crop(lq_img, gt_img)
            lq_img, gt_img = self._random_augment(lq_img, gt_img)
        
        # 转换为 tensor，值域 [-1, 1]
        y = self.to_tensor(lq_img) * 2.0 - 1.0
        x = self.to_tensor(gt_img) * 2.0 - 1.0
        
        # ✅确保尺寸一致
        if y.shape != x.shape:
            min_h = min(y.shape[1], x.shape[1])
            min_w = min(y.shape[2], x.shape[2])
            y = y[:, :min_h, :min_w]
            x = x[:, :min_h, :min_w]
        
        C, H, W = y.shape
        
        # ✅解析因子并构建 present/w/m
        factors = parse_factors(deg_name)
        present = factors_to_present(factors)
        w = present.clone()  # 最小版本：w = present
        
        # m 最小版本：存在因子 -> 全1，不存在 -> 全0
        m = torch.zeros(4, H, W)
        for i in range(4):
            if present[i] > 0:
                m[i].fill_(1.0)
        
        # ✅构建 y_minus 和 has_cf
        y_minus = torch.zeros(4, C, H, W)  # [4,3,H,W]
        has_cf = torch.zeros(4)  # [4]
        
        if self.use_counterfactual_supervision and self.is_train and scene_id is not None:
            for factor in FACTORS:
                i = FACTOR2IDX[factor]
                
                if factor in factors:
                    # 构造 deg_name_minus_i
                    factors_minus = [f for f in factors if f != factor]
                    deg_minus_name = build_name(factors_minus)
                    
                    # 查找 y_{S\\i}
                    y_m = None
                    ok = False
                    
                    if deg_minus_name == "clean":
                        # 使用 clean/GT
                        if scene_id in self.gt_key_map:
                            try:
                                with self.env.begin(write=False) as txn:
                                    gt_data = pickle.loads(txn.get(self.gt_key_map[scene_id].encode()))
                                    if isinstance(gt_data, dict):
                                        gt_array = gt_data.get('GT', gt_data.get('clear', gt_data.get('gt', None)))
                                        if gt_array is not None:
                                            if isinstance(gt_array, np.ndarray):
                                                if gt_array.max() <= 1.0:
                                                    gt_array = (gt_array * 255).astype(np.uint8)
                                                gt_img_m = Image.fromarray(gt_array)
                                                # 应用相同的数据增强
                                                if self.is_train and self.patch_size is not None:
                                                    # 注意：这里应该使用相同的 crop 位置，简化处理使用 x
                                                    y_m = x.clone()
                                                else:
                                                    y_m = self.to_tensor(gt_img_m) * 2.0 - 1.0
                                                ok = True
                            except Exception:
                                pass
                    else:
                        # 查找对应的退化图像
                        if (scene_id, deg_minus_name) in self.key_map:
                            try:
                                with self.env.begin(write=False) as txn:
                                    minus_data = pickle.loads(txn.get(self.key_map[(scene_id, deg_minus_name)].encode()))
                                    if isinstance(minus_data, dict):
                                        minus_array = minus_data.get('LQ', minus_data.get('input', None))
                                        if minus_array is not None:
                                            if isinstance(minus_array, np.ndarray):
                                                if minus_array.max() <= 1.0:
                                                    minus_array = (minus_array * 255).astype(np.uint8)
                                                minus_img = Image.fromarray(minus_array)
                                                # 应用相同的数据增强
                                                if self.is_train and self.patch_size is not None:
                                                    # 简化：使用相同的 crop（实际应该同步 crop）
                                                    minus_img, _ = self._random_crop(minus_img, gt_img)
                                                    minus_img, _ = self._random_augment(minus_img, gt_img)
                                                y_m = self.to_tensor(minus_img) * 2.0 - 1.0
                                                # 确保尺寸匹配
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
                        # 缺失则填 y（fallback）
                        y_minus[i] = y
                        has_cf[i] = 0.0
                else:
                    # 因子不在 S 中，y_minus 设为 y（或 0）
                    y_minus[i] = y
                    has_cf[i] = 0.0
        
        # 确保deg_name不为None（collate函数无法处理None）
        if deg_name is None:
            deg_name = ""
        
        # 构建返回字典
        result = {
            'x': x,  # clean/GT
            'y': y,  # composite degraded
            'deg_name': deg_name,  # 确保不为None
            'present': present,
            'w': w,
            'm': m,
            'y_minus': y_minus,
            'has_cf': has_cf,
        }
        
        # 兼容旧格式（LQ/GT）
        result['LQ'] = y
        result['GT'] = x
        
        # 深度图（如果使用）
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
            # 如果use_precomputed_depth为False或depth_array为None，不添加depth字段
        # 注意：为了兼容collate函数，不在result中添加None值字段
        
        return result
    
    def _random_crop(self, lq_img, gt_img):
        """随机裁剪"""
        w, h = lq_img.size
        if w < self.patch_size or h < self.patch_size:
            # 如果图像太小，先放大
            scale = max(self.patch_size / w, self.patch_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            lq_img = lq_img.resize((new_w, new_h), Image.BILINEAR)
            gt_img = gt_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h
        
        # 随机裁剪
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        
        lq_img = TF.crop(lq_img, i, j, self.patch_size, self.patch_size)
        gt_img = TF.crop(gt_img, i, j, self.patch_size, self.patch_size)
        
        return lq_img, gt_img
    
    def _random_augment(self, lq_img, gt_img):
        """随机数据增强"""
        # 随机水平翻转
        if random.random() > 0.5:
            lq_img = TF.hflip(lq_img)
            gt_img = TF.hflip(gt_img)
        
        # 随机垂直翻转
        if random.random() > 0.5:
            lq_img = TF.vflip(lq_img)
            gt_img = TF.vflip(gt_img)
        
        # ✅移除随机旋转，避免90/270度旋转导致的宽高互换问题
        # 如果需要旋转，可以使用180度旋转（不会改变宽高）
        # if random.random() > 0.5:
        #     angle = 180  # 只使用180度旋转，避免宽高互换
        #     lq_img = TF.rotate(lq_img, angle)
        #     gt_img = TF.rotate(gt_img, angle)
        
        return lq_img, gt_img
    
    def __del__(self):
        """关闭 LMDB 连接"""
        if hasattr(self, 'env') and self.env is not None:
            self.env.close()


# 如果 LMDB 不可用，可以尝试使用 ImageFolderDataset
# 但训练脚本默认使用 LMDBAllWeatherDataset
try:
    import lmdb
    LMDB_AVAILABLE = True
except ImportError:
    LMDB_AVAILABLE = False
    print("Warning: lmdb not installed. Install with: pip install lmdb")


class ImageFolderDataset(Dataset):
    """
    图像文件夹数据集（如果 LMDB 不可用，可以使用这个）。
    
    支持 CDD-11 格式的文件夹结构：
    - input/: 退化图像
    - clear/: 清晰图像
    """
    
    def __init__(
        self,
        root_dir,
        patch_size=256,
        is_train=True,
        use_precomputed_depth=False,
    ):
        """
        Args:
            root_dir: 数据集根目录
            patch_size: patch 大小
            is_train: 是否为训练模式
            use_precomputed_depth: 是否使用预计算的深度图
        """
        self.root_dir = root_dir
        self.input_dir = os.path.join(root_dir, 'input')
        self.clear_dir = os.path.join(root_dir, 'clear')
        self.patch_size = patch_size
        self.is_train = is_train
        self.use_precomputed_depth = use_precomputed_depth
        
        # 构建样本列表
        self.samples = self._build_samples()
        
        # Transforms
        self.to_tensor = transforms.ToTensor()
    
    def _build_samples(self):
        """构建样本列表"""
        samples = []
        
        if not os.path.exists(self.input_dir):
            return samples
        
        # 获取所有输入图像
        input_files = [f for f in os.listdir(self.input_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        for input_file in input_files:
            input_path = os.path.join(self.input_dir, input_file)
            
            # 查找对应的清晰图像
            # 尝试多种命名方式
            clear_path = None
            name_without_ext = os.path.splitext(input_file)[0]
            
            # 方式1: 直接匹配文件名
            for ext in ['.png', '.jpg', '.jpeg']:
                candidate = os.path.join(self.clear_dir, name_without_ext + ext)
                if os.path.exists(candidate):
                    clear_path = candidate
                    break
            
            # 方式2: 提取数字ID（CDD-11格式：degradation_id.png）
            if clear_path is None:
                import re
                match = re.search(r'(\d+)\.(png|jpg|jpeg)$', input_file, re.IGNORECASE)
                if match:
                    img_id = match.group(1)
                    for ext in ['.png', '.jpg', '.jpeg']:
                        candidate = os.path.join(self.clear_dir, f"{img_id}{ext}")
                        if os.path.exists(candidate):
                            clear_path = candidate
                            break
            
            if clear_path and os.path.exists(clear_path):
                # 提取退化名称（从文件名）
                deg_name = self._extract_degradation_name(input_file)
                samples.append((input_path, clear_path, deg_name))
        
        return samples
    
    def _extract_degradation_name(self, filename):
        """从文件名提取退化名称"""
        # CDD-11 格式：degradation_id.png
        # 例如：low_haze_rain_001.png -> low_haze_rain
        import re
        match = re.match(r'^(.+)_\d+\.(png|jpg|jpeg)$', filename, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """获取一个样本"""
        input_path, clear_path, deg_name = self.samples[idx]
        
        # 加载图像
        lq_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(clear_path).convert('RGB')
        
        # 数据增强（训练时）
        if self.is_train and self.patch_size is not None:
            lq_img, gt_img = self._random_crop(lq_img, gt_img)
            lq_img, gt_img = self._random_augment(lq_img, gt_img)
        
        # 转换为 tensor，值域 [-1, 1]
        lq_tensor = self.to_tensor(lq_img) * 2.0 - 1.0
        gt_tensor = self.to_tensor(gt_img) * 2.0 - 1.0
        
        result = {
            'LQ': lq_tensor,
            'GT': gt_tensor,
            'deg_name': deg_name,
        }
        
        # 深度图（如果使用）
        if self.use_precomputed_depth:
            # 尝试加载预计算的深度图
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
        """随机裁剪"""
        w, h = lq_img.size
        if w < self.patch_size or h < self.patch_size:
            scale = max(self.patch_size / w, self.patch_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            lq_img = lq_img.resize((new_w, new_h), Image.BILINEAR)
            gt_img = gt_img.resize((new_w, new_h), Image.BILINEAR)
            w, h = new_w, new_h
        
        i = random.randint(0, h - self.patch_size)
        j = random.randint(0, w - self.patch_size)
        
        lq_img = TF.crop(lq_img, i, j, self.patch_size, self.patch_size)
        gt_img = TF.crop(gt_img, i, j, self.patch_size, self.patch_size)
        
        return lq_img, gt_img
    
    def _random_augment(self, lq_img, gt_img):
        """随机数据增强"""
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

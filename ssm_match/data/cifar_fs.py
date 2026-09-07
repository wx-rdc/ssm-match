"""CIFAR-FS（跨域评测用）：cifar-100-python pickle + Bertinetto superclass 划分。

划分文件由 scripts/prepare_cifar_fs.py 生成（data/splits/cifar_fs/{train,val,test}.txt，
每行一个 0–99 的细类索引；12/4/4 个 superclass → 60/20/20 个细类）。
cifar_root 指向含 train/test/meta pickle 的目录（平台挂载 Open_Dataset/cifar-100-python）。
"""

import os
import pickle

import numpy as np
import torch


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")


def build_class_images(cifar_root: str, splits_dir: str, split: str,
                       cache_dir: str | None = None) -> dict:
    with open(os.path.join(splits_dir, f"{split}.txt")) as f:
        wanted = {int(line.strip()) for line in f if line.strip()}

    tensors = {c: [] for c in wanted}
    for batch_name in ("train", "test"):
        data = _load(os.path.join(cifar_root, f"{batch_name}"))
        fine = np.asarray(data["fine_labels"])
        images = torch.from_numpy(data["data"].reshape(-1, 3, 32, 32).astype(np.uint8))
        for c in wanted:
            sel = images[fine == c]
            tensors[c].append(sel)
    out = {}
    for c in wanted:
        out[str(c)] = torch.cat(tensors[c])  # (N,3,32,32) uint8，600 张/类
    return out


class CIFARFSTensorDataset(torch.utils.data.Dataset):
    """在内存 uint8 张量上做变换（跨域评测直接 32→224 上采样，遵循论文协议）。"""

    def __init__(self, class_images: dict, transform):
        self.items = [(c, i, img) for c, imgs in class_images.items()
                      for i, img in enumerate(imgs)]
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        c, i, img = self.items[i]
        from PIL import Image
        import torchvision.transforms.functional as TF
        pil = TF.to_pil_image(img)
        return c, i, self.transform(pil)

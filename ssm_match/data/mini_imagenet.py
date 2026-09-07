"""miniImageNet：Ravi & Larochelle 划分（仓库内置 CSV）+ 平台挂载/本地 images 目录。

数据来源两种方式：
  1) 启智平台：建任务时挂载公开数据集 HeartTo/mini-imagenet，容器内路径
     <data_root>/mini-imagenet/images/*.jpg（或 <data_root>/images/，自动探测）；
  2) 本地：openi dataset download HeartTo/mini-imagenet mini-imagenet -s ./data
"""

import csv
import os


def read_split_csv(path: str) -> dict:
    """返回 {wnid: [文件名,...]}，顺序与 CSV 一致。"""
    classes = {}
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0] == "filename":
                continue
            fname, label = row[0].strip(), row[1].strip()
            classes.setdefault(label, []).append(fname)
    return classes


def find_images_dir(data_root: str) -> str:
    for cand in (os.path.join(data_root, "mini-imagenet", "images"),
                 os.path.join(data_root, "mini-imagenet"),
                 os.path.join(data_root, "images"),
                 data_root):
        if os.path.isdir(cand) and any(f.endswith(".jpg") for f in os.listdir(cand)):
            return cand
    raise FileNotFoundError(f"未在 {data_root} 下找到 miniImageNet jpg 目录")


def build_class_images(data_root: str, splits_dir: str) -> tuple:
    """→ (class_images: {wnid: [绝对路径]}, n_total)。"""
    img_dir = find_images_dir(data_root)
    classes = read_split_csv(os.path.join(splits_dir, "train.csv"))
    for split in ("val", "test"):
        classes.update(read_split_csv(os.path.join(splits_dir, f"{split}.csv")))
    n = 0
    out = {}
    for wnid, files in classes.items():
        paths = [os.path.join(img_dir, f) for f in files]
        missing = [p for p in paths if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(f"{wnid} 缺少 {len(missing)} 张图，示例: {missing[0]}")
        out[wnid] = paths
        n += len(paths)
    return out, n

import os

from .episode import EpisodeBatcher, EpisodeDataset, EpisodeSampler
from .mini_imagenet import build_class_images as build_mini_imagenet
from .transforms import build_transform

__all__ = ["EpisodeBatcher", "EpisodeDataset", "EpisodeSampler",
           "build_mini_imagenet", "build_transform", "load_split"]

DATASETS = ("mini", "cub", "cars", "cifar_fs")


def load_split(dataset: str, data_root: str, splits_dir: str, split: str) -> dict:
    """按数据集加载一个划分 → {类名: [图片路径]} 或 {类名: uint8 张量}（cifar_fs）。

    mini: Ravi & Larochelle CSV（train/val/test.csv）；cub/cars: class-folder
    + make_class_splits.py 生成的 txt；cifar_fs: Bertinetto txt + pickle 张量。
    """
    if dataset == "cifar_fs":
        from .cifar_fs import build_class_images
        return build_class_images(data_root, splits_dir, split)
    if dataset == "mini":
        import csv
        classes, _ = build_mini_imagenet(data_root, splits_dir)
        with open(os.path.join(splits_dir, f"{split}.csv"), newline="") as f:
            wanted = {row[1].strip() for row in csv.reader(f)
                      if row and row[0] != "filename"}
        return {k: v for k, v in classes.items() if k in wanted}
    if dataset in ("cub", "cars"):
        from .class_folder import build_class_images
        return build_class_images(data_root, splits_dir, split)
    raise ValueError(f"未知数据集 {dataset!r}，可选 {DATASETS}")

"""通用 class-folder 数据集（CUB-200-2011 / Stanford Cars）+ 划分文件。

目录结构：<root>/<wnid or 类名文件夹>/*.jpg
划分：data/splits/<name>/{train,val,test}.txt，每行一个类文件夹名；
由 scripts/make_class_splits.py 确定性生成。
"""

import os


def build_class_images(root: str, splits_dir: str, split: str) -> dict:
    with open(os.path.join(splits_dir, f"{split}.txt")) as f:
        class_names = [line.strip() for line in f if line.strip()]
    out = {}
    for name in class_names:
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            raise FileNotFoundError(folder)
        exts = (".jpg", ".jpeg", ".png")
        out[name] = sorted(os.path.join(folder, f) for f in os.listdir(folder)
                           if f.lower().endswith(exts))
    return out

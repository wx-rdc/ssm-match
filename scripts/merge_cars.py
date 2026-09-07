"""把 Stanford Cars 的 train/val 类目录合并为单一 196 类根目录。

shisan/standfordcars 镜像把官方 8,144 train + 8,041 test 重打包为
StandfordCars/{train,val}/<类名>/*.jpg；训练入口要求单一根目录下同时含
train/val/test 划分类。本脚本以文件级符号链接合并（零拷贝），并处理
"Ram C/V Cargo Van Minivan 2012" 因类名含 / 被拆成嵌套目录的情况。

用法：
    python scripts/merge_cars.py --src data/hf/stanford-cars/StandfordCars \
        --dst data/hf/cars-merged
之后：
    python scripts/make_class_splits.py --root data/hf/cars-merged \
        --out data/splits/cars --ratio 130 17 49 --seed 42
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="StandfordCars/{train,val} 所在目录")
    ap.add_argument("--dst", required=True, help="合并输出的 196 类根目录")
    args = ap.parse_args()

    n = 0
    for split in ("train", "val"):
        split_dir = os.path.join(args.src, split)
        for cls in sorted(os.listdir(split_dir)):
            cpath = os.path.join(split_dir, cls)
            if not os.path.isdir(cpath):
                continue  # car_train.txt 等清单文件
            cdir = os.path.join(args.dst, cls.replace("/", "_"))
            os.makedirs(cdir, exist_ok=True)
            for root, _, files in os.walk(cpath):
                for f in files:
                    if not f.lower().endswith(".jpg"):
                        continue
                    rel = os.path.relpath(os.path.join(root, f), cpath)
                    link = os.path.join(cdir, f"{split}_{rel.replace(os.sep, '_')}")
                    if not os.path.lexists(link):
                        os.symlink(os.path.relpath(os.path.join(root, f), cdir), link)
                    n += 1

    classes = sorted(d for d in os.listdir(args.dst)
                     if os.path.isdir(os.path.join(args.dst, d)))
    imgs = sum(len(os.listdir(os.path.join(args.dst, c))) for c in classes)
    print(f"merged: {len(classes)} 类, {imgs} 张（官方应为 196 类 16,185 张）")
    if len(classes) != 196 or imgs != 16185:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

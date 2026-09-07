"""为 class-folder 数据集（CUB-200-2011 / Stanford Cars）生成确定性 few-shot 类划分。

用法：
    python scripts/make_class_splits.py --root data/CUB_200_2011/images \
        --out data/splits/cub --ratio 100 50 50 --seed 42

注意：CUB/Cars 没有官方 FSL 类划分标准（miniImageNet 的 Ravi 划分与 CIFAR-FS 的
Bertinetto 划分除外），此处用固定种子确定性划分并在实验记录中注明。
"""

import argparse
import os
import random

EXTS = (".jpg", ".jpeg", ".png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ratio", nargs=3, type=int, default=[100, 50, 50],
                    help="train/val/test 类数（如 CUB 100 50 50，Cars 130 17 49）")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    classes = sorted(d for d in os.listdir(args.root)
                     if os.path.isdir(os.path.join(args.root, d)))
    assert sum(args.ratio) == len(classes), \
        f"ratio {args.ratio} 总和应等于类数 {len(classes)}"
    rng = random.Random(args.seed)
    rng.shuffle(classes)

    os.makedirs(args.out, exist_ok=True)
    pos = 0
    for name, cnt in zip(("train", "val", "test"), args.ratio):
        part = classes[pos:pos + cnt]
        pos += cnt
        with open(os.path.join(args.out, f"{name}.txt"), "w") as f:
            f.write("\n".join(part) + "\n")
        print(f"{name}: {cnt} 类 -> {os.path.join(args.out, name + '.txt')}")


if __name__ == "__main__":
    main()

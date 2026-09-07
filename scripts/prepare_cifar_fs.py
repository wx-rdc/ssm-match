"""生成 CIFAR-FS 划分文件（Bertinetto r2d2 标准 superclass 12/4/4 → 60/20/20 类）。

用法：
    python scripts/prepare_cifar_fs.py --cifar_root data/cifar-100-python \
        --out data/splits/cifar_fs

superclass → 细类映射直接从 cifar-100-python 的 train 批次读取（权威、无手工表），
划分断言 60/20/20。跨域评测（mini→CIFAR-FS）仅使用 test 划分。
"""

import argparse
import os
import pickle
from collections import defaultdict

SUPERCLASS_SPLITS = {
    # r2d2 发布的 CIFAR-FS 标准划分（superclass 索引见 CIFAR-100 coarse_labels）
    "train": [0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 13],
    "val": [12, 15, 16, 17],
    "test": [7, 14, 18, 19],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cifar_root", required=True, help="含 train/test/meta 的目录")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.cifar_root, "train"), "rb") as f:
        batch = pickle.load(f, encoding="latin1")
    fine = batch["fine_labels"]
    coarse = batch["coarse_labels"]

    fine_to_coarse = {}
    for fi, ci in zip(fine, coarse):
        fine_to_coarse[fi] = ci
    assert len(fine_to_coarse) == 100, "细类→粗类映射应覆盖 100 类"

    coarse_to_fine = defaultdict(list)
    for fi in sorted(fine_to_coarse):
        coarse_to_fine[fine_to_coarse[fi]].append(fi)

    os.makedirs(args.out, exist_ok=True)
    assigned = []
    for split, scs in SUPERCLASS_SPLITS.items():
        fine_classes = sorted(fi for sc in scs for fi in coarse_to_fine[sc])
        assert len(fine_classes) == len(scs) * 5, f"{split}: 每 superclass 应含 5 细类"
        assigned += fine_classes
        path = os.path.join(args.out, f"{split}.txt")
        with open(path, "w") as f:
            f.write("\n".join(map(str, fine_classes)) + "\n")
        print(f"{split}: {len(scs)} superclass / {len(fine_classes)} 类 -> {path}")
    assert sorted(assigned) == list(range(100)), "划分必须恰好覆盖全部 100 类且不重叠"
    print("OK: 100 类划分无重叠无遗漏")


if __name__ == "__main__":
    main()

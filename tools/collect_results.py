"""汇总评测 JSON → markdown 表（P2 结果回收后回填论文用）。

扫描目录下所有 eval_*.json（tools/eval.py 的 --out 约定）或 evaluator dump，
输出：
  1. 逐 run 明细（suite / seed / 数据集 / split / mean±CI95）；
  2. 按 (dataset, k_shot, dump 名) 聚合的跨种子 mean ± std（5/3 种子），
     即论文表格单元格的最终数字。

用法：
    python tools/collect_results.py output/ --output output/summary.md
    python tools/collect_results.py output/main_mini_1shot
"""

import argparse
import glob
import json
import os

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="汇总 eval_*.json 为论文可用的表格")
    p.add_argument("roots", nargs="+", help="output 根目录或 suite 目录（可多个）")
    p.add_argument("--output", default=None, help="markdown 另存路径（缺省打印）")
    return p.parse_args()


def collect(roots):
    rows = []
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "**", "eval_*.json"),
                                     recursive=True)):
            with open(path, encoding="utf-8") as f:
                d = json.load(f)
            accs = np.asarray(d["accs"], dtype=np.float64)
            mean = accs.mean()
            ci = 1.96 * accs.std(ddof=1) / np.sqrt(accs.size) if accs.size > 1 else 0.0
            rel = os.path.relpath(path, root if os.path.isdir(root) else ".")
            # <suite>/seed_<s>/eval_<dataset>_<k>shot[_<split>|_cifar_fs].json
            parts = rel.split(os.sep)
            suite = parts[0] if len(parts) > 2 else ""
            seed = next((p for p in parts if p.startswith("seed_")), "-")
            rows.append({"suite": suite, "seed": seed, "file": os.path.basename(path),
                         "method": d.get("method", "?"), "n": accs.size,
                         "mean": mean, "ci95": ci, "path": path})
    return rows


def aggregate(rows):
    groups = {}
    for r in rows:
        key = (r["method"], r["file"])
        groups.setdefault(key, []).append(r)
    out = []
    for (method, fname), rs in sorted(groups.items()):
        means = np.array([r["mean"] for r in rs])
        out.append({"method": method, "file": fname, "n_runs": len(rs),
                    "mean": means.mean(), "std": means.std(ddof=1) if len(rs) > 1 else 0.0})
    return out


def main():
    args = parse_args()
    rows = collect(args.roots)
    if not rows:
        raise SystemExit("未找到 eval_*.json（先跑 tools/eval.py 或 boot 一条龙）")

    lines = ["# P2 评测汇总", "", "## 逐 run 明细", "",
             "| suite | seed | 文件 | method | n_ep | mean±CI95 |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['suite']} | {r['seed']} | {r['file']} | {r['method']} "
                     f"| {r['n']} | {r['mean']*100:.2f} ± {r['ci95']*100:.2f} |")

    lines += ["", "## 跨种子聚合（论文表格单元格）", "",
              "| method(suite) | dump | runs | mean ± std |", "|---|---|---|---|"]
    for a in aggregate(rows):
        lines.append(f"| {a['method']} | {a['file']} | {a['n_runs']} "
                     f"| {a['mean']*100:.2f} ± {a['std']*100:.2f} |")

    text = "\n".join(lines) + "\n"
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"已写入 {args.output}")
    print(text)


if __name__ == "__main__":
    main()

"""统计检验脚本 —— 该脚本回应审稿意见 #R3-5：
"Add statistical tests, confidence intervals and provide results for identical
test cases to support statements of improved performance. Current results are
based on only three runs."

输入：两个及以上的 JSON 文件，每个包含同一批测试 episodes 的逐 episode 正确数/
准确率列表，格式约定（engine/evaluator.evaluate(..., dump_path=...) 即按此落盘）：
    {"method": 方法名, "n_way": 5, "k_shot": K, "n_episodes": N,
     "accs": [每 episode 准确率 float]}
各方法的 episodes 必须同 seed 同 episode 集（逐条对齐），配对检验才有意义；
长度不一致的方法会从配对检验中剔除并告警。

功能：
  - 各方法总体准确率 mean ± 95% CI（正态近似与 bootstrap 各给一份）；
  - 指定两两配对比较（--pair "A vs B"，可多次；缺省=全部两两组合）：
    配对 t 检验、Wilcoxon 符号秩检验（scipy 缺失则跳过并提示）、准确率差的
    bootstrap 95% CI（重采样 10000 次，种子固定）、配对胜/平/负计数；
  - 多重比较时给 Holm-Bonferroni 校正后 p 值（t 检验族与 Wilcoxon 族各自校正）；
  - 结果以 markdown 表格打印，--output 可另存。

纯 numpy(+可选 scipy) 实现，无 torch/GPU 依赖，任何机器可跑。

用法：
    python tools/stats_tests.py run_seed1.json run_seed2.json run_seed3.json \
        --pair "ssm-match vs proto-baseline" --output output/stats.md
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

try:
    from scipy import stats as sps
except ImportError:
    sps = None

N_BOOT = 10000
BOOT_SEED = 20260907  # bootstrap 固定种子，重跑可复现


def parse_args():
    p = argparse.ArgumentParser(
        description="逐 episode 结果的配对统计检验（回应审稿意见 #R3-5）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("runs", nargs="+", help="评测 JSON 文件（≥2 个做配对比较）")
    p.add_argument("--pair", action="append", default=[],
                   help='配对比较 "A vs B"，可多次给出；缺省=全部两两组合')
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--output", default=None, help="markdown 结果另存路径")
    return p.parse_args()


def load_run(path: str) -> dict:
    """加载并校验单个评测 JSON。"""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    for key in ("method", "n_way", "k_shot", "n_episodes", "accs"):
        if key not in d:
            raise KeyError(f"{path} 缺少约定字段 '{key}'（应由 evaluator dump_path 生成）")
    accs = np.asarray(d["accs"], dtype=np.float64)
    if accs.ndim != 1 or accs.size != int(d["n_episodes"]):
        raise ValueError(f"{path}: accs 长度 {accs.size} 与 n_episodes={d['n_episodes']} 不符")
    if not (np.all(accs >= 0) and np.all(accs <= 1)):
        raise ValueError(f"{path}: accs 应为 [0,1] 的逐 episode 准确率")
    return {"method": str(d["method"]), "n_way": int(d["n_way"]),
            "k_shot": int(d["k_shot"]), "accs": accs, "path": path}


def ci_normal(accs: np.ndarray) -> tuple:
    """mean ± 1.96·SE（正态近似，样本标准差 ddof=1）。"""
    n = accs.size
    mean = float(accs.mean())
    half = 1.96 * float(accs.std(ddof=1)) / np.sqrt(n) if n > 1 else 0.0
    return mean, half


def ci_bootstrap(accs: np.ndarray, n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                 stat=np.mean) -> tuple:
    """bootstrap 百分位 95% CI（重采样 n_boot 次，固定种子可复现）。"""
    rng = np.random.default_rng(seed)
    n = accs.size
    idx = rng.integers(0, n, size=(n_boot, n))
    stats = stat(accs[idx], axis=1)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


def paired_test(a: np.ndarray, b: np.ndarray) -> dict:
    """配对比较：t 检验 / Wilcoxon / bootstrap 差值 CI / 胜平负。"""
    diff = a - b
    out = {
        "diff_mean": float(diff.mean()),
        "win": int(np.sum(diff > 0)), "tie": int(np.sum(diff == 0)),
        "loss": int(np.sum(diff < 0)), "n": int(diff.size),
    }
    if sps is None:
        out["t_p"] = out["w_p"] = None
        print("[提示] 未安装 scipy：配对 t 检验与 Wilcoxon 检验被跳过"
              "（pip install scipy 后重跑可得 p 值）")
    else:
        if diff.std(ddof=1) == 0:
            out["t_p"] = 1.0                       # 完全无差异，t 检验退化
        else:
            out["t_p"] = float(sps.ttest_rel(a, b).pvalue)
        try:
            out["w_p"] = float(sps.wilcoxon(a, b).pvalue)
        except ValueError:                          # 全为零差等退化输入
            out["w_p"] = 1.0
    rng = np.random.default_rng(BOOT_SEED)
    n = diff.size
    idx = rng.integers(0, n, size=(N_BOOT, n))
    boots = diff[idx].mean(axis=1)
    out["boot_lo"], out["boot_hi"] = (float(np.percentile(boots, 2.5)),
                                      float(np.percentile(boots, 97.5)))
    return out


def holm_bonferroni(pvals: list) -> np.ndarray:
    """Holm-Bonferroni 逐步校正：升序遍历，adj_i = max_{j<=i}(m-j)·p_j，截断到 1。"""
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(np.argsort(p, kind="stable")):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def parse_pairs(names: list, pair_args: list) -> list:
    """解析 --pair "A vs B"；未给出时默认全部两两组合（按输入文件顺序）。"""
    pairs = []
    for spec in pair_args:
        if " vs " not in spec:
            raise ValueError(f'--pair 需要形如 "A vs B"，得到: {spec}')
        a, b = (s.strip() for s in spec.split(" vs ", 1))
        if a not in names or b not in names:
            raise ValueError(f"--pair 的方法名未在输入文件中找到: {spec}；"
                             f"可用: {names}")
        pairs.append((a, b))
    if not pairs:
        pairs = list(itertools.combinations(names, 2))
    return pairs


def render_markdown(runs: list, pairs: list, results: dict) -> str:
    lines = ["# SSM-Match 统计检验（回应审稿意见 #R3-5）", ""]
    lines += ["## 各方法总体准确率", "",
              "| 方法 | n_way/k_shot | episodes | mean | 95% CI(正态) | 95% CI(bootstrap) |",
              "|---|---|---|---|---|---|"]
    for r in runs:
        mean, half = ci_normal(r["accs"])
        lo, hi = ci_bootstrap(r["accs"])
        lines.append(f"| {r['method']} | {r['n_way']}/{r['k_shot']} | {r['accs'].size} "
                     f"| {mean:.4f} | ±{half:.4f} | [{lo:.4f}, {hi:.4f}] |")

    lines += ["", "## 两两配对比较（同一批 episodes，逐条配对）", ""]
    if sps is None:
        lines += ["> 注：未安装 scipy，t/Wilcoxon p 值缺失；bootstrap CI 与胜平负不受影响。", ""]
    lines += ["| 配对 | Δmean(B−A) | 配对 t p | t p (Holm) | Wilcoxon p | W p (Holm)"
              " | Δ 的 bootstrap 95% CI | 胜/平/负 (A 视角) |",
              "|---|---|---|---|---|---|---|---|"]
    for (a_name, b_name) in pairs:
        key = (a_name, b_name)
        r = results[key]
        t_p = "—" if r["t_p"] is None else f"{r['t_p']:.2e}"
        t_holm = "—" if r["t_holm"] is None else f"{r['t_holm']:.2e}"
        w_p = "—" if r["w_p"] is None else f"{r['w_p']:.2e}"
        w_holm = "—" if r["w_holm"] is None else f"{r['w_holm']:.2e}"
        sig = ""
        if r["t_holm"] is not None:
            sig = " *" if r["t_holm"] < 0.05 else ""
        lines.append(f"| {a_name} vs {b_name}{sig} | {r['diff_mean']:+.4f} | {t_p} "
                     f"| {t_holm} | {w_p} | {w_holm} "
                     f"| [{r['boot_lo']:+.4f}, {r['boot_hi']:+.4f}] "
                     f"| {r['win']}/{r['tie']}/{r['loss']} |")
    lines += ["", "注：Δmean = mean(B) − mean(A)；胜/平/负按逐 episode 准确率差 "
              "(>0 / =0 / <0) 计；Holm-Bonferroni 在本次全部配对内分别对 t 检验族与 "
              "Wilcoxon 族逐步校正；`*` 表示 Holm 校正后 p<0.05（A 显著优于 B 时 "
              "Δmean<0 且显著）。bootstrap 重采样 10000 次（种子固定 "
              f"{BOOT_SEED}）。", ""]
    return "\n".join(lines)


def main():
    args = parse_args()
    if len(args.runs) < 2:
        print("[错误] 至少需要 2 个评测 JSON 文件才能做配对比较")
        sys.exit(1)
    runs = [load_run(p) for p in args.runs]
    names = [r["method"] for r in runs]
    dup = {n for n in names if names.count(n) > 1}
    if dup:
        print(f"[错误] method 名重复（配对检验需要唯一标识）: {dup}；"
              f"请修改 JSON 中的 method 字段区分（如 ssm-match-seed1）")
        sys.exit(1)

    base = runs[0]
    for r in runs[1:]:
        if (r["n_way"], r["k_shot"]) != (base["n_way"], base["k_shot"]):
            print(f"[错误] {r['path']} 的 n_way/k_shot 与 {base['path']} 不一致，"
                  f"不构成同一批测试")
            sys.exit(1)
    n_ref = base["accs"].size
    aligned = [r for r in runs if r["accs"].size == n_ref]
    for r in runs:
        if r["accs"].size != n_ref:
            print(f"[警告] {r['path']} 的 episode 数({r['accs'].size})与 "
                  f"{n_ref} 不一致：总体 CI 照常给出，但该方法的配对比较被剔除")
    by_name = {r["method"]: r["accs"] for r in aligned}

    pairs = parse_pairs([r["method"] for r in aligned], args.pair)
    results = {}
    for a_name, b_name in pairs:
        results[(a_name, b_name)] = paired_test(by_name[a_name], by_name[b_name])
    # Holm-Bonferroni：在本次全部配对内、按检验族分别校正
    t_ps = [results[k]["t_p"] for k in results]
    w_ps = [results[k]["w_p"] for k in results]
    t_adj = holm_bonferroni(t_ps) if sps is not None and t_ps else None
    w_adj = holm_bonferroni(w_ps) if sps is not None and w_ps else None
    for i, k in enumerate(results):
        results[k]["t_holm"] = float(t_adj[i]) if t_adj is not None else None
        results[k]["w_holm"] = float(w_adj[i]) if w_adj is not None else None

    md = render_markdown(runs, pairs, results)
    print(md)
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"markdown 已保存: {args.output}")


if __name__ == "__main__":
    main()

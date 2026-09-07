"""支持集顺序/置换敏感性评测 —— 该脚本回应审稿意见 #R2-3 / #R3-4：
"Since the backward scan is over the sequence of support separator query, the
query states are support-conditioned. Describe this in mathematical terms, and
test sensitivity to support-sample ordering and to permutation of samples in
the support set."

复用 engine/evaluator.evaluate 的数据管线（EpisodeSampler/EpisodeDataset/
EpisodeBatcher，评测独立种子），但对每个测试 episode 额外做：
  1) 规范顺序评测一次；
  2) 同一批支持图像沿 K 维做 R 次随机置换（--n-perms，默认 20）各评一次；
  3) 再做一次完全逆序。
同一 episode 的支持图像在所有置换中完全相同：骨干特征只算一次，置换通过沿 K 维
索引特征实现（只变顺序，不改内容、不重采样）。

输出：规范顺序准确率、置换后准确率 mean±std、二者差值及其逐 episode 分布分位数、
|Δacc| 超过 5% 的 episode 占比；--dump 可落盘逐 episode 结果（JSON）。
必须 --ckpt 加载训练好的模型。k_shot=1 时置换无意义（K=1 只有一种顺序），
脚本会明确提示并退化为恒等置换。

用法：
    python tools/eval_ordering_sensitivity.py --ckpt output/best.pt \
        --data_root data/mini-imagenet --k-shot 5 --n-episodes 100 --dump output/ordering.json
"""

import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="支持集顺序/置换敏感性评测（回应审稿意见 #R2-3/#R3-4）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="训练好的 SSMMatch 检查点（必须）")
    p.add_argument("--dataset", default="mini", choices=["mini", "class-folder"],
                   help="数据集类型（CIFAR-FS 的 class_images 是内存张量而非路径，"
                        "与 EpisodeDataset 的按路径加载不兼容，故不在此支持）")
    p.add_argument("--data_root", required=True, help="图像数据根目录")
    p.add_argument("--splits_dir", default=os.path.join(REPO_ROOT, "data", "splits", "mini"))
    p.add_argument("--model_dir", default=".", help="Vim 权重目录")
    p.add_argument("--vim_ckpt", default="vim_t_midclstok_ft_78p3acc.pth")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5)
    p.add_argument("--n_query", type=int, default=15)
    p.add_argument("--n_episodes", type=int, default=100)
    p.add_argument("--n_perms", type=int, default=20, help="每 episode 随机置换次数 R")
    p.add_argument("--query_chunk", type=int, default=5)
    p.add_argument("--batch_episodes", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=12345,
                   help="episode 采样种子（置换用 seed+1，独立于采样）")
    p.add_argument("--dump", default=None, help="逐 episode 结果 JSON 保存路径")
    return p.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "cuda" and not torch.cuda.is_available():
        print("[错误] 指定 --device cuda 但无可用 GPU；真实 mamba_ssm 前向需要 CUDA。")
        sys.exit(1)
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[提示] 无 GPU，退回 CPU 运行（真实 mamba_ssm 通常仍需 CUDA，可能失败；"
              "建议减小 --n-episodes）")
        return torch.device("cpu")
    return torch.device(choice)


def build_test_class_images(args) -> dict:
    """从 --data_root + split 文件读测试类 → {类 ID: [图像绝对路径]}（复用现有 data 模块）。"""
    if args.dataset == "mini":
        from ssm_match.data.mini_imagenet import build_class_images, read_split_csv
        classes, _ = build_class_images(args.data_root, args.splits_dir)
        test_names = set(read_split_csv(os.path.join(args.splits_dir, "test.csv")))
        out = {k: v for k, v in classes.items() if k in test_names}
        if not out:
            raise FileNotFoundError(f"{args.splits_dir}/test.csv 与数据不匹配")
        return out
    from ssm_match.data.class_folder import build_class_images
    return build_class_images(args.data_root, args.splits_dir, "test")


def backbone_features(backbone, images: torch.Tensor) -> dict:
    """(...,3,H,W) → 尺度名→(...,L,C)。

    VimTiny.forward 仅接受 4D 输入（nn.Conv2d 拒绝 >4D），故先展平批维再还原。
    （evaluator.evaluate 里 backbone(6D) 的调用在当前实现下会因此报错，
    本脚本自行展平规避；见最终汇报的已知问题。）
    """
    lead = images.shape[:-3]
    feats = backbone(images.reshape(-1, *images.shape[-3:]))._asdict()
    return {k: v.reshape(*lead, *v.shape[1:]) for k, v in feats.items()}


@torch.no_grad()
def episode_acc(model, feats_s: dict, feats_q: dict, perm=None,
                n_way: int = 5, n_query: int = 15, device="cpu") -> float:
    """单 episode 准确率；perm 非空时按它沿 K 维重排支持特征（内容不变只变顺序）。"""
    if perm is not None:
        feats_s = {k: v[:, :, perm] for k, v in feats_s.items()}
    logits = model(feats_s, feats_q)["logits"]                # (1,m,n)
    pred = logits.argmax(dim=-1).squeeze(0).view(n_way, n_query)
    labels = torch.arange(n_way, device=device).unsqueeze(1)
    return (pred == labels).float().mean().item()


def make_perms(K: int, n_perms: int, rng: np.random.Generator) -> tuple:
    """R 个互不相同且异于规范序的置换 + 一个完全逆序（K 太小时自动截断并提示）。"""
    max_distinct = math.factorial(K) - 1                      # 除规范序外的全部排列
    r = min(n_perms, max_distinct)
    if r < n_perms:
        print(f"[提示] K={K} 只有 {max_distinct} 个非规范排列，置换次数截断为 {r}"
              + ("（K=1 时顺序无意义，结果应与规范序完全一致）" if K == 1 else ""))
    perms, seen = [], {tuple(range(K))}
    while len(perms) < r:
        cand = tuple(int(i) for i in rng.permutation(K))
        if cand not in seen:
            seen.add(cand)
            perms.append(np.array(cand))
    reversed_perm = np.arange(K)[::-1].copy()
    return perms, reversed_perm


def main():
    args = parse_args()
    try:
        run(args)
    except ImportError as e:
        print(f"[错误] 缺少依赖：{e}（真实 mamba_ssm/causal-conv1d 需在 GPU 平台环境安装）")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[错误] 文件/目录未找到：{e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[错误] 运行失败（常见原因：检查点与当前 n_way/query_chunk 不一致，"
              f"或真实 mamba_ssm 前向需要 GPU）：{e}")
        sys.exit(1)


def run(args):
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)

    from ssm_match.backbone import FrozenVimBackbone
    from ssm_match.data import (EpisodeBatcher, EpisodeDataset, EpisodeSampler,
                                build_transform)
    from ssm_match.models import SSMMatch

    vim_path = os.path.join(args.model_dir, args.vim_ckpt)
    if not os.path.isfile(vim_path):
        print(f"[错误] 未找到 Vim 权重 {vim_path}（--model_dir/--vim_ckpt）")
        sys.exit(1)
    backbone = FrozenVimBackbone(vim_path).to(device)

    model = SSMMatch(n_way=args.n_way, query_chunk=args.query_chunk).to(device)
    payload = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"[ckpt] 已加载 {args.ckpt}"
          f"（epoch={payload.get('epoch')}，best_acc={payload.get('best_acc')}）")

    class_images = build_test_class_images(args)
    sampler = EpisodeSampler(class_images, args.n_way, args.k_shot, args.n_query,
                             seed=args.seed)
    dataset = EpisodeDataset(sampler, args.n_episodes, build_transform(train=False))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_episodes * args.n_way * (args.k_shot + args.n_query),
        shuffle=False, num_workers=args.num_workers,
        collate_fn=EpisodeBatcher(args.n_way).collate)

    # 置换 RNG 与 episode 采样分离（采样由 sampler 内部 numpy Generator 驱动）
    perm_rng = np.random.default_rng(args.seed + 1)

    accs_canonical, accs_reversed, accs_perm_mean, accs_perm_all = [], [], [], []
    done = 0
    for sup, queries in loader:
        if done >= args.n_episodes:
            break
        sup = sup.to(device)                                  # (b,n,K,3,H,W)
        feats_s = backbone_features(backbone, sup)            # (b,n,K,L,C)，本批只算一次
        for qi, q_img in enumerate(queries):
            if done >= args.n_episodes:
                break
            q_img = q_img.to(device)
            feats_q = backbone_features(backbone, q_img.unsqueeze(0))  # (1,m,L,C)
            fs = {k: v[qi:qi + 1] for k, v in feats_s.items()}
            # 支持特征一次前向得到，之后所有置换只沿 K 维索引 → 内容严格相同
            perms, rev = make_perms(args.k_shot, args.n_perms, perm_rng)
            acc_c = episode_acc(model, fs, feats_q, None, args.n_way,
                                args.n_query, device)
            accs_p = [episode_acc(model, fs, feats_q, p, args.n_way,
                                  args.n_query, device) for p in perms]
            acc_r = episode_acc(model, fs, feats_q, rev, args.n_way,
                                args.n_query, device)
            accs_canonical.append(acc_c)
            accs_perm_all.append(accs_p)
            accs_perm_mean.append(float(np.mean(accs_p)) if accs_p else acc_c)
            accs_reversed.append(acc_r)
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{args.n_episodes}: running canonical="
                      f"{sum(accs_canonical) / done:.4f}", flush=True)

    c = np.array(accs_canonical)
    p_mean = np.array(accs_perm_mean)
    r = np.array(accs_reversed)
    diff = c - p_mean                                          # 规范序 − 置换均值
    diff_rev = c - r
    q10, q25, q50, q75, q90 = np.percentile(diff, [10, 25, 50, 75, 90])
    frac_gt5 = float(np.mean(np.abs(diff) > 0.05))
    flat_perms = [x for a in accs_perm_all for x in a]
    summary = {
        "n_way": args.n_way, "k_shot": args.k_shot, "n_query": args.n_query,
        "n_episodes": done, "n_perms": len(accs_perm_all[0]) if accs_perm_all else 0,
        "canonical_acc": float(c.mean()),
        "permuted_acc_mean": float(p_mean.mean()),
        "permuted_acc_std": float(p_mean.std(ddof=1)) if done > 1 else 0.0,
        "permuted_acc_std_pooled": float(np.std(flat_perms, ddof=1)) if len(flat_perms) > 1 else 0.0,
        "reversed_acc_mean": float(r.mean()),
        "diff_mean": float(diff.mean()),
        "diff_quantiles": {"p10": float(q10), "p25": float(q25), "p50": float(q50),
                           "p75": float(q75), "p90": float(q90)},
        "diff_reversed_mean": float(diff_rev.mean()),
        "frac_abs_change_gt_5pct": frac_gt5,
        "n_abs_change_gt_5pct": int(round(frac_gt5 * done)),
    }

    print("\n==== 支持集顺序/置换敏感性（回应 #R2-3/#R3-4）====")
    print(f"规范顺序 acc                : {summary['canonical_acc']:.4f}")
    print(f"置换后 acc (mean±std)       : {summary['permuted_acc_mean']:.4f} "
          f"± {summary['permuted_acc_std']:.4f}"
          f"（逐 episode 置换均值，跨 episode std；合并 std="
          f"{summary['permuted_acc_std_pooled']:.4f}）")
    print(f"完全逆序 acc                : {summary['reversed_acc_mean']:.4f} "
          f"(Δ={summary['diff_reversed_mean']:+.4f})")
    print(f"差值 canonical−perm (mean)  : {summary['diff_mean']:+.4f}")
    print("逐 episode 差值分位数        : "
          + " ".join(f"{k}={v:+.4f}" for k, v in summary["diff_quantiles"].items()))
    print(f"|Δacc|>5% 的 episode 占比    : {frac_gt5 * 100:.1f}% "
          f"({summary['n_abs_change_gt_5pct']}/{done})")

    if args.dump:
        parent = os.path.dirname(os.path.abspath(args.dump))
        os.makedirs(parent, exist_ok=True)
        with open(args.dump, "w", encoding="utf-8") as f:
            json.dump({"summary": summary,
                       "accs_canonical": accs_canonical,
                       "accs_permuted": accs_perm_all,
                       "accs_reversed": accs_reversed},
                      f, ensure_ascii=False, indent=2)
        print(f"逐 episode 结果已保存: {args.dump}")


if __name__ == "__main__":
    main()

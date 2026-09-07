"""可视化脚本 —— 该脚本回应审稿意见 #R3-7（Mamba 无注意力图，图怎么来的）：
"It is unclear how the 'attention maps' and the t-SNE embeddings were produced
by Mamba. The typical attention map produced by e.g. Transformer-based models
is not produced by Mamba..."

Mamba 不产生注意力图，本脚本产出三类可复现的替代证据（全部由真实前向/反向
计算得出，无任何手工绘制的示意成分）：
  1) 匹配显著性图：对 episode 的每个查询图像，计算真类匹配得分对输入图像的
     梯度 saliency（|∇| 对通道取最大 → 224×224），叠加原图保存。骨干虽冻结，
     但允许梯度流到输入——绕过 FrozenVimBackbone.forward 的 @torch.no_grad
     装饰器，直接 to_multiscale(backbone.vim(x))（模型参数已冻结，梯度只流向
     查询张量）。
  2) 逐 patch 读出状态强度图：取匹配器查询区扫描输出的每 patch 隐藏态范数，
     重排为 14×14 网格双线性上采样叠加。明示：这是"读出状态强度图"，
     不是注意力图（模型 forward 只输出均值池化后的 fused，未暴露逐 patch
     状态，故在可视化侧以同一 matcher 权重复刻 _seq_forward 的前向分支）。
  3) t-SNE：对指定 5 个测试类的查询图像（每类默认 20 张），抽取 SSMMatch 的
     融合嵌入（真类上下文 fused 向量）与冻结骨干 cosine 基线特征（f2 token
     均值池化），各自 t-SNE（固定 perplexity=30、init=pca、random_state=0）
     到 2D 并排画图，silhouette 系数（cosine 度量）标注在子图标题。

类别 ID 与图像路径从 --data_root + split 文件读取（复用现有 data 模块）。
必须 --ckpt 加载模型；输出到 docs/figures_generated/。
无 GPU 时提示后可用 CPU 运行（自动调小骨干前向分块；真实 mamba_ssm 前向
通常仍需 CUDA，CPU 路径主要用于带 stub 的流程验证）。

用法：
    python tools/visualize.py --ckpt output/best.pt \
        --data_root data/mini-imagenet --model_dir /path/to/vim-tiny-midclstok
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser(
        description="SSM-Match 可视化（回应审稿意见 #R3-7：无注意力图的替代证据）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--ckpt", required=True, help="训练好的 SSMMatch 检查点（必须）")
    p.add_argument("--model_dir", default=".", help="Vim 权重目录")
    p.add_argument("--vim_ckpt", default="vim_t_midclstok_ft_78p3acc.pth")
    p.add_argument("--dataset", default="mini", choices=["mini", "class-folder"])
    p.add_argument("--data_root", required=True, help="图像数据根目录")
    p.add_argument("--splits_dir", default=os.path.join(REPO_ROOT, "data", "splits", "mini"))
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--k_shot", type=int, default=5,
                   help="episode 支持数（读出/上下文更稳，1 亦可）")
    p.add_argument("--n_query", type=int, default=15)
    p.add_argument("--tsne_classes", default="",
                   help="逗号分隔的 5 个测试类 ID；缺省取 test split 字典序前 5 类")
    p.add_argument("--tsne_per_class", type=int, default=20, help="t-SNE 每类查询数")
    p.add_argument("--query_chunk", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=25, help="骨干前向分块（CPU 自动减半）")
    p.add_argument("--out_dir", default=os.path.join(REPO_ROOT, "docs", "figures_generated"))
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def resolve_device(choice: str, batch: int) -> tuple:
    """auto → cuda 优先；无 GPU 提示后用 CPU 并把 batch 减半（返回 device, batch）。"""
    if choice == "cuda" and not torch.cuda.is_available():
        print("[错误] 指定 --device cuda 但无可用 GPU。")
        sys.exit(1)
    if choice == "auto" and not torch.cuda.is_available():
        print("[提示] 无 GPU：改用 CPU 并把 batch 调小。注意真实 mamba_ssm 前向"
              "通常仍需 CUDA，CPU 路径主要用于带 stub 的流程验证。")
        return torch.device("cpu"), max(1, batch // 2)
    dev = torch.device(choice if choice != "auto" else
                       ("cuda" if torch.cuda.is_available() else "cpu"))
    return dev, batch


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
    """(...,3,H,W) → 尺度名→(...,L,C)。VimTiny 仅接受 4D 输入，先展平批维再还原。"""
    lead = images.shape[:-3]
    feats = backbone(images.reshape(-1, *images.shape[-3:]))._asdict()
    return {k: v.reshape(*lead, *v.shape[1:]) for k, v in feats.items()}


def load_images(paths: list, tfm, device) -> torch.Tensor:
    from PIL import Image
    return torch.stack([tfm(Image.open(p).convert("RGB")) for p in paths]).to(device)


def denormalize(img: torch.Tensor) -> np.ndarray:
    """归一化张量 (3,H,W) → 可显示 RGB（反 transform 的 Normalize）。"""
    from ssm_match.data.transforms import MEAN, STD
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return (img.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


def norm01(x) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    rng = float(x.max() - x.min())
    return (x - x.min()) / rng if rng > 0 else x * 0.0


@torch.no_grad()
def query_readout_states(matcher, support: torch.Tensor, query: torch.Tensor,
                         class_idx: int) -> torch.Tensor:
    """复刻 CrossImageMatcher._seq_forward 的前向扫描分支，但保留查询区逐 patch 状态。

    模型 forward 只输出均值池化后的 fused，未暴露逐 patch 隐藏态，故在可视化侧
    以同一 matcher 实例的权重复刻前向（仅前向分支：支持条件化的查询表示）；
    需与 ssm_match/models/matching.py 的 _seq_forward 保持同步（尊重
    use_prior/use_sep 消融开关）。scan_direction='bwd' 时模型没有前向查询区
    读出，读出强度图无定义，直接报错提示。返回 (m,L,dim)。
    """
    if matcher.scan_direction == "bwd":
        raise ValueError("scan_direction='bwd' 时不存在前向查询区读出，"
                         "读出状态强度图未定义（仅支持 bi/fwd）")
    n, K, L, _ = support.shape
    m = query.shape[0]
    s = matcher.input_norm(matcher.input_proj(support))       # (n,K,L,dim)
    q = matcher.input_norm(matcher.input_proj(query))         # (m,L,dim)
    pooled = s.mean(dim=(1, 2))
    prior = matcher.prior_proj(pooled)
    sep = matcher.sep.to(s.dtype)
    parts = [prior[class_idx].unsqueeze(0)] if matcher.use_prior else []
    for j in range(K):
        if matcher.use_sep:
            parts += [s[class_idx, j], sep.expand(L, -1)]
        else:
            parts.append(s[class_idx, j])
    prefix = torch.cat(parts, dim=0) if parts else s.new_zeros(0, s.shape[-1])
    T = prefix.shape[0] + L
    states = []
    for lo in range(0, m, matcher.query_chunk):
        qc = q[lo:lo + matcher.query_chunk]
        seq = torch.cat([prefix.unsqueeze(0).expand(qc.shape[0], -1, -1), qc], dim=1)
        states.append(matcher.out_norm(matcher.scan(seq))[:, T - L:, :])  # 查询区
    return torch.cat(states, dim=0)                           # (m,L,dim)


def saliency_maps(model, backbone, feats_s: dict, query: torch.Tensor,
                  labels: torch.Tensor, batch: int) -> torch.Tensor:
    """真类匹配得分对输入查询图像的梯度 saliency → (m,224,224)。

    查询张量 leaf 化并 enable_grad；骨干 frozen（参数 requires_grad=False）但
    允许梯度流到输入——绕过 backbone.forward 的 @torch.no_grad 装饰器直接调
    vim。查询分块独立前向/反向（查询之间无交互，逐块 backward 等价于对全部
    查询的真类得分和做一次 backward），梯度按切片累加进 leaf。
    """
    from ssm_match.backbone.vim_tiny import to_multiscale
    qry = query.detach().requires_grad_(True)                 # leaf 化
    m = qry.shape[0]
    for lo in range(0, m, batch):
        chunk = qry[lo:lo + batch]
        with torch.enable_grad():
            fq = {k: v.unsqueeze(0) for k, v in
                  to_multiscale(backbone.vim(chunk))._asdict().items()}
            out = model(feats_s, fq)
            idx = torch.arange(chunk.shape[0], device=qry.device)
            score = out["logits"][0, idx, labels[lo:lo + chunk.shape[0]]].sum()
            score.backward()
    return qry.grad.abs().amax(dim=1)                         # 通道维取绝对值最大


def run_episode_figures(model, backbone, args, qry: torch.Tensor,
                        feats_s: dict, feats_q: dict, out_dir: str, batch: int) -> None:
    """图 1/2：对给定 episode 输出匹配显著性图与逐 patch 读出状态强度图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    n, nq, m = args.n_way, args.n_query, qry.shape[0]
    labels = torch.arange(n, device=qry.device).unsqueeze(1).repeat(1, nq).flatten()
    sal = saliency_maps(model, backbone, feats_s, qry, labels, batch)     # (m,H,W)
    print(f"[1/3] 显著性图计算完成: {tuple(sal.shape)}")

    # ---- 逐 patch 读出状态强度图（f2 为 Vim 天然 14×14 网格）----
    matcher = model.matchers["f2"]
    heat = []
    for c in range(n):
        states = query_readout_states(matcher, feats_s["f2"][0], feats_q["f2"][0], c)
        # 每类上下文下全部 m 个查询的读出强度；只取该类自己的 nq 个查询（真类上下文）
        heat.append(states.norm(dim=-1).reshape(m, 14, 14)[c * nq:(c + 1) * nq])
    heat = torch.cat(heat, dim=0)                             # (m,14,14)
    heat = F.interpolate(heat.unsqueeze(1), size=(224, 224), mode="bilinear",
                         align_corners=False).squeeze(1).cpu().numpy()
    print("[2/3] 读出状态强度图计算完成（明示：读出强度，非注意力图）")

    # ---- 渲染：总览（全部查询的显著性叠加）+ 细节（每类首查询 三联图）----
    sal_np = norm01(sal.cpu().numpy())
    heat_np = norm01(heat)
    fig, axes = plt.subplots(n, nq, figsize=(nq * 1.4, n * 1.55))
    axes = np.asarray(axes).reshape(n, nq)                    # nq=1 时 subplots 退化为一维
    for c in range(n):
        for j in range(nq):
            ax = axes[c, j]
            ax.imshow(denormalize(qry[c * nq + j]))
            ax.imshow(sal_np[c * nq + j], cmap="jet", alpha=0.45)
            ax.set_xticks([])
            ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(f"class {c}", fontsize=7)
    fig.suptitle("Matching saliency: |grad| of true-class score wrt input "
                 "(channel-max), overlaid")
    fig.tight_layout()
    p1 = os.path.join(out_dir, "saliency_grid.png")
    fig.savefig(p1, dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(n, 3, figsize=(8.4, n * 2.1))
    for c in range(n):
        i = c * nq
        panels = (
            lambda ax: ax.imshow(denormalize(qry[i])),
            lambda ax: (ax.imshow(denormalize(qry[i])),
                        ax.imshow(sal_np[i], cmap="jet", alpha=0.45)),
            lambda ax: (ax.imshow(denormalize(qry[i])),
                        ax.imshow(heat_np[i], cmap="viridis", alpha=0.5)))
        for j, draw in enumerate(panels):
            ax = axes[c, j]
            draw(ax)
            ax.set_xticks([])
            ax.set_yticks([])
            if c == 0:
                ax.set_title(("query", "saliency overlay",
                              "readout intensity (NOT attention)")[j], fontsize=8)
    fig.suptitle("Per-class first query: input-gradient saliency vs query-region "
                 "readout state intensity")
    fig.tight_layout()
    p2 = os.path.join(out_dir, "readout_maps.png")
    fig.savefig(p2, dpi=160)
    plt.close(fig)
    print(f"      已保存 {p1} / {p2}")


def run_tsne_figure(model, backbone, args, tfm, device, batch,
                    test_classes: dict, out_dir: str) -> None:
    """图 3：SSM-Match 融合嵌入 vs 冻结骨干 cosine 基线的 t-SNE 并排对比。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    os.makedirs(out_dir, exist_ok=True)
    names = sorted(test_classes)
    if args.tsne_classes:
        names = [c.strip() for c in args.tsne_classes.split(",") if c.strip()]
    assert len(names) == args.n_way, f"t-SNE 需要 {args.n_way} 个类，得到 {names}"
    rng = np.random.default_rng(args.seed + 2)
    per = args.tsne_per_class

    sup_paths, qry_paths, q_labels = [], [], []
    for ci, cname in enumerate(names):
        imgs = test_classes[cname]
        need = args.k_shot + per
        if len(imgs) < need:
            raise ValueError(f"类 {cname} 只有 {len(imgs)} 张图，不足 {need}")
        idx = rng.choice(len(imgs), need, replace=False)
        sup_paths += [imgs[i] for i in idx[:args.k_shot]]
        qry_paths += [imgs[i] for i in idx[args.k_shot:]]
        q_labels += [ci] * per
    q_labels = np.array(q_labels)

    sup = load_images(sup_paths, tfm, device).reshape(
        args.n_way, args.k_shot, 3, 224, 224)
    with torch.no_grad():
        feats_s = backbone_features(backbone, sup.unsqueeze(0))
    fused_vecs, base_vecs = [], []
    with torch.no_grad():
        for lo in range(0, len(qry_paths), batch):
            q = load_images(qry_paths[lo:lo + batch], tfm, device)
            fq = backbone_features(backbone, q.unsqueeze(0))
            out = model(feats_s, fq)
            # 融合嵌入：fused 本身即查询读出在真类上下文下的均值池化向量；各尺度
            # 维度不同（160/256/320）不可直接平均，故各尺度 L2 归一化后拼接
            per_scale = [F.normalize(fu[0], dim=-1) for fu in out["fused"]]
            fused_vecs.append(torch.cat(per_scale, dim=-1))              # (n,m,D)
            base_vecs.append(F.normalize(fq["f2"].mean(dim=2)[0], dim=-1))  # (m,192)
    fused = torch.cat(fused_vecs, dim=1)                                 # (n,m_total,D)
    vecs = fused[q_labels, np.arange(len(q_labels))]                     # 真类上下文向量
    base = torch.cat(base_vecs, dim=0)
    print(f"[3/3] 嵌入抽取完成: fused{tuple(vecs.shape)} baseline{tuple(base.shape)}")

    tsne = dict(n_components=2, perplexity=30, init="pca", random_state=0)
    emb_f = TSNE(**tsne).fit_transform(vecs.cpu().numpy())
    emb_b = TSNE(**tsne).fit_transform(base.cpu().numpy())
    s_f = silhouette_score(vecs.cpu().numpy(), q_labels, metric="cosine")
    s_b = silhouette_score(base.cpu().numpy(), q_labels, metric="cosine")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, emb, s, ttl in (
            (axes[0], emb_f, s_f,
             f"SSM-Match fused embedding (silhouette={s_f:.3f})"),
            (axes[1], emb_b, s_b,
             f"Frozen Vim-Tiny cosine baseline (silhouette={s_b:.3f})")):
        for ci in range(args.n_way):
            sel = q_labels == ci
            ax.scatter(emb[sel, 0], emb[sel, 1], s=14, alpha=0.8, label=f"c{ci}")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(fontsize=7, markerscale=0.8)
    fig.suptitle(f"t-SNE (perplexity=30, init=pca, random_state=0), "
                 f"{len(names)} test classes x {per} queries")
    fig.tight_layout()
    p3 = os.path.join(out_dir, "tsne_silhouette.png")
    fig.savefig(p3, dpi=160)
    plt.close(fig)
    print(f"      已保存 {p3}（silhouette: fused={s_f:.3f}, baseline={s_b:.3f}）")
    with open(os.path.join(out_dir, "tsne_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"classes": names, "per_class": per,
                   "tsne": {"perplexity": 30, "init": "pca", "random_state": 0},
                   "silhouette_fused_cosine": float(s_f),
                   "silhouette_baseline_cosine": float(s_b)},
                  f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    try:
        run(args)
    except ImportError as e:
        print(f"[错误] 缺少依赖：{e}（matplotlib/sklearn/torchvision/mamba_ssm 均需要）")
        sys.exit(1)
    except FileNotFoundError as e:
        print(f"[错误] 文件/目录未找到：{e}")
        sys.exit(1)
    except RuntimeError as e:
        print(f"[错误] 运行失败（常见原因：检查点与当前 n_way/query_chunk 不一致，"
              f"或真实 mamba_ssm 前向需要 GPU）：{e}")
        sys.exit(1)


def run(args):
    device, batch = resolve_device(args.device, args.batch_size)
    torch.manual_seed(args.seed)

    import matplotlib  # noqa: F401  绘图依赖确认（Agg 后端在各绘制函数内设置）
    import sklearn  # noqa: F401
    import torchvision  # noqa: F401

    from ssm_match.backbone import FrozenVimBackbone
    from ssm_match.data import EpisodeSampler, build_transform
    from ssm_match.engine.checkpoint import load_checkpoint
    from ssm_match.models import SSMMatch

    vim_path = os.path.join(args.model_dir, args.vim_ckpt)
    if not os.path.isfile(vim_path):
        print(f"[错误] 未找到 Vim 权重 {vim_path}（--model_dir/--vim_ckpt）")
        sys.exit(1)
    backbone = FrozenVimBackbone(vim_path).to(device)

    model = SSMMatch(n_way=args.n_way, query_chunk=args.query_chunk).to(device)
    load_checkpoint(args.ckpt, model, restore_rng=False)   # 不恢复 RNG：可视化自带种子
    model.eval()
    for p in model.parameters():                           # 冻结参数：梯度只流向查询输入
        p.requires_grad_(False)
    print(f"[ckpt] 已加载 {args.ckpt}")

    test_classes = build_test_class_images(args)
    os.makedirs(args.out_dir, exist_ok=True)
    tfm = build_transform(train=False)

    # 采样一个 episode，图 1/2 共用同一批支持/查询图像
    sampler = EpisodeSampler(test_classes, args.n_way, args.k_shot, args.n_query,
                             seed=args.seed)
    support_paths, query_paths, _ = sampler.sample_episode()
    sup = load_images([p for row in support_paths for p in row], tfm, device)
    sup = sup.reshape(args.n_way, args.k_shot, 3, 224, 224)
    qry = load_images([p for row in query_paths for p in row], tfm, device)
    with torch.no_grad():
        feats_s = backbone_features(backbone, sup.unsqueeze(0))
        feats_q = backbone_features(backbone, qry.unsqueeze(0))
    print(f"episode 就绪: {args.n_way}-way {args.k_shot}-shot, m={qry.shape[0]} 查询")

    run_episode_figures(model, backbone, args, qry, feats_s, feats_q,
                        args.out_dir, batch)
    run_tsne_figure(model, backbone, args, tfm, device, batch, test_classes,
                    args.out_dir)
    print(f"\n全部图已输出到 {args.out_dir}")


if __name__ == "__main__":
    main()

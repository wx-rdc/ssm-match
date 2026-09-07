"""效率实测脚本 —— 该脚本回应审稿意见 #R3-6 / #R4-3（效率数据应为实测值）：
"Instead of referring to theoretical values for the efficiency of the proposed
system, measured values for the total number of parameters, number of trainable
parameters, number of FLOPs, memory usage on the GPU, and the latency of the
entire pipeline should be provided. It also should be clarified whether the
memory values for the single-scale and multi-scale cases are compatible."

对以下匹配器变体逐一实测（--shot 可给多种 episode 配置，默认 1-shot 与 5-shot）：
  full        三尺度 SSMMatch（f1+f2+f3）
  single-f1/2/3  仅激活一个尺度的 SSMMatch（临时收窄模块级 SCALES/SCALE_SPECS，
              不改模型源码，参数量/显存/FLOPs 与实际计算严格一致）
  cross-attn  同容量显式交叉注意力匹配基线（本脚本内实现 CrossAttentionMatcher，
              逐层对齐 CrossImageMatcher：input_proj/norm → 逐类 multihead
              attention（support 为 K/V、query 为 Q）→ 均值读出 → score_head，
              尺度金字塔与凸组合复用 SSMMatch，保证除匹配器外完全一致）

每个变体输出：匹配器总参数量与可训练参数量（冻结骨干不计入可训练，Vim-Tiny
参数单独统计）、每 episode 前向+反向峰值显存 torch.cuda.max_memory_allocated、
推理延迟（预热 10 次后计时 50 次取 mean±std）、吞吐（episodes/s）、FLOPs
（fvcore 或 thop 可用则实测；都不可用则打印"未安装依赖"并给出解析估算，
估算值在结果中显式标注 estimate）。另实测冻结骨干特征抽取延迟与端到端延迟。

profile 用随机特征张量即可（不需要真实图像），但形状按真实 episode
(n_way=5, K, L_k, C_k)；--ckpt 可加载训练好的匹配器权重（经
engine/checkpoint.py 的 load_checkpoint 恢复 SSMMatch state_dict）。

用法（GPU 机器）：
    python tools/profile_efficiency.py --shot 1 5 --output output/profile.json
    python tools/profile_efficiency.py --ckpt output/best.pt --shot 1
"""

import argparse
import contextlib
import copy
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 三尺度特征的真实形状 (L_k, C_k)，来自 backbone/vim_tiny.to_multiscale
SCALE_SHAPES = {"f1": (28 * 28, 48), "f2": (14 * 14, 192), "f3": (7 * 7, 768)}
# 变体定义：(名称, 激活尺度, 种类)；ssm=收窄尺度的 SSMMatch，attn=交叉注意力对照
VARIANTS = (
    ("full", ("f1", "f2", "f3"), "ssm"),
    ("single-f1", ("f1",), "ssm"),
    ("single-f2", ("f2",), "ssm"),
    ("single-f3", ("f3",), "ssm"),
    ("cross-attn", ("f1", "f2", "f3"), "attn"),
)
AUX_WEIGHT = 0.1  # 与训练路径一致：CE + 0.1·InfoNCE，显存/延迟按真实训练口径


@contextlib.contextmanager
def narrowed_scales(scales: tuple):
    """临时收窄 ssm_match.models.ssm_match 的模块级 SCALES/SCALE_SPECS，退出时恢复。

    SSMMatch.__init__ 与 forward 都在运行时读取这两个模块级全局，收窄后模型只构建
    并计算指定尺度（单尺度变体的参数量/显存/FLOPs 因此与实际计算一致），全程不改
    模型源码。注意 SSMMatch.forward 以 support["f2"] 取 batch 维，故输入特征字典
    仍须含 f2 键——本脚本始终喂入完整三尺度字典，收窄只影响匹配器实际计算的尺度。
    """
    import ssm_match.models.ssm_match as M
    old_scales, old_specs = M.SCALES, M.SCALE_SPECS
    M.SCALES = tuple(scales)
    M.SCALE_SPECS = {k: old_specs[k] for k in scales}
    try:
        yield
    finally:
        M.SCALES, M.SCALE_SPECS = old_scales, old_specs


class CrossAttentionMatcher(nn.Module):
    """同容量显式交叉注意力匹配基线（对照实验用，结构与 CrossImageMatcher 对齐）。

    逐层对应：input_proj + input_norm（同维度投影）→ 逐类标准 multihead attention
    （support tokens 为 K/V、query tokens 为 Q）→ out_norm → 查询 token 均值读出
    → score_head 打分；prior_proj 同样保留以对齐容量。forward 签名与
    CrossImageMatcher 完全一致：(n,K,L,d_in), (m,L,d_in) → (n,m), (n,m,dim)。
    """

    def __init__(self, d_in: int, dim: int, n_heads: int = 4):
        super().__init__()
        self.dim = dim
        self.input_proj = nn.Linear(d_in, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.prior_proj = nn.Linear(dim, dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.score_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.score_head.bias)

    def _seq_forward(self, s: torch.Tensor, q: torch.Tensor) -> tuple:
        # s (K,L,dim)，q (m,L,dim) → (scores(m), fused(m,dim))
        # 每张查询图的 L 个 token 对该类全部 K·L 个支持 token 做交叉注意力
        kv = s.reshape(-1, s.shape[-1]).unsqueeze(0) \
            .expand(q.shape[0], -1, -1)                       # (m,K·L,dim)
        att, _ = self.attn(q, kv, kv)                          # (m,L,dim)
        fused = self.out_norm(att).mean(dim=1)                 # (m,dim) 均值读出
        return self.score_head(fused).squeeze(-1), fused

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> tuple:
        n, K, L, _ = support.shape
        m = query.shape[0]
        s = self.input_norm(self.input_proj(support))
        q = self.input_norm(self.input_proj(query))
        prior = self.prior_proj(s.mean(dim=(1, 2)))              # (n,dim) 容量对齐保留
        scores = s.new_empty(n, m)
        fused = s.new_empty(n, m, self.dim)
        for ci in range(n):
            sc, fu = self._seq_forward(s[ci], q)
            scores[ci], fused[ci] = sc, fu
        return scores, fused


def import_models():
    """延迟导入 ssm_match.models（依赖真实 mamba_ssm），失败时优雅退出而非崩溃。"""
    try:
        import ssm_match.models.ssm_match as M
        from ssm_match.models import SSMMatch
        return SSMMatch, M
    except Exception as e:
        print(f"[错误] 无法导入 ssm_match.models（需要 mamba_ssm/causal-conv1d 等 "
              f"CUDA 依赖，且通常需要 GPU）：{e}")
        sys.exit(1)


def build_model(kind: str, scales: tuple, n_way: int, query_chunk: int):
    """kind='ssm' → 收窄尺度的 SSMMatch；kind='attn' → 同结构仅替换逐尺度匹配器。

    注意力版复用 SSMMatch.forward（尺度凸组合、类先验、对比头逻辑完全一致，
    且 CrossAttentionMatcher 提供同名的 input_proj/input_norm/dim 属性），
    保证除匹配器本体外的所有计算与 SSMMatch 逐位对齐，对比才公平。
    """
    SSMMatch, M = import_models()

    if kind == "attn":
        with narrowed_scales(scales):
            model = SSMMatch(n_way=n_way, query_chunk=query_chunk, scales=scales)
            for k in list(model.matchers):
                spec = M.SCALE_SPECS[k]
                model.matchers[k] = CrossAttentionMatcher(spec["d_in"], spec["dim"])
        return model

    class NarrowedSSMMatch(SSMMatch):
        """仅以 scales 子集构建与运行的 SSMMatch（自包含，不改模型源码）。

        构建用 SSMMatch 原生 scales 参数；每次 forward 再临时收窄模块级
        SCALES/SCALE_SPECS——当前 ssm_match.forward 以模块级 len(SCALES) 计算
        尺度凸组合（与 self.scales 可能不一致），收窄使两者一致；若模型侧日后
        改为 len(self.scales)，该收窄自动退化为无害冗余。
        """

        def __init__(self):
            with narrowed_scales(scales):
                super().__init__(n_way=n_way, query_chunk=query_chunk, scales=scales)

        def forward(self, support, query):
            with narrowed_scales(scales):
                return super().forward(support, query)

    return NarrowedSSMMatch()


def load_ckpt_into(path: str, model, scales: tuple, kind: str) -> None:
    """经 engine/checkpoint.py 加载训练好的匹配器权重。

    full 变体直接 load_checkpoint 恢复完整 SSMMatch state_dict；单尺度变体的
    state_dict 键集与检查点不同（只含 matchers.k.*），过滤后 strict=False 加载
    （scale_logits 维度随之不同，保留随机初始化并明示）。cross-attn 基线结构
    不同，不加载（它对照的是未训练容量，不算SSM-Match 的成绩）。
    """
    if kind == "attn":
        print("[ckpt] cross-attn 基线与检查点结构不同，跳过加载（随机初始化）")
        return
    if len(scales) == 3:
        from ssm_match.engine.checkpoint import load_checkpoint
        load_checkpoint(path, model, restore_rng=False)
        print(f"[ckpt] 已从 {path} 恢复完整 SSMMatch state_dict")
        return
    payload = __import__("torch").load(path, map_location="cpu", weights_only=False)
    sd = payload["model"] if "model" in payload else payload
    prefix = tuple(f"matchers.{k}." for k in scales)
    sub = {kk: vv for kk, vv in sd.items() if kk.startswith(prefix)}
    missing, unexpected = model.load_state_dict(sub, strict=False)
    assert not unexpected, f"检查点出现多余键: {unexpected[:4]}"
    print(f"[ckpt] 已加载 matchers.{scales[0]} 权重 {len(sub)} 个张量"
          f"（缺 {len(missing)} 个收窄后不存在的键，如 scale_logits）")


def make_random_episode(n_way: int, k_shot: int, n_query: int, device):
    """随机特征张量按真实 episode 形状 (1,n,K,L_k,C_k)；另出随机图像张量供骨干计时。"""
    sup = {k: torch.randn(1, n_way, k_shot, L, C, device=device)
           for k, (L, C) in SCALE_SHAPES.items()}
    m = n_way * n_query
    qry = {k: torch.randn(1, m, L, C, device=device)
           for k, (L, C) in SCALE_SHAPES.items()}
    labels = torch.arange(n_way, device=device).unsqueeze(1) \
        .repeat(1, n_query).flatten()                          # (m,)
    imgs_sup = torch.randn(n_way * k_shot, 3, 224, 224, device=device)
    imgs_qry = torch.randn(m, 3, 224, 224, device=device)
    return sup, qry, labels, imgs_sup, imgs_qry


def train_step_loss(model, sup, qry, labels, n_way: int) -> torch.Tensor:
    """与 trainer._run_epoch 相同的损失：CE + 0.1·对比辅助（决定反向显存口径）。"""
    out = model(sup, qry)
    loss = F.cross_entropy(out["logits"].reshape(-1, n_way), labels)
    if AUX_WEIGHT > 0:
        loss = loss + AUX_WEIGHT * model.contrastive_loss(out["fused"], out["protos"])
    return loss


def measure_peak_memory(model, sup, qry, labels, n_way: int, device) -> float | None:
    """每 episode 计算图前向+反向的峰值显存（GB）。输入张量就位后再清零统计，
    峰值含 episode 输入特征、匹配器权重与全部前向+反向激活。"""
    if device.type != "cuda":
        return None
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    loss = train_step_loss(model, sup, qry, labels, n_way)
    loss.backward()
    torch.cuda.synchronize()
    model.zero_grad(set_to_none=True)
    return torch.cuda.max_memory_allocated() / 1024 ** 3


def measure_latency(model, sup, qry, warmup: int, iters: int, device) -> tuple:
    """推理延迟：预热 warmup 次后计时 iters 次，返回 (mean_ms, std_ms, eps_per_s)。"""
    model.eval()
    is_cuda = device.type == "cuda"

    def one():
        t0 = time.perf_counter()
        with torch.no_grad():
            model(sup, qry)
        if is_cuda:
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0

    for _ in range(warmup):
        one()
    times = [one() for _ in range(iters)]
    total_s = sum(times) / 1000.0
    mean = statistics.mean(times)
    std = statistics.stdev(times) if len(times) > 1 else 0.0
    return mean, std, iters / total_s


class _FlatInputWrapper(nn.Module):
    """把 dict 输入摊平为固定顺序的张量入参，便于 fvcore/thop 追踪。

    子模块顺序：support[f1..], query[f1..]；SSMMatch.forward 需要字典含 "f2"
    键取 batch 维，单尺度时以激活尺度的张量别名补齐（仅读 shape[0]）。
    """

    def __init__(self, model, scales: tuple):
        super().__init__()
        self.model = model
        self.scales = tuple(scales)

    def forward(self, *tensors):
        sup, qry, it = {}, {}, iter(tensors)
        for k in self.scales:
            sup[k] = next(it)
        for k in self.scales:
            qry[k] = next(it)
        sup.setdefault("f2", sup[self.scales[0]])
        qry.setdefault("f2", qry[self.scales[0]])
        return self.model(sup, qry)["logits"]


def flops_via_fvcore(module: nn.Module, tensors: list) -> int:
    from fvcore.nn import FlopCountAnalysis
    fca = FlopCountAnalysis(module, tensors)
    fca.unsupported_ops_warnings(False)
    fca.uncalled_modules_warnings(False)
    return int(fca.total())


def flops_via_thop(module: nn.Module, tensors: list) -> int:
    from thop import profile
    flops, _ = profile(copy.deepcopy(module), inputs=tensors, verbose=False)
    return int(flops)


def estimate_matcher_flops(kind: str, scales: tuple, n_way: int,
                           k_shot: int, n_query: int) -> int:
    """匹配器单 episode FLOPs 解析估算（结果标注为估算值，非实测）。

    乘加计 2 FLOPs，仅主项；对 ssm 变体：
      - input_proj：2·d_in·dim·L·(n·K + m)；
      - Mamba 扫描：单方向 ≈ 8·dim²/token（in_proj/out_proj 主项，x_proj/dt_proj/
        selective_scan 核低阶项忽略）；matcher 显式正/逆两路调用 ×2，
        bimamba_type="v" 内部再双向 ×2 → 共 4·8·dim²·m·T·n，T = 1+K(L+1)+L
        （查询分块重复前缀的开销与之同阶，按连续近似并入）；
      - prior_proj / dir_gate / evi_gate / score_head 为低阶项，逐项计入。
    对 attn 变体：投影项同上，注意力主项 = QK^T 与 ·V 各
    2·dim·(m·L)·(K·L)，MHA in_proj/out_proj 按 m·L 与 n·K·L token 数计。
    """
    import ssm_match.models.ssm_match as M
    m = n_way * n_query
    total = 0
    for k in scales:
        L, C = SCALE_SHAPES[k]
        dim = M.SCALE_SPECS[k]["dim"]
        T = 1 + k_shot * (L + 1) + L
        total += 2 * C * dim * L * (n_way * k_shot + m)          # input_proj
        total += 2 * dim * dim * n_way                           # prior_proj
        total += 2 * dim * m * n_way                             # score_head
        if kind == "ssm":
            total += 4 * 8 * dim * dim * m * T * n_way           # 双路×双向 Mamba 主项
            total += 2 * (2 * dim * 2) * m * n_way               # dir_gate (2dim→2)
            total += 2 * (2 * dim * dim) * m * n_way             # evi_gate (2dim→dim)
        else:
            total += n_way * 2 * 3 * dim * dim * (m * L + k_shot * L)  # MHA qkv 投影
            total += n_way * 2 * dim * dim * m * L               # MHA out_proj
            total += n_way * 4 * dim * m * k_shot * L * L        # QK^T + ·V
    return total


def measure_flops(model, kind: str, scales: tuple, n_way: int, k_shot: int,
                  n_query: int, device) -> tuple:
    """返回 (flops, source)。优先 fvcore 实测，其次 thop，都失败给解析估算。"""
    tensors = [torch.randn(1, n_way, k_shot, L, C) for _, (L, C)
               in [(k, SCALE_SHAPES[k]) for k in scales]]
    tensors += [torch.randn(1, n_way * n_query, L, C) for _, (L, C)
                in [(k, SCALE_SHAPES[k]) for k in scales]]
    target: nn.Module = model
    if kind == "ssm":
        target = _FlatInputWrapper(model, scales)
    for name, fn in (("fvcore", flops_via_fvcore), ("thop", flops_via_thop)):
        try:
            return fn(target.cpu(), tensors), name
        except Exception as e:
            print(f"  [flops] {name} 实测失败（{type(e).__name__}: {e}），尝试下一方案")
        finally:
            model.to(device)
    est = estimate_matcher_flops(kind, scales, n_way, k_shot, n_query)
    print(f"  [flops] fvcore/thop 均不可用或失败：跳过 FLOPs 实测，"
          f"给出解析估算 {est / 1e9:.2f} G（标注为估算值）")
    return est, "estimate(解析近似)"


def count_params(model) -> tuple:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def fmt_m(x: float | None) -> str:
    return "—" if x is None else f"{x / 1e6:.2f}M"


def main():
    ap = argparse.ArgumentParser(
        description="SSM-Match 效率实测（回应审稿意见 #R3-6/#R4-3）")
    ap.add_argument("--shot", type=int, nargs="+", default=[1, 5],
                    help="episode 的 k-shot 配置，可多个（默认 1 5）")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--n_query", type=int, default=15)
    ap.add_argument("--query_chunk", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=10, help="延迟预热情节数")
    ap.add_argument("--iters", type=int, default=50, help="延迟计时节节数")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                    help="默认要求 CUDA（显存指标仅 CUDA 可测）；显式 cpu 可跑通流程")
    ap.add_argument("--ckpt", default=None, help="训练好的 SSMMatch 检查点（可选）")
    ap.add_argument("--model_dir", default=None,
                    help="Vim 权重目录（可选：仅影响骨干计时是否加载预训练权重）")
    ap.add_argument("--vim_ckpt", default="vim_t_midclstok_ft_78p3acc.pth")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", default=None, help="结果 JSON 保存路径")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[错误] 未检测到 CUDA GPU：显存峰值/延迟等效率指标需要 GPU 实测。"
              "请在 GPU 机器（如启智云脑）上运行，或显式 --device cpu 仅做流程验证。")
        sys.exit(1)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    SSMMatch, M = import_models()

    # 冻结骨干 Vim-Tiny 参数单独统计（SSMMatch 不含骨干，检查点也不存它）。
    # 未给 --model_dir 时用随机初始化：效率只取决于计算图/参数量，与权重值无关。
    from ssm_match.backbone.vim_tiny import VimTiny, to_multiscale
    vim = VimTiny().eval()
    if args.model_dir:
        vim.load_pretrained(os.path.join(args.model_dir, args.vim_ckpt))
        vim_weights = f"pretrained({args.vim_ckpt})"
    else:
        vim_weights = "random-init(效率与权重值无关)"
    for p in vim.parameters():
        p.requires_grad_(False)
    vim_total, _ = count_params(vim)
    print(f"[backbone] Vim-Tiny 参数 {fmt_m(vim_total)}（冻结，不计入可训练；权重: {vim_weights}）")

    results = {
        "meta": {
            "review": "R3-6/R4-3 效率实测",
            "torch": torch.__version__,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "backbone_params": vim_total,
            "backbone_weights": vim_weights,
            "n_way": args.n_way, "n_query": args.n_query,
            "query_chunk": args.query_chunk,
            "warmup": args.warmup, "iters": args.iters, "seed": args.seed,
            "ckpt": os.path.abspath(args.ckpt) if args.ckpt else None,
        },
        "variants": [],
    }

    for k_shot in args.shot:
        sup, qry, labels, imgs_sup, imgs_qry = make_random_episode(
            args.n_way, k_shot, args.n_query, device)

        # 骨干特征抽取延迟（冻结 no_grad，所有变体共享；与 matcher 延迟相加=端到端）
        def backbone_once():
            t0 = time.perf_counter()
            with torch.no_grad():
                to_multiscale(vim(imgs_sup))
                to_multiscale(vim(imgs_qry))
            if device.type == "cuda":
                torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000.0

        for _ in range(args.warmup):
            backbone_once()
        bb_times = [backbone_once() for _ in range(args.iters)]
        bb_mean = statistics.mean(bb_times)
        bb_std = statistics.stdev(bb_times) if len(bb_times) > 1 else 0.0
        print(f"\n==== 5-way {k_shot}-shot（m={args.n_way * args.n_query} 查询）====")
        print(f"骨干特征抽取（冻结，共享）: {bb_mean:.1f}±{bb_std:.1f} ms/episode")

        rows = []
        for name, scales, kind in VARIANTS:
            model = build_model(kind, scales, args.n_way, args.query_chunk)
            if args.ckpt:
                load_ckpt_into(args.ckpt, model, scales, kind)
            model.to(device)
            p_total, p_train = count_params(model)
            peak = measure_peak_memory(model, sup, qry, labels, args.n_way, device)
            lat_mean, lat_std, tput = measure_latency(
                model, sup, qry, args.warmup, args.iters, device)
            flops, fsrc = measure_flops(
                model, kind, scales, args.n_way, k_shot, args.n_query, device)
            model.cpu()
            e2e = lat_mean + bb_mean
            rows.append({
                "variant": name, "scales": list(scales), "kind": kind,
                "params_matcher": p_total, "params_trainable": p_train,
                "peak_mem_gb": peak,
                "latency_ms_mean": lat_mean, "latency_ms_std": lat_std,
                "throughput_eps": tput,
                "flops": flops, "flops_source": fsrc,
                "end2end_ms_mean": e2e,
                "backbone_ms_mean": bb_mean, "backbone_ms_std": bb_std,
            })
            mem_s = f"{peak:.2f}" if peak is not None else "—"
            print(f"  {name:<11} 参数 {fmt_m(p_total):>6} | 显存峰值 {mem_s:>6} GB | "
                  f"延迟 {lat_mean:7.1f}±{lat_std:5.1f} ms | 吞吐 {tput:6.2f} eps/s | "
                  f"FLOPs {flops / 1e9:7.2f} G ({fsrc})")

        results.setdefault("shots", {})[str(k_shot)] = {
            "backbone_ms": {"mean": bb_mean, "std": bb_std},
            "rows": rows,
        }
        del sup, qry, imgs_sup, imgs_qry
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print_markdown(results)
    if args.output:
        parent = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(parent, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n结果 JSON 已保存: {args.output}")


def print_markdown(results: dict) -> None:
    """把实测结果打成 markdown 表格（可直接粘进论文 IV-F 效率表）。"""
    meta = results["meta"]
    print("\n---- markdown ----")
    print(f"（设备: {meta['gpu']}，torch {meta['torch']}，warmup {meta['warmup']}，"
          f"iters {meta['iters']}；骨干 Vim-Tiny {fmt_m(meta['backbone_params'])} 冻结，"
          f"单列统计不计入可训练）")
    for shot, block in results.get("shots", {}).items():
        print(f"\n**5-way {shot}-shot**\n")
        print("| 变体 | 匹配器参数 | 可训练参数 | 峰值显存(前向+反向,GB) "
              "| 推理延迟(ms, mean±std) | 端到端延迟(ms) | 吞吐(episodes/s) | FLOPs(G) |")
        print("|---|---|---|---|---|---|---|---|")
        for r in block["rows"]:
            mem = "—" if r["peak_mem_gb"] is None else f"{r['peak_mem_gb']:.2f}"
            print(f"| {r['variant']} | {fmt_m(r['params_matcher'])} "
                  f"| {fmt_m(r['params_trainable'])} | {mem} "
                  f"| {r['latency_ms_mean']:.1f}±{r['latency_ms_std']:.1f} "
                  f"| {r['end2end_ms_mean']:.1f} "
                  f"| {r['throughput_eps']:.2f} "
                  f"| {r['flops'] / 1e9:.2f} ({r['flops_source']}) |")
        bb = block["backbone_ms"]
        print(f"\n骨干特征抽取（冻结，所有变体共享）: "
              f"{bb['mean']:.1f}±{bb['std']:.1f} ms/episode；"
              f"端到端 = 骨干 + 匹配器推理。单尺度与三尺度显存均含同一份输入特征，"
              f"差异只来自匹配器激活（回应'显存数值是否 compatible'）。")


if __name__ == "__main__":
    main()

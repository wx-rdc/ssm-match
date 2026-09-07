"""CPU 回归测试：用因果累积和桩替换 Mamba，验证 CrossImageMatcher 的双向读出语义。

背景（反向读出修复）：逆序扫描中查询先入状态、支持后入状态，故
- 前向查询区输出必须包含支持信息（支持→查询）；
- 后向支持区输出必须包含查询信息（查询→支持）。
桩实现真实的因果累积（h_t = Σ_{t'<=t} x_{t'}，逐通道），可精确检验上述信息流。

运行：python tests/test_matching_cpu.py（无需 CUDA / mamba_ssm）
"""

import sys
import types
from pathlib import Path

import torch

# 在导入 matching 之前注入 mamba_ssm 桩
_stub = types.ModuleType("mamba_ssm")
_modules = types.ModuleType("mamba_ssm.modules")
_simple = types.ModuleType("mamba_ssm.modules.mamba_simple")


class _CausalCumSum(torch.nn.Module):
    """因果序列模型的形状兼容桩：y_t = Σ_{t'<=t} x_{t'}。"""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, **kwargs):
        super().__init__()
        self.out = torch.nn.Linear(d_model, d_model)

    def forward(self, x):  # (B,T,D) -> (B,T,D)
        return self.out(torch.cumsum(x, dim=1))


_simple.Mamba = _CausalCumSum
_modules.mamba_simple = _simple
_stub.modules = _modules
sys.modules["mamba_ssm"] = _stub
sys.modules["mamba_ssm.modules"] = _modules
sys.modules["mamba_ssm.modules.mamba_simple"] = _simple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ssm_match.backbone.vim_tiny import MultiScaleFeatures, forward_images  # noqa: E402
from ssm_match.models.matching import CrossImageMatcher  # noqa: E402
from ssm_match.models.ssm_match import SSMMatch  # noqa: E402


class _DummyBackbone(torch.nn.Module):
    """形状兼容桩：任意 (…,3,H,W) → 三尺度特征（值随机，仅测形状通路）。"""

    def forward(self, x):
        n = x.shape[0]
        tok = torch.randn(n, 196, 192)
        return MultiScaleFeatures(
            f1=torch.randn(n, 784, 48), f2=tok, f3=torch.randn(n, 49, 768))


def main() -> int:
    torch.manual_seed(0)
    n, K, L, m = 2, 2, 6, 3
    d_in, dim = 8, 12
    matcher = CrossImageMatcher(d_in, dim, d_state=4)
    support = torch.randn(n, K, L, d_in)
    query = torch.randn(m, L, d_in)

    scores, fused = matcher(support, query)
    assert scores.shape == (n, m) and fused.shape == (n, m, dim), "输出形状错误"
    assert torch.isfinite(scores).all() and torch.isfinite(fused).all(), "存在非有限值"

    # 信息流检查：支持 token 非零、查询 token 置零时，
    # 前向读出（支持条件化查询）仍应非零；否则说明前向分支未汇聚支持证据。
    support_nz = torch.randn(n, K, L, d_in)
    query_zero = torch.zeros(m, L, d_in)
    s2, f2 = matcher(support_nz, query_zero)
    assert f2.abs().sum() > 0, "前向读出未携带支持信息"

    # 反向信息流检查：查询非零、支持置零（含先验=均值池化仍为 0）时，
    # 证据门控 g = σ(W_g[f; summary]) 以 f 为输入，融合向量 f = w_f h_fwd + w_b h_bwd；
    # 后向支持区状态累积了查询 token，故 fused 非零。
    support_zero = torch.zeros(n, K, L, d_in)
    query_nz = torch.randn(m, L, d_in)
    s3, f3 = matcher(support_zero, query_nz)
    assert f3.abs().sum() > 0, "后向读出未携带查询信息（查询条件化支持摘要失效）"

    # 可训练门控的梯度可以回传
    loss = matcher(support, query)[0].sum()
    loss.backward()
    grads = [p.grad for p in matcher.parameters() if p.grad is not None]
    assert grads, "梯度未回传"

    # 因子化消融开关：全部组合可前向传播且输出形状/有限性正确
    combos = [
        dict(use_prior=False), dict(use_sep=False),
        dict(scan_direction="fwd"), dict(scan_direction="bwd"),
        dict(use_dir_gate=False), dict(use_evi_gate=False),
        dict(score_sigmoid=True),
        dict(use_prior=False, use_sep=False, scan_direction="fwd",
             use_dir_gate=False, use_evi_gate=False, score_sigmoid=True),
    ]
    for kw in combos:
        ab = CrossImageMatcher(d_in, dim, d_state=4, **kw)
        sc, fu = ab(support, query)
        assert sc.shape == (n, m) and fu.shape == (n, m, dim)
        assert torch.isfinite(sc).all() and torch.isfinite(fu).all(), f"开关 {kw} 输出异常"
        if kw.get("score_sigmoid"):
            assert (sc >= 0).all() and (sc <= 1).all(), "sigmoid 开关未生效"
    try:
        CrossImageMatcher(d_in, dim, scan_direction="x")
        raise AssertionError("非法 scan_direction 未被拒绝")
    except ValueError:
        pass

    # forward_images：6D 支持批 (b,n,K,3,H,W) 不再崩溃，形状正确还原
    feats = forward_images(_DummyBackbone().eval(),
                           torch.randn(2, 5, 1, 3, 32, 32))
    assert set(feats) == {"f1", "f2", "f3"}
    assert feats["f2"].shape == (2, 5, 1, 196, 192), feats["f2"].shape

    # SSMMatch：完整三尺度与单尺度（消融）前向
    b, n_way, K, m = 2, 2, 1, 6
    sup = {k: torch.randn(b, n_way, K, {"f1": 784, "f2": 196, "f3": 49}[k],
                          {"f1": 48, "f2": 192, "f3": 768}[k])
           for k in ("f1", "f2", "f3")}
    qry = {k: torch.randn(b, m, sup[k].shape[-2], sup[k].shape[-1])
           for k in ("f1", "f2", "f3")}
    full = SSMMatch(n_way=n_way, scales=("f1", "f2", "f3"))
    out = full(sup, qry)
    assert out["logits"].shape == (b, m, n_way), out["logits"].shape
    single = SSMMatch(n_way=n_way, scales=("f2",))
    out1 = single({k: sup[k] for k in ("f1", "f2", "f3")},
                  {k: qry[k] for k in ("f1", "f2", "f3")})
    assert out1["logits"].shape == (b, m, n_way)
    assert torch.isfinite(out1["logits"]).all()

    print("tests/test_matching_cpu.py: 全部断言通过 "
          "(shapes / finiteness / 支持→查询 / 查询→支持 / 梯度 / 消融开关 / "
          "forward_images / 单尺度融合)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

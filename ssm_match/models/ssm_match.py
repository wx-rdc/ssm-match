"""SSM-Match 整体模型：多尺度 SSM 匹配金字塔 + 学习凸组合融合 + 对比辅助头。

- 每个尺度一个独立的 CrossImageMatcher（架构同、参数独立，对应论文 φ_r）；
- 三尺度得分经 softmax(logits) 凸组合加权（论文"学习到的凸组合"）；
- 对比辅助头：以查询在真类上下文下的融合嵌入 cos(·, 类先验) 做 InfoNCE（论文 (d) 组件）。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .matching import CrossImageMatcher

# 尺度 → (特征通道, 匹配器隐维, 查询分块)
# 隐维按论文"三尺度匹配模块合计约 1.3M 可训练参数"预算选取
SCALE_SPECS = {
    "f1": dict(d_in=48, dim=160),
    "f2": dict(d_in=192, dim=256),
    "f3": dict(d_in=768, dim=320),
}
SCALES = ("f1", "f2", "f3")


def extract_features(backbone, support: torch.Tensor, query: torch.Tensor) -> tuple:
    """冻结骨干抽特征：support (B,n,K,3,H,W)、query (B,m,3,H,W) → 尺度名字典。
    骨干独立于 SSMMatch 模块（不进 state_dict，检查点不存 28MB 冗余权重）。"""
    fs = backbone(support)   # MultiScaleFeatures
    fq = backbone(query)
    sup = {"f1": fs.f1, "f2": fs.f2, "f3": fs.f3}
    qry = {"f1": fq.f1, "f2": fq.f2, "f3": fq.f3}
    return sup, qry


class SSMMatch(nn.Module):
    def __init__(self, n_way: int = 5, d_state: int = 16, query_chunk: int = 5,
                 aux_weight: float = 0.1, aux_temp: float = 0.2,
                 scales: tuple = SCALES, **matcher_kwargs):
        """scales：使用的尺度子集（消融单尺度时传 ("f2",) 等）；
        matcher_kwargs：透传 CrossImageMatcher 消融开关
        （use_prior/use_sep/scan_direction/use_dir_gate/use_evi_gate/score_sigmoid）。"""
        super().__init__()
        self.n_way = n_way
        self.aux_weight = aux_weight
        self.aux_temp = aux_temp
        self.scales = tuple(scales)
        self.matchers = nn.ModuleDict({
            k: CrossImageMatcher(SCALE_SPECS[k]["d_in"], SCALE_SPECS[k]["dim"],
                                 d_state, query_chunk, **matcher_kwargs)
            for k in self.scales
        })
        self.scale_logits = nn.Parameter(torch.zeros(len(self.scales)))

    def forward(self, support: dict, query: dict) -> dict:
        """support/query: 尺度名 → 张量。
        support[k]: (B, n_way, K, L_k, C_k)；query[k]: (B, m, L_k, C_k)，m = n_way*n_query。

        返回 dict：logits (B, m, n_way)、fused (B, n_way, m, dim_s) 按尺度堆叠、
        protos (B, n_way, dim_s) 类先验（各尺度各自一份），供对比辅助损失使用。
        """
        b = support[self.scales[0]].shape[0]   # 取实际使用的第一个尺度（单尺度消融兼容）
        out_scores, out_fused, out_protos = [], [], []
        for si, k in enumerate(self.scales):
            sup, qry = support[k], query[k]            # (B,n,K,L,C) / (B,m,L,C)
            n, K, L, _ = sup.shape[1:]
            m = qry.shape[1]
            sc = sup.new_empty(b, n, m)
            fu = sup.new_empty(b, n, m, self.matchers[k].dim)
            pr = sup.new_empty(b, n, self.matchers[k].dim)
            matcher = self.matchers[k]
            for bi in range(b):
                sup_b = sup[bi]
                qry_b = qry[bi]
                sc_b, fu_b = matcher(sup_b, qry_b)      # (n,m) (n,m,dim)
                sc[bi], fu[bi] = sc_b, fu_b
                pr[bi] = matcher.input_norm(matcher.input_proj(
                    sup_b.mean(dim=(1, 2))))            # (n,dim) 类先验
            out_scores.append(sc)
            out_fused.append(fu)
            out_protos.append(pr)
        w = F.softmax(self.scale_logits, dim=0)
        logits = sum(w[i] * out_scores[i] for i in range(len(self.scales)))  # (B,n,m)
        return {
            "logits": logits.permute(0, 2, 1).contiguous(),  # (B, m, n)：每查询对各类得分
            "scale_weights": w.detach(),
            "fused": out_fused,
            "protos": out_protos,
        }

    @staticmethod
    def contrastive_loss(fused: list, protos: list) -> torch.Tensor:
        """InfoNCE：查询在真类上下文的融合嵌入 ↔ 对应类先验（各尺度独立计算后平均）。"""
        losses = []
        for fu, pr in zip(fused, protos):
            # fu (B,n,m,dim)：查询 q 在类 c 上下文下的嵌入 → 取每类平均作为 q 的类上下文表示
            q_repr = F.normalize(fu.mean(dim=2), dim=-1)      # (B,n,dim)
            p_repr = F.normalize(pr, dim=-1)                  # (B,n,dim)
            sim = torch.einsum("bnd,bmd->bnm", q_repr, p_repr) / 0.2
            target = torch.arange(sim.shape[1], device=sim.device).expand_as(sim[:, :, 0])
            losses.append(F.cross_entropy(sim, target))
        return torch.stack(losses).mean()

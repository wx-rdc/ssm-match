"""跨图像选择性状态空间匹配模块（论文 2.2 核心机制）。

对一个类 c 的单尺度匹配流程：
  1. 序列构造（长度 T = 1 + K*(L+1) + L）：
         [P_c, S₁, SEP, S₂, SEP, …, S_K, SEP, Q]
     - P_c = Linear(支持集均值)：**类别条件状态初始化**——类先验置于序列头部，
       作为选择性扫描的初始状态先验；
     - SEP 为可学习分隔符，隔离各图像段；支持与查询在统一状态空间内交互。
  2. 双向扫描：同一 Mamba 权重分别处理正序（先验→支持→查询）与逆序（查询→支持→先验）
     序列，两路读出语义互补：
     - 前向取查询区输出：状态已累积全部支持证据 → **支持条件化的查询表示**；
     - 后向取支持区输出（逆序中查询先被扫描、支持后被扫描）：支持位置的状态
       已累积全部查询上下文 → **查询条件化的支持摘要**。
     注意后向分支不能取查询区输出——逆序扫描到查询位置时尚未经过支持 token，
     状态不含支持证据（此处曾为 bug，已修正）。
  3. **方向不对称融合**：两路查询输出经输入相关权重 softmax 加权融合（非对称平均）。
  4. **输入依赖跨图像门控**：g = σ(W[fused; 支持摘要])，门控后线性映射为该类匹配得分。

复杂度：单次扫描 O(T)=O(KL)，总复杂度线性于支持-查询联合长度，符合论文主张。
查询分块（query_chunk）控制反向传播激活显存。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm.modules.mamba_simple import Mamba


class CrossImageMatcher(nn.Module):
    """消融开关（对应论文 Table ablation_factor，默认全开=完整模型）：
    - use_prior：类别先验 token（False=零状态启动）；
    - use_sep：逐支持图 [SEP] 分隔符（False=完全去除）；
    - scan_direction："bi"（默认）/ "fwd" / "bwd"；
    - use_dir_gate：方向不对称融合（False=对称平均，仅 bi 有意义）；
    - use_evi_gate：输入依赖跨图像证据门控；
    - score_sigmoid：打分头 sigmoid（论文消融项，默认不用，见 R1-4）。
    """

    def __init__(self, d_in: int, dim: int, d_state: int = 16, query_chunk: int = 5,
                 use_prior: bool = True, use_sep: bool = True,
                 scan_direction: str = "bi", use_dir_gate: bool = True,
                 use_evi_gate: bool = True, score_sigmoid: bool = False):
        super().__init__()
        if scan_direction not in ("bi", "fwd", "bwd"):
            raise ValueError(f"scan_direction={scan_direction} 不合法")
        self.dim = dim
        self.query_chunk = query_chunk
        self.use_prior = use_prior
        self.use_sep = use_sep
        self.scan_direction = scan_direction
        self.use_dir_gate = use_dir_gate
        self.use_evi_gate = use_evi_gate
        self.score_sigmoid = score_sigmoid
        self.input_proj = nn.Linear(d_in, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.sep = nn.Parameter(torch.randn(1, dim) * 0.02)
        self.prior_proj = nn.Linear(dim, dim)
        self.scan = Mamba(d_model=dim, d_state=d_state, d_conv=4, expand=2)
        self.out_norm = nn.LayerNorm(dim)
        self.dir_gate = nn.Linear(2 * dim, 2)
        self.evi_gate = nn.Linear(2 * dim, dim)
        self.score_head = nn.Linear(dim, 1)
        nn.init.zeros_(self.score_head.bias)

    def _seq_forward(self, s: torch.Tensor, prior: torch.Tensor,
                     q: torch.Tensor, summary: torch.Tensor) -> tuple:
        """单类单块：s (K,L,dim)，prior (dim)，q (m,L,dim)，summary (dim) → (scores(m), fused(m,dim))"""
        K, L, dim = s.shape
        m = q.shape[0]
        sep = self.sep.to(s.dtype)
        # 支持前缀 [P_c?, S₁, SEP?, …, S_K, SEP?]：(Tp, dim)
        parts = [prior.unsqueeze(0)] if self.use_prior else []
        for j in range(K):
            if self.use_sep:
                parts += [s[j], sep.expand(L, -1)]
            else:
                parts.append(s[j])
        prefix = torch.cat(parts, dim=0) if parts else s.new_zeros(0, dim)
        Tp = prefix.shape[0]
        T = Tp + L

        scores = s.new_empty(m)
        fused = s.new_empty(m, dim)
        for lo in range(0, m, self.query_chunk):
            qc = q[lo:lo + self.query_chunk]                 # (c,L,dim)
            c = qc.shape[0]
            seq = torch.cat([prefix.unsqueeze(0).expand(c, -1, -1), qc], dim=1)  # (c,T,dim)
            h_fwd = h_bwd = None
            if self.scan_direction in ("bi", "fwd"):
                h_fwd = self.out_norm(self.scan(seq))[:, T - L:, :].mean(dim=1)  # 支持条件化查询
            if self.scan_direction in ("bi", "bwd"):
                # 逆序扫描：查询区在前(0..L-1)、支持区在后(L..T-1)；取支持区输出
                h_bwd = self.out_norm(self.scan(torch.flip(seq, dims=(1,))))[:, L:, :].mean(dim=1)
            if h_fwd is not None and h_bwd is not None:
                if self.use_dir_gate:
                    w = F.softmax(self.dir_gate(torch.cat([h_fwd, h_bwd], dim=-1)), dim=-1)
                    f = w[:, :1] * h_fwd + w[:, 1:] * h_bwd      # 方向不对称融合
                else:
                    f = 0.5 * h_fwd + 0.5 * h_bwd                # 对称平均（消融）
            else:
                f = h_fwd if h_fwd is not None else h_bwd
            if self.use_evi_gate:
                g = torch.sigmoid(self.evi_gate(torch.cat([f, summary.expand_as(f)], dim=-1)))
                f = g * f                                        # 跨图像门控
            fused[lo:lo + c] = f
            sc = self.score_head(f).squeeze(-1)
            if self.score_sigmoid:
                sc = torch.sigmoid(sc)                           # 打分头 sigmoid（消融）
            scores[lo:lo + c] = sc
        return scores, fused

    def forward(self, support: torch.Tensor, query: torch.Tensor) -> tuple:
        """support (n,K,L,d_in)、query (m,L,d_in) → scores (n,m)、fused (n,m,dim)"""
        n, K, L, _ = support.shape
        m = query.shape[0]
        s = self.input_norm(self.input_proj(support))
        q = self.input_norm(self.input_proj(query))
        pooled = s.mean(dim=(1, 2))                          # (n,dim)：K 张图 × L token 全池化
        prior = self.prior_proj(pooled)
        summary = pooled

        scores = s.new_empty(n, m)
        fused = s.new_empty(n, m, self.dim)
        for ci in range(n):
            sc, fu = self._seq_forward(s[ci], prior[ci], q, summary[ci])
            scores[ci] = sc
            fused[ci] = fu
        return scores, fused

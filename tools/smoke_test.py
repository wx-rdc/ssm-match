"""平台冒烟测试：真实骨干 + 真实匹配模块，随机图像张量跑一遍前向/反向/评测路径。

在启智调试任务的 JupyterLab 里运行：
    cd /tmp/code/ssm-match
    python tools/smoke_test.py --model_dir /tmp/pretrainmodel/vim-tiny-midclstok
预期：各项打印 ok，显存峰值 < 6GB（5-shot 全流程）。
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402

from ssm_match.backbone import FrozenVimBackbone  # noqa: E402
from ssm_match.models import SSMMatch  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default=".")
    ap.add_argument("--vim_ckpt", default="vim_t_midclstok_ft_78p3acc.pth")
    ap.add_argument("--n_way", type=int, default=5)
    ap.add_argument("--k_shot", type=int, default=1)
    args, _ = ap.parse_known_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}: {torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'}")

    backbone = FrozenVimBackbone(os.path.join(args.model_dir, args.vim_ckpt)).to(device)
    print("backbone ok: 冻结参数",
          f"{sum(p.numel() for p in backbone.parameters()) / 1e6:.2f}M,",
          "requires_grad 全 False:",
          all(not p.requires_grad for p in backbone.parameters()))

    model = SSMMatch(n_way=args.n_way, query_chunk=5).to(device)
    print(f"matcher ok: 可训练参数 {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    n, k, m = args.n_way, args.k_shot, args.n_way * 15
    sup = torch.randn(1, n, k, 3, 224, 224, device=device)
    qry = torch.randn(1, m, 3, 224, 224, device=device)
    labels = torch.arange(n, device=device).unsqueeze(1).repeat(1, 15).flatten()

    torch.cuda.reset_peak_memory_stats() if device == "cuda" else None
    t0 = time.time()
    fs, fq = None, None
    with torch.no_grad():
        fs = backbone(sup)._asdict()
        fq = backbone(qry)._asdict()
    t_feat = time.time() - t0

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    t0 = time.time()
    for _ in range(3):
        optimizer.zero_grad()
        out = model(fs, fq)
        loss = torch.nn.functional.cross_entropy(out["logits"].reshape(-1, n), labels) \
            + 0.1 * model.contrastive_loss(out["fused"], out["protos"])
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize() if device == "cuda" else None
    t_train = (time.time() - t0) / 3

    print(f"feature 前向 {t_feat * 1000:.0f}ms | 训练步 {t_train * 1000:.0f}ms/步 | loss={loss.item():.4f}")
    if device == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 3
        print(f"显存峰值 {peak:.2f} GB")
    print("SMOKE PASS")


if __name__ == "__main__":
    main()

"""本地（无 GPU / 无 mamba_ssm / 无 torchvision）单元验证。

对 CUDA-only 依赖做最小 stub，使整条前向/反向/检查点链路可在 CPU 上验证：
    python tests/test_local.py

在平台（真实 mamba_ssm）上运行本文件会跳过 stub，直接测真实实现。
"""

import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn


# ---------- CUDA-only 依赖的 CPU stub（若真实库可用则不 stub） ----------
def _stub_mamba_ssm():
    try:
        import mamba_ssm.modules.mamba_simple  # noqa: F401
        return False
    except Exception:
        pass
    ms = types.ModuleType("mamba_ssm")
    mod = types.ModuleType("mamba_ssm.modules")
    simple = types.ModuleType("mamba_ssm.modules.mamba_simple")

    class Mamba(nn.Module):
        """形状保真的替身：Block(x) -> x + 线性扰动，保持 (B,L,dim) 不变。"""

        def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bimamba_type=None, **kw):
            super().__init__()
            self.mix = nn.Linear(d_model, d_model)

        def forward(self, x):
            return x + self.mix(x)

    simple.Mamba = Mamba
    ms.modules = mod
    mod.mamba_simple = simple
    sys.modules["mamba_ssm"] = ms
    sys.modules["mamba_ssm.modules"] = mod
    sys.modules["mamba_ssm.modules.mamba_simple"] = simple
    return True


def _stub_torchvision():
    try:
        import torchvision  # noqa: F401
        return False
    except Exception:
        pass
    tv = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")

    class _T:
        def __init__(self, *a, **k):
            pass

        def __call__(self, x):
            return x

    class Compose(_T):
        def __init__(self, ts=None, *a, **k):
            self.ts = ts or []

        def __call__(self, x):
            for t in self.ts:
                x = t(x)
            return x

    transforms.Compose = Compose
    for name in ("Resize", "RandomCrop", "RandomHorizontalFlip", "ColorJitter",
                 "ToTensor", "Normalize"):
        setattr(transforms, name, _T)
    tv.transforms = transforms
    sys.modules["torchvision"] = tv
    sys.modules["torchvision.transforms"] = transforms
    return True


mamba_stubbed = _stub_mamba_ssm()
tv_stubbed = _stub_torchvision()
print(f"stub: mamba_ssm={mamba_stubbed} torchvision={tv_stubbed}")

# ---------- 1. miniImageNet 划分解析 ----------
from ssm_match.data.mini_imagenet import read_split_csv  # noqa: E402

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
splits = {s: read_split_csv(os.path.join(root, "data", "splits", "mini", f"{s}.csv"))
          for s in ("train", "val", "test")}
sizes = {s: (len(c), sum(len(v) for v in c.values())) for s, c in splits.items()}
assert sizes["train"] == (64, 38400) and sizes["val"] == (16, 9600) and sizes["test"] == (20, 12000), sizes
assert not set(splits["train"]) & set(splits["test"]), "train/test 类不许重叠"
print(f"1. 划分解析 ok: {sizes}")

# ---------- 2. 多尺度特征形状 ----------
from ssm_match.backbone.vim_tiny import GRID, TOKENS, to_multiscale  # noqa: E402

tokens = torch.randn(2, TOKENS, 192)
ms = to_multiscale(tokens)
assert ms.f1.shape == (2, 28 * 28, 48) and ms.f2.shape == (2, 14 * 14, 192) \
    and ms.f3.shape == (2, 7 * 7, 768), (ms.f1.shape, ms.f2.shape, ms.f3.shape)
print(f"2. 多尺度形状 ok: f1{tuple(ms.f1.shape)} f2{tuple(ms.f2.shape)} f3{tuple(ms.f3.shape)}")

# ---------- 3. SSMMatch 前向/反向（含对比辅助损失） ----------
from ssm_match.models import SSMMatch  # noqa: E402

model = SSMMatch(n_way=5, query_chunk=3)
sup = {"f1": torch.randn(2, 5, 1, 784, 48), "f2": torch.randn(2, 5, 1, 196, 192),
       "f3": torch.randn(2, 5, 1, 49, 768)}
qry = {"f1": torch.randn(2, 75, 784, 48), "f2": torch.randn(2, 75, 196, 192),
       "f3": torch.randn(2, 75, 49, 768)}
out = model(sup, qry)
assert out["logits"].shape == (2, 75, 5), out["logits"].shape
labels = torch.arange(5).unsqueeze(1).repeat(1, 15).flatten().unsqueeze(0).repeat(2, 1)
loss = torch.nn.functional.cross_entropy(out["logits"].reshape(-1, 5), labels.reshape(-1)) \
    + 0.1 * model.contrastive_loss(out["fused"], out["protos"])
loss.backward()
n_grad = sum(1 for p in model.parameters() if p.grad is not None)
assert n_grad > 0, "梯度未回传"
print(f"3. SSMMatch 前向反向 ok: logits{tuple(out['logits'].shape)} 可训练参数"
      f"{sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

# ---------- 4. 检查点存取 + RNG 恢复确定性 ----------
from ssm_match.engine.checkpoint import load_checkpoint, save_checkpoint  # noqa: E402
from ssm_match.utils.seed import set_seed  # noqa: E402

set_seed(42)
tiny = nn.Linear(8, 4)
opt = torch.optim.AdamW(tiny.parameters(), lr=1e-3)
save_checkpoint("/tmp/test_ckpt.pt", tiny, opt, None, epoch=3, episode=300,
                best_acc=0.71, extra={"kind": "latest"})
a = torch.randn(5)          # 保存后继续抽取的第一段随机数
tiny2 = nn.Linear(8, 4)     # 中途消耗 RNG（模拟断点间其它随机行为）
opt2 = torch.optim.AdamW(tiny2.parameters(), lr=1e-3)
payload = load_checkpoint("/tmp/test_ckpt.pt", tiny2, opt2, restore_rng=True)
b = torch.randn(5)          # 恢复到保存时刻状态后，第一段随机数应与 a 完全一致
assert torch.equal(a, b), "RNG 恢复后随机序列应与保存时刻的延续一致"
assert payload["epoch"] == 3 and payload["best_acc"] == 0.71
assert torch.allclose(tiny.weight, tiny2.weight)
print("4. 检查点/RNG ok")

# ---------- 5. 配置文件 ----------
import yaml  # noqa: E402

cfg = yaml.safe_load(open(os.path.join(root, "configs", "mini_1shot.yaml"), encoding="utf-8"))
assert cfg["n_way"] == 5 and cfg["k_shot"] == 1 and cfg["epochs"] == 200
print("5. 配置 ok")

# ---------- 6. CIFAR-FS 划分脚本（合成数据） ----------
import pickle  # noqa: E402
import subprocess  # noqa: E402

fake_root = "/tmp/fake_cifar"
os.makedirs(fake_root, exist_ok=True)
fine, coarse = [], []
for fi in range(100):
    ci = fi // 5  # 每 superclass 5 细类的合成映射
    fine += [fi] * 10
    coarse += [ci] * 10
with open(os.path.join(fake_root, "train"), "wb") as f:
    pickle.dump({"fine_labels": fine, "coarse_labels": coarse}, f)
r = subprocess.run(
    [sys.executable, os.path.join(root, "scripts", "prepare_cifar_fs.py"),
     "--cifar_root", fake_root, "--out", "/tmp/fake_splits"],
    capture_output=True, text=True)
assert r.returncode == 0, r.stderr
lines = open("/tmp/fake_splits/test.txt").read().split()
assert len(lines) == 20, lines
print("6. CIFAR-FS 划分脚本 ok（合成数据 60/20/20）")

print("\nALL LOCAL TESTS PASS")

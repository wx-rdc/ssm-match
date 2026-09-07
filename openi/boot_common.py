"""启智训练任务启动公共逻辑（被 openi/boots/ 下生成的启动文件调用）。

职责：
1. 探测云脑环境（c2net）并安装依赖（优先挂载模型里的预编译 wheel）；
2. 在挂载目录中定位各数据集的根路径（不同数据集仓的内部层级不一）；
3. 顺序执行 训练 → 终评（→ 跨域评测）并 fail-fast。
"""

import glob
import os
import subprocess
import sys


def cloud_context():
    """→ (ctx|None)。云脑返回 c2net 上下文，本地返回 None。"""
    try:
        from c2net.context import prepare
        return prepare()
    except Exception as exc:
        print(f"[boot] c2net unavailable ({exc.__class__.__name__}) -> local mode",
              flush=True)
        return None


def ensure_env(ctx):
    """装齐依赖：einops/timm 走镜像；mamba_ssm/causal_conv1d 优先离线 wheel。"""
    repo = _repo_root()
    sys.path.insert(0, repo)
    from openi.env_check import ensure_packages
    wheel_dir = None
    if ctx is not None:
        for sub in sorted(glob.glob(os.path.join(ctx.pretrain_model_path, "*"))):
            if glob.glob(os.path.join(sub, "*.whl")):
                wheel_dir = sub
                break
    ensure_packages(wheel_dir)


def find_class_root(base: str, n_classes: int, max_depth: int = 3) -> str:
    """在挂载目录里向上找含 >= n_classes 个子目录的层（class-folder 根）。"""
    dirs = [base]
    for _ in range(max_depth):
        nxt = []
        for d in dirs:
            subs = [p for p in glob.glob(os.path.join(d, "*")) if os.path.isdir(p)]
            if len(subs) >= n_classes:
                return d
            nxt.extend(subs)
        dirs = nxt
    raise RuntimeError(f"{base} 下 {max_depth} 层内找不到含 {n_classes} 个类子目录的层")


def data_root_for(dataset: str, ctx, n_classes: int) -> str:
    """各数据集在任务容器内的图片根目录。"""
    if ctx is None:  # 本地：仓库内约定路径
        local = {
            "mini": "data/mini-imagenet",
            "cub": "data/CUB_200_2011/images",
            "cars": "data/Stanford_Cars",
            "cifar_fs": "data/cifar-100-python",
        }
        return os.path.join(_repo_root(), local[dataset])
    base = ctx.dataset_path
    if dataset == "mini":
        for cand in (os.path.join(base, "mini-imagenet"), base):
            if os.path.isdir(cand):
                return cand
    if dataset == "cub":
        return find_class_root(base, n_classes)
    if dataset == "cars":
        return find_class_root(base, n_classes)
    if dataset == "cifar_fs":
        for cand in (os.path.join(base, "cifar-100-python"), base):
            if os.path.isfile(os.path.join(cand, "train")):
                return cand
    raise RuntimeError(f"无法定位 {dataset} 数据根目录（挂载于 {base}）")


def run(cmd: list) -> None:
    print("[boot] " + " ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc != 0:
        print(f"[boot] FAIL exit={rc}: {' '.join(cmd)}", flush=True)
        raise SystemExit(rc)


def suite_output_dir(ctx, name: str) -> str:
    out_root = os.path.join(ctx.output_path, name) if ctx is not None \
        else os.path.join(_repo_root(), "output", name)
    os.makedirs(out_root, exist_ok=True)
    return out_root


def _repo_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

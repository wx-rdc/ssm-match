"""启智云脑环境与骨干自检脚本（可作为调试/训练任务的启动文件）。

在任务容器内依次验证：
1. GPU / CUDA / PyTorch 版本与显存；
2. mamba-ssm 与 causal-conv1d 可用性（缺失则自动 pip 安装），并执行一次 GPU 选择性扫描前向；
3. 挂载的 Vim-Tiny 预训练权重可加载，统计参数量（骨干约 7.0M 全部冻结，匹配模块约 2.3M 可训练，合计约 9.4M）。

用法（本地调试）：
    python openi/env_check.py --model_dir /path/to/vim-tiny-midclstok

用法（云脑任务）：启动文件填 openi/env_check.py，预训练模型挂载
edwardzhu/vim-tiny-midclstok，脚本自动从 c2net_context 读取路径，
报告写入 output_path 并回传平台。
"""

import argparse
import glob
import json
import os
import subprocess
import sys

torch = None  # 延迟导入，便于把安装日志先打到 stdout


def _importable(mod: str) -> bool:
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _pip_install(args: list, mirror: str | None = None) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-build-isolation", *args]
    if mirror:
        cmd += ["-i", mirror]
    print(f"[env_check] pip install {' '.join(args)}", flush=True)
    subprocess.run(cmd, check=False)


def ensure_packages(wheel_dir: str | None) -> None:
    """云脑镜像只预装 torch；缺失的依赖在任务启动时装齐。

    - einops/timm 为纯 Python 包，走阿里云镜像（清华源限流严重）；
    - causal-conv1d/mamba-ssm 是 CUDA 扩展：优先用挂载模型目录里的
      预编译 wheel（与 torch2.1+cu118+py310 匹配），PyPI 上只有源码包，
      编译既慢又依赖镜像内 nvcc；无 GPU 环境直接跳过。
    """
    import torch
    mirror = "https://mirrors.aliyun.com/pypi/simple/"

    if not (_importable("einops") and _importable("timm")):
        _pip_install(["einops", "timm"], mirror)

    if not torch.cuda.is_available():
        return

    wheels = sorted(glob.glob(os.path.join(wheel_dir, "*.whl"))) if wheel_dir else []
    if wheels:
        # causal_conv1d 先装，mamba_ssm 运行时依赖它
        wheels.sort(key=lambda p: (0 if "causal_conv1d" in os.path.basename(p) else 1, p))
        _pip_install(wheels)
    elif not (_importable("causal_conv1d") and _importable("mamba_ssm")):
        _pip_install(["causal-conv1d>=1.4.0", "mamba-ssm==2.2.2"], mirror)


def resolve_context():
    """优先使用 c2net 上下文（云脑），否则退回命令行参数（本地）。"""
    try:
        from c2net.context import prepare
        ctx = prepare()
        return ctx, True
    except Exception as exc:  # 本地环境无 c2net
        print(f"[env_check] c2net unavailable ({exc.__class__.__name__}), fallback to args")
        return None, False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default=".", help="本地调试时 Vim-Tiny 权重目录")
    parser.add_argument("--wheel_dir", default="", help="预编译 wheel 目录（causal_conv1d/mamba_ssm），云脑下默认自动从挂载模型中探测")
    # 平台会注入额外参数，必须用 parse_known_args
    args, _ = parser.parse_known_args()

    report = {"stage": "env_check"}

    import torch
    report["python"] = sys.version.split()[0]
    report["torch"] = torch.__version__
    report["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        report["cuda"] = torch.version.cuda
        report["gpu"] = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory
        report["gpu_mem_gb"] = round(total / 1024 ** 3, 1)
    print("[env_check] " + json.dumps(report, ensure_ascii=False), flush=True)

    ctx, in_cloudbrain = resolve_context()
    wheel_dir = args.wheel_dir or None
    if in_cloudbrain and not wheel_dir:
        # 从挂载的预训练模型里找含 *.whl 的目录（如 cu118-torch210-wheels）
        for sub in sorted(glob.glob(os.path.join(ctx.pretrain_model_path, "*"))):
            if glob.glob(os.path.join(sub, "*.whl")):
                wheel_dir = sub
                break

    ensure_packages(wheel_dir)

    # ---- mamba_ssm GPU 前向冒烟测试 ----
    try:
        from mamba_ssm import Mamba
        layer = Mamba(d_model=64, d_state=16, d_conv=4, expand=2).cuda()
        out = layer(torch.randn(2, 128, 64, device="cuda"))
        report["mamba_forward_ok"] = bool(torch.isfinite(out).all())
    except Exception as exc:
        report["mamba_forward_ok"] = False
        report["mamba_error"] = f"{exc.__class__.__name__}: {exc}"

    # ---- Vim-Tiny 权重加载 ----
    model_dir = args.model_dir
    if in_cloudbrain:
        model_dir = os.path.join(ctx.pretrain_model_path, "vim-tiny-midclstok")
    ckpts = sorted(glob.glob(os.path.join(model_dir, "*.pth")))
    report["model_dir"] = model_dir
    report["checkpoints"] = [os.path.basename(p) for p in ckpts]
    for ckpt in ckpts:
        try:
            try:
                state = torch.load(ckpt, map_location="cpu")
            except Exception:
                # torch>=2.6 默认 weights_only=True，老式 checkpoint 需显式关闭
                state = torch.load(ckpt, map_location="cpu", weights_only=False)
            state_dict = state.get("model", state) if isinstance(state, dict) else state
            n_params = sum(v.numel() for v in state_dict.values() if hasattr(v, "numel"))
            report[f"params:{os.path.basename(ckpt)}"] = f"{n_params / 1e6:.2f}M"
        except Exception as exc:
            report[f"load_error:{os.path.basename(ckpt)}"] = str(exc)[:200]

    ok = report.get("cuda_available") and report.get("mamba_forward_ok") and bool(ckpts)
    report["verdict"] = "PASS" if ok else "FAIL"
    print("[env_check] final: " + json.dumps(report, ensure_ascii=False), flush=True)

    # ---- 报告写入 output_path 并回传 ----
    out_dir = "."
    if in_cloudbrain:
        out_dir = ctx.output_path
    with open(os.path.join(out_dir, "openi_env_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if in_cloudbrain:
        from c2net.context import upload_output
        upload_output()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

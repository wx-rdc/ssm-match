"""检查点：原子保存 / 加载 / 自动发现，含三方 RNG 状态（断点续训的关键）。"""

import glob
import os

import torch

from ..utils.seed import get_rng_state, set_rng_state


def save_checkpoint(path: str, model: torch.nn.Module, optimizer, scheduler,
                    epoch: int, episode: int, best_acc: float, extra: dict | None = None):
    tmp = path + ".tmp"
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer else None,
        "scheduler": scheduler.state_dict() if scheduler else None,
        "epoch": epoch,
        "episode": episode,
        "best_acc": best_acc,
        "rng": get_rng_state(),
        "extra": extra or {},
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)  # 原子替换，断电不留半个文件


def load_checkpoint(path: str, model: torch.nn.Module, optimizer=None, scheduler=None,
                    restore_rng: bool = True) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng and payload.get("rng"):
        set_rng_state(payload["rng"])
    return payload


def find_latest(search_dirs: list) -> str | None:
    """在多个目录中找最新的 latest.pt（平台 output_path 与本地目录都搜）。"""
    candidates = []
    for d in search_dirs:
        if not d:
            continue
        candidates += glob.glob(os.path.join(d, "**", "latest.pt"), recursive=True)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)

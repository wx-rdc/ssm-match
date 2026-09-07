"""episode 评测：n 个随机 episode 的 5-way 准确率，均值 ± 95% 置信区间。

评测使用独立种子（默认 12345），与训练 RNG 分离；支持任意 class_images
字典（mini / CIFAR-FS / class-folder）。

dump_path（回应审稿意见 #R3-5，可选）：提供时把逐 episode accs 以约定 JSON 格式
落盘（{"method","n_way","k_shot","n_episodes","accs"}），供 tools/stats_tests.py
对同一批测试 episodes 做配对统计检验；默认 None，不影响既有调用。
"""

import json
import math
import os

import torch

from ..backbone import forward_images
from ..data.episode import EpisodeBatcher, EpisodeDataset, EpisodeSampler
from ..data.transforms import build_transform


@torch.no_grad()
def evaluate(model: torch.nn.Module, backbone, class_images: dict,
             n_way: int, k_shot: int, n_query: int, n_episodes: int,
             device: str, batch_episodes: int = 4, seed: int = 12345,
             progress_every: int = 200, dump_path: str | None = None,
             method: str = "ssm-match") -> dict:
    model.eval()
    sampler = EpisodeSampler(class_images, n_way, k_shot, n_query, seed=seed)
    images_per_episode = n_way * (k_shot + n_query)
    dataset = EpisodeDataset(sampler, n_episodes, build_transform(train=False))
    batcher = EpisodeBatcher(n_way)

    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_episodes * images_per_episode,
        shuffle=False, num_workers=4, collate_fn=batcher.collate)

    accs = []
    pending = 0
    for sup, queries in loader:
        sup = sup.to(device, non_blocking=True)               # (b,n,K,3,H,W)
        feats_s = forward_images(backbone, sup)               # 尺度名→(b,n,K,L,C)
        for qi, q in enumerate(queries):                      # 逐 episode
            q = q.to(device, non_blocking=True)               # (n*nq,3,H,W)
            feat_q = forward_images(backbone, q)              # 尺度名→(n*nq,L,C)
            feats_q = {k: v.unsqueeze(0) for k, v in feat_q.items()}
            sup_q = {k: v[qi:qi + 1] for k, v in feats_s.items()}
            logits = model(sup_q, feats_q)["logits"]          # (1, m, n)
            pred = logits.argmax(dim=-1).squeeze(0).view(n_way, n_query)
            labels = torch.arange(n_way, device=device).unsqueeze(1)
            accs.append((pred == labels).float().mean().item())
            pending += 1
            if progress_every and pending % progress_every == 0:
                print(f"  eval {pending}/{n_episodes}: "
                      f"running acc={sum(accs) / len(accs):.4f}", flush=True)
    mean = sum(accs) / len(accs)
    ci95 = 1.96 * math.sqrt(mean * (1 - mean) / len(accs))
    if dump_path:
        parent = os.path.dirname(dump_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(dump_path, "w", encoding="utf-8") as f:
            json.dump({"method": method, "n_way": n_way, "k_shot": k_shot,
                       "n_episodes": len(accs), "accs": accs},
                      f, ensure_ascii=False, indent=2)
    return {"acc": round(mean, 4), "ci95": round(ci95, 4), "n_episodes": n_episodes}

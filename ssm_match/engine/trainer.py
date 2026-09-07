"""训练循环：episode 交叉熵 + InfoNCE 辅助损失。

断点续训策略：
  - 每 epoch 末原子保存 latest.pt（模型/优化器/调度器/进度/三方 RNG）；
  - 验证刷新最佳时另存 best.pt；
  - SIGTERM（平台任务到期/手动停止）触发：先在当前 epoch 内尽快收尾，落盘
    latest-onexit 检查点后退出；--resume auto 可无缝续训。
平台计费按 6 分钟粒度、结果文件保留 30 天，故不采用"攒很久才存"的策略。
"""

import os
import signal

import torch
import torch.nn.functional as F

from ..backbone import forward_images
from ..utils.logging import Logger
from .checkpoint import find_latest, load_checkpoint, save_checkpoint


class Trainer:
    def __init__(self, model, backbone, optimizer, scheduler, train_loader,
                 n_way: int, n_query: int, output_dir: str, device: str,
                 epochs: int, log: Logger, aux_weight: float = 0.1,
                 eval_fn=None, eval_every_epochs: int = 10,
                 start_epoch: int = 0, start_episode: int = 0, best_acc: float = 0.0):
        self.model, self.backbone = model, backbone
        self.optimizer, self.scheduler = optimizer, scheduler
        self.train_loader = train_loader
        self.n_way, self.n_query = n_way, n_query
        self.output_dir, self.device, self.epochs = output_dir, device, epochs
        self.log = log
        self.aux_weight = aux_weight
        self.eval_fn, self.eval_every = eval_fn, eval_every_epochs
        self.epoch, self.episode = start_epoch, start_episode
        self.best_acc = best_acc
        self._stop = False
        signal.signal(signal.SIGTERM, self._on_sigterm)

    def _on_sigterm(self, signum, frame):
        print("[trainer] 收到 SIGTERM，完成当前 batch 后保存并退出", flush=True)
        self._stop = True

    def _run_epoch(self) -> dict:
        self.model.train()
        ce_sum, n_steps, correct, total = 0.0, 0, 0, 0
        for sup, queries in self.train_loader:
            sup = sup.to(self.device, non_blocking=True)
            query_batches = [q.to(self.device, non_blocking=True) for q in queries]
            labels = torch.cat([
                torch.arange(self.n_way, device=self.device)
                .unsqueeze(1).repeat(1, self.n_query).flatten()
                for _ in query_batches])                      # (b*n*nq,)
            with torch.no_grad():                             # 冻结骨干，无梯度
                feats_s = forward_images(self.backbone, sup)  # (b,n,K,3,H,W)→(b,n,K,L,C)
                feats_q = forward_images(
                    self.backbone, torch.cat(query_batches))  # (b*nq,3,H,W)→(b*nq,L,C)

            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(feats_s, feats_q)
            logits = out["logits"]                            # (b, m, n)
            loss = F.cross_entropy(logits.reshape(-1, self.n_way), labels)
            if self.aux_weight > 0:
                loss = loss + self.aux_weight * self.model.contrastive_loss(
                    out["fused"], out["protos"])
            loss.backward()
            self.optimizer.step()

            pred = logits.detach().argmax(-1)                 # (b, m)
            correct += (pred == labels.view_as(pred)).sum().item()
            total += pred.numel()
            ce_sum += loss.item()
            n_steps += 1
            self.episode += 1
            if self._stop:
                break
        return {"loss": round(ce_sum / max(n_steps, 1), 4),
                "train_acc": round(correct / max(total, 1), 4)}

    def fit(self):
        while self.epoch < self.epochs and not self._stop:
            stats = self._run_epoch()
            rec = {"epoch": self.epoch, "episode": self.episode, **stats}
            if self.eval_fn and (self.epoch + 1) % self.eval_every == 0 and not self._stop:
                ev = self.eval_fn()
                rec["val_acc"] = ev["acc"]
                if ev["acc"] > self.best_acc:
                    self.best_acc = ev["acc"]
                    save_checkpoint(os.path.join(self.output_dir, "best.pt"),
                                    self.model, self.optimizer, self.scheduler,
                                    self.epoch + 1, self.episode, self.best_acc,
                                    {"kind": "best"})
            self.log.log(rec)
            if self._stop:
                break
            save_checkpoint(os.path.join(self.output_dir, "latest.pt"),
                            self.model, self.optimizer, self.scheduler,
                            self.epoch + 1, self.episode, self.best_acc,
                            {"kind": "latest"})
            self.epoch += 1
        # 退出前兜底保存（无论正常结束还是被停）
        save_checkpoint(os.path.join(self.output_dir, "latest.pt"),
                        self.model, self.optimizer, self.scheduler,
                        self.epoch, self.episode, self.best_acc,
                        {"kind": "latest-onexit"})
        self.log.log({"msg": "训练结束", "epoch": self.epoch,
                      "best_acc": round(self.best_acc, 4)})


def resume_if_needed(model, optimizer, scheduler, search_dirs: list,
                     resume: str, log: Logger) -> dict:
    """resume: 'auto' | 检查点路径 | 'none'。返回恢复进度 dict。"""
    path = resume
    if resume == "auto":
        path = find_latest(search_dirs)
        if path:
            log.log({"msg": f"发现检查点，自动续训: {path}"})
    if not path or path == "none":
        return {"epoch": 0, "episode": 0, "best_acc": 0.0}
    payload = load_checkpoint(path, model, optimizer, scheduler, restore_rng=True)
    log.log({"msg": f"已恢复检查点 {path}", "epoch": payload["epoch"],
             "episode": payload["episode"], "best_acc": payload.get("best_acc")})
    return {"epoch": payload["epoch"], "episode": payload["episode"],
            "best_acc": payload.get("best_acc", 0.0)}

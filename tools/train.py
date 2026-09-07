"""SSM-Match 训练入口（本地 / 启智云脑通用）。

平台用法（训练任务启动文件 tools/train.py）：
    自动读取 c2net 路径，挂载 mini-imagenet 数据集与 vim-tiny-midclstok 模型即可。

本地用法：
    python tools/train.py --config configs/mini_1shot.yaml --dataset mini \
        --data_root data/mini-imagenet --model_dir /path/to/vim-tiny-midclstok

CUB / Stanford Cars（class-folder + data/splits/<name>/{train,val,test}.txt）：
    python tools/train.py --config configs/cub_1shot.yaml --dataset cub \
        --data_root data/CUB_200_2011/images

断点续训：
    --resume auto    自动在输出目录与平台 output_path 中找 latest.pt
    --resume <path>  指定检查点
    --resume none    全新训练
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import yaml  # noqa: E402

from ssm_match.backbone import FrozenVimBackbone  # noqa: E402
from ssm_match.data import (DATASETS, EpisodeBatcher, EpisodeDataset,  # noqa: E402
                            EpisodeSampler, build_transform, load_split)
from ssm_match.engine import Trainer, evaluate, resume_if_needed  # noqa: E402
from ssm_match.models import build_model_from_config  # noqa: E402
from ssm_match.platform import resolve_context  # noqa: E402
from ssm_match.utils import Logger, set_seed  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "mini_1shot.yaml"))
    p.add_argument("--dataset", default="mini", choices=DATASETS,
                   help="mini / cub / cars 训练；cifar_fs 仅用于跨域评测（tools/eval.py）")
    p.add_argument("--data_root", default=None, help="图片根目录（默认走 c2net）")
    p.add_argument("--model_dir", default=None, help="Vim 权重目录（默认走 c2net 挂载）")
    p.add_argument("--output_dir", default=None)
    p.add_argument("--splits_dir", default=None,
                   help="划分目录（默认 data/splits/<dataset>，需已提交到代码仓）")
    p.add_argument("--resume", default="auto", help="auto|none|checkpoint 路径")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None, help="覆盖配置里的 epochs")
    # 平台会注入额外参数，必须 parse_known_args
    args, _ = p.parse_known_args()
    if args.splits_dir is None:
        args.splits_dir = os.path.join(REPO_ROOT, "data", "splits", args.dataset)
    return args


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    seed = args.seed if args.seed is not None else cfg["seed"]
    epochs = args.epochs if args.epochs is not None else cfg["epochs"]
    ctx = resolve_context(args.data_root, args.model_dir, args.output_dir)
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[train] device={device} data_root={ctx.data_root} output={ctx.output_dir}",
          flush=True)
    backbone = FrozenVimBackbone(
        os.path.join(ctx.model_dir, cfg.get("vim_ckpt", "vim_t_midclstok_ft_78p3acc.pth"))
    ).to(device)

    n_way, k_shot = cfg["n_way"], cfg["k_shot"]
    n_query = cfg.get("n_query", 15)
    train_split = load_split(args.dataset, ctx.data_root, args.splits_dir, "train")
    sampler = EpisodeSampler(train_split, n_way, k_shot, n_query, seed=seed)
    per_episode_images = n_way * (k_shot + n_query)
    dataset = EpisodeDataset(sampler, cfg["episodes_per_epoch"], build_transform(train=True))
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg["batch_episodes"] * per_episode_images,
        shuffle=False, num_workers=cfg.get("num_workers", 8),
        collate_fn=EpisodeBatcher(n_way).collate, pin_memory=True, drop_last=True)

    # 因子化消融开关（缺省=完整模型）
    model = build_model_from_config(cfg).to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - 1)

    log = Logger(ctx.output_dir)
    log.log({"msg": "开始", "config": args.config, "seed": seed,
             "trainable_params": sum(p.numel() for p in model.parameters()
                                     if p.requires_grad)})

    val_classes = load_split(args.dataset, ctx.data_root, args.splits_dir, "val")

    progress = resume_if_needed(model, optimizer, scheduler,
                                [ctx.output_dir, os.path.join(REPO_ROOT, "output")],
                                args.resume, log)

    def eval_fn():
        return evaluate(model, backbone, val_classes, n_way, k_shot, n_query,
                        cfg.get("val_episodes", 200), device,
                        batch_episodes=cfg.get("batch_episodes", 4), seed=777,
                        progress_every=0)

    trainer = Trainer(model, backbone, optimizer, scheduler, loader,
                      n_way=n_way, n_query=n_query, output_dir=ctx.output_dir,
                      device=device, epochs=epochs, log=log,
                      aux_weight=cfg.get("aux_weight", 0.1), eval_fn=eval_fn,
                      eval_every_epochs=cfg.get("eval_every", 10),
                      start_epoch=progress["epoch"], start_episode=progress["episode"],
                      best_acc=progress["best_acc"])
    trainer.fit()
    ctx.finish()  # 云脑：回传 output_path（结果保留 30 天）


if __name__ == "__main__":
    main()

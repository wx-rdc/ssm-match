"""终评 / 跨域评测入口（本地 / 启智云脑通用）。

加载训练产出的 best.pt（或指定 ckpt），在指定数据集的 test 划分上跑
N 个 episode（论文协议 10000），结果 JSON 落盘（逐 episode accs，
供 tools/stats_tests.py 做配对检验）。

用法：
    # 主结果终评（mini 1-shot，5 个种子各跑一次）
    python tools/eval.py --dataset mini --k_shot 1 \
        --run_dir output/main_mini_1shot/seed_42 --data_root data/mini-imagenet

    # 跨域：mini 训练的 ckpt 直接评 CIFAR-FS test（32×32 上采样协议）
    python tools/eval.py --dataset cifar_fs --k_shot 1 \
        --run_dir output/main_mini_1shot/seed_42 \
        --data_root data/cifar-100-python --splits_dir data/splits/cifar_fs

    # CUB/Cars（class-folder）
    python tools/eval.py --dataset cub --k_shot 5 \
        --run_dir output/main_cub_5shot/seed_42 --data_root data/CUB_200_2011/images
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch  # noqa: E402
import yaml  # noqa: E402

from ssm_match.backbone import FrozenVimBackbone  # noqa: E402
from ssm_match.data import DATASETS, load_split  # noqa: E402
from ssm_match.engine import evaluate  # noqa: E402
from ssm_match.models import build_model_from_config  # noqa: E402
from ssm_match.platform import resolve_context  # noqa: E402
from ssm_match.utils import set_seed  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True,
                   help="训练输出目录（自动找 best.pt，缺省 latest.pt）或 --ckpt 优先")
    p.add_argument("--ckpt", default=None, help="显式检查点路径（覆盖 run_dir 探测）")
    p.add_argument("--config", default=None,
                   help="模型开关来源配置（默认按 dataset/k_shot 选 configs/<ds>_<shot>shot.yaml）")
    p.add_argument("--dataset", default="mini", choices=DATASETS)
    p.add_argument("--data_root", default=None)
    p.add_argument("--model_dir", default=None)
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--split", default="test", choices=("test", "val", "train"))
    p.add_argument("--k_shot", type=int, default=None, help="默认取配置文件里的 k_shot")
    p.add_argument("--n_way", type=int, default=5)
    p.add_argument("--n_query", type=int, default=15)
    p.add_argument("--n_episodes", type=int, default=10000)
    p.add_argument("--seed", type=int, default=12345, help="评测采样种子（与训练分离）")
    p.add_argument("--batch_episodes", type=int, default=4)
    p.add_argument("--method", default="ssm-match", help="写入 dump JSON 的方法名")
    p.add_argument("--out", default=None,
                   help="结果 JSON 路径（默认 <run_dir>/eval_<dataset>_<k>shot.json）")
    # 平台会注入额外参数，必须 parse_known_args
    args, _ = p.parse_known_args()
    if args.config is None:
        shot = args.k_shot if args.k_shot is not None else 1
        args.config = os.path.join(REPO_ROOT, "configs", f"{args.dataset}_{shot}shot.yaml")
    if args.splits_dir is None:
        args.splits_dir = os.path.join(REPO_ROOT, "data", "splits", args.dataset)
    return args


def find_checkpoint(run_dir: str, explicit: str | None) -> str:
    if explicit:
        return explicit
    for name in ("best.pt", "latest.pt"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"{run_dir} 下没有 best.pt / latest.pt")


def main():
    args = parse_args()
    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    k_shot = args.k_shot if args.k_shot is not None else cfg["k_shot"]
    ctx = resolve_context(args.data_root, args.model_dir,
                          None if args.out is None else os.path.dirname(args.out))
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[eval] device={device} dataset={args.dataset} split={args.split} "
          f"data_root={ctx.data_root}", flush=True)

    backbone = FrozenVimBackbone(
        os.path.join(ctx.model_dir, cfg.get("vim_ckpt", "vim_t_midclstok_ft_78p3acc.pth"))
    ).to(device)

    model = build_model_from_config(cfg, n_way=args.n_way).to(device)
    ckpt_path = find_checkpoint(args.run_dir, args.ckpt)
    try:
        payload = torch.load(ckpt_path, map_location="cpu")
    except Exception:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(payload["model"])
    print(f"[eval] loaded {ckpt_path} (epoch {payload.get('epoch', '?')})", flush=True)

    classes = load_split(args.dataset, ctx.data_root, args.splits_dir, args.split)
    print(f"[eval] {len(classes)} 类", flush=True)

    out_path = args.out or os.path.join(
        args.run_dir, f"eval_{args.dataset}_{k_shot}shot_{args.split}.json")
    result = evaluate(model, backbone, classes, args.n_way, k_shot, args.n_query,
                      args.n_episodes, device, batch_episodes=args.batch_episodes,
                      seed=args.seed, progress_every=1000, dump_path=out_path,
                      method=args.method)
    result.update({"dataset": args.dataset, "split": args.split,
                   "ckpt": ckpt_path, "dump": out_path})
    print("[eval] " + json.dumps(result, ensure_ascii=False), flush=True)
    ctx.finish()  # 云脑：回传 output_path


if __name__ == "__main__":
    main()

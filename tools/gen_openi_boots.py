"""生成 P2 实验矩阵的配置与启智启动文件（openi/boots/），幂等可重复执行。

产物：
1. configs/{cub,cars}_{1,5}shot.yaml            —— CUB / Stanford Cars 主配置
2. configs/ablations/{mini1,cub5}_<factor>.yaml —— 因子化消融配置（论文 tab:ablation_factor 的 12 个变体）
3. openi/boots/main_*.py (6) / abl_*.py (24)    —— 训练任务启动文件（训练→终评→跨域 一条龙）
4. openi/boots/MANIFEST.md                      —— 任务清单（平台建任务时的对照表）

用法：python tools/gen_openi_boots.py
"""

import os
import sys

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAIN_SEEDS = [42, 123, 456, 2026, 777]
ABL_SEEDS = [42, 123, 456]

# (id, 论文 tab:ablation_factor 行, 覆盖键)
FACTORS = [
    ("no_prior",   "w/o class prior token",    {"use_prior": False}),
    ("no_sep",     "w/o [SEP] separators",     {"use_sep": False}),
    ("fwd",        "Forward scan only",        {"scan_direction": "fwd"}),
    ("bwd",        "Backward scan only",       {"scan_direction": "bwd"}),
    ("no_dirgate", "Symmetric fusion",         {"use_dir_gate": False}),
    ("no_evigate", "w/o evidence gate",        {"use_evi_gate": False}),
    ("sigmoid",    "Sigmoid score head",       {"score_sigmoid": True}),
    ("scale_f1",   "Single scale 28x28",       {"scales": ["f1"]}),
    ("scale_f2",   "Single scale 14x14",       {"scales": ["f2"]}),
    ("scale_f3",   "Single scale 7x7",         {"scales": ["f3"]}),
    ("ns8",        "State dim Ns=8",           {"d_state": 8}),
    ("ns32",       "State dim Ns=32",          {"d_state": 32}),
]

IMAGE = "ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0"
MODELS = "vim-tiny-midclstok + cu118-torch210-wheels"
N_CLASSES = {"mini": 100, "cub": 200, "cars": 196}
MOUNTS = {"mini": "mini-imagenet", "cub": "CUB_200_2011", "cars": "StanfordCars"}


def base_cfg(k_shot: int) -> dict:
    with open(os.path.join(REPO, "configs", f"mini_{k_shot}shot.yaml"),
              encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_yaml(path: str, cfg: dict, header: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# {header}\n# 由 tools/gen_openi_boots.py 生成，勿手改。\n\n")
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


BOOT_TEMPLATE = '''# -*- coding: utf-8 -*-
# 由 tools/gen_openi_boots.py 自动生成，勿手改。
# 任务：@DESC@
# 平台配置：镜像 @IMAGE@；预训练模型挂 @MODELS@；启动文件选本文件。
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from openi import boot_common  # noqa: E402

SPEC = @SPEC@


def main():
    ctx = boot_common.cloud_context()
    boot_common.ensure_env(ctx)
    data_root = boot_common.data_root_for(SPEC["dataset"], ctx, SPEC["n_classes"])
    out_root = boot_common.suite_output_dir(ctx, SPEC["name"])
    print(f"[boot] {SPEC['name']} data={data_root} out={out_root}", flush=True)

    for seed in SPEC["seeds"]:
        run_dir = os.path.join(out_root, f"seed_{seed}")
        boot_common.run([sys.executable, "tools/train.py",
                         "--config", SPEC["config"], "--dataset", SPEC["dataset"],
                         "--data_root", data_root, "--seed", str(seed),
                         "--output_dir", run_dir])
        boot_common.run([sys.executable, "tools/eval.py",
                         "--dataset", SPEC["dataset"], "--k_shot", str(SPEC["k_shot"]),
                         "--config", SPEC["config"],
                         "--data_root", data_root, "--run_dir", run_dir,
                         "--method", SPEC["name"],
                         "--out", os.path.join(run_dir, "eval_test.json")])
        for ds, cfg in SPEC["cross_eval"].items():
            root2 = boot_common.data_root_for(ds, ctx, 100)
            boot_common.run([sys.executable, "tools/eval.py",
                             "--dataset", ds, "--k_shot", str(SPEC["k_shot"]),
                             "--config", cfg, "--data_root", root2,
                             "--splits_dir", os.path.join(ROOT, "data", "splits", ds),
                             "--run_dir", run_dir, "--method", SPEC["name"],
                             "--out", os.path.join(run_dir, f"eval_{ds}.json")])


if __name__ == "__main__":
    main()
'''


def make_spec(name, dataset, k, config, seeds, desc, cross_eval=None):
    return dict(name=name, dataset=dataset, k_shot=k, config=config, seeds=seeds,
                n_classes=N_CLASSES[dataset], method=name,
                cross_eval=cross_eval or {}, desc=desc)


def write_boot(path: str, spec: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    body = (BOOT_TEMPLATE
            .replace("@SPEC@", repr(spec))
            .replace("@DESC@", spec["desc"])
            .replace("@IMAGE@", IMAGE)
            .replace("@MODELS@", MODELS))
    with open(path, "w", encoding="utf-8") as f:
        f.write(body)


def main():
    generated = []

    # 1) CUB / Cars 主配置（沿用 mini 协议，仅数据集不同）
    for dataset in ("cub", "cars"):
        for k in (1, 5):
            path = os.path.join(REPO, "configs", f"{dataset}_{k}shot.yaml")
            write_yaml(path, base_cfg(k),
                       f"{dataset.upper()} 5-way {k}-shot（协议同 mini_{k}shot.yaml）")
            generated.append(os.path.relpath(path, REPO))

    # 2) 因子化消融配置：mini 1-shot 与 CUB 5-shot 两列
    abl_cfgs = {}
    for tag, dataset, k in (("mini1", "mini", 1), ("cub5", "cub", 5)):
        base = base_cfg(k)
        base["seed"] = ABL_SEEDS[0]
        for fid, desc, overrides in FACTORS:
            cfg = dict(base)
            cfg.update(overrides)
            path = os.path.join(REPO, "configs", "ablations", f"{tag}_{fid}.yaml")
            write_yaml(path, cfg,
                       f"消融 {desc}（{dataset} {k}-shot；种子在 boot 里覆盖为 {ABL_SEEDS}）")
            abl_cfgs[(tag, fid)] = os.path.relpath(path, REPO)
            generated.append(os.path.relpath(path, REPO))

    # 3) 启动文件：主结果（5 种子；mini 附带 CIFAR-FS 跨域终评）
    boots = []
    for dataset in ("mini", "cub", "cars"):
        for k in (1, 5):
            name = f"main_{dataset}_{k}shot"
            cross = {"cifar_fs": f"configs/mini_{k}shot.yaml"} if dataset == "mini" else {}
            desc = f"主结果 {dataset} {k}-shot，5 种子，训练+终评"
            if cross:
                desc += "+CIFAR-FS 跨域评测"
            boots.append((name, make_spec(name, dataset, k,
                                          f"configs/{dataset}_{k}shot.yaml",
                                          MAIN_SEEDS, desc, cross)))

    # 4) 启动文件：消融（3 种子）
    for tag, dataset, k in (("mini1", "mini", 1), ("cub5", "cub", 5)):
        for fid, desc, _ in FACTORS:
            name = f"abl_{tag}_{fid}"
            boots.append((name, make_spec(name, dataset, k, abl_cfgs[(tag, fid)],
                                          ABL_SEEDS, f"消融[{tag}] {desc}，3 种子")))

    for name, spec in boots:
        path = os.path.join(REPO, "openi", "boots", f"{name}.py")
        write_boot(path, spec)
        generated.append(os.path.relpath(path, REPO))

    write_manifest(boots)
    generated.append("openi/boots/MANIFEST.md")

    print(f"生成 {len(generated)} 个文件：")
    for g in generated:
        print("  " + g)


def write_manifest(boots) -> None:
    lines = [
        "# 启智训练任务清单（由 tools/gen_openi_boots.py 生成）",
        "",
        f"镜像统一选 `{IMAGE}`；预训练模型统一挂 {MODELS}。",
        "「数据集挂载」含 cifar-100-python 的任务仅为 mini 主结果（跨域评测用）。",
        "",
        "| 任务名（建议） | 启动文件 | 数据集挂载 | 种子数 | 内容 |",
        "|---|---|---|---|---|",
    ]
    for name, spec in boots:
        ds = MOUNTS[spec["dataset"]]
        extra = " + cifar-100-python" if spec["cross_eval"] else ""
        kind = "主结果" if name.startswith("main") else "消融"
        lines.append(f"| ssm-{name.replace('_', '-')} | openi/boots/{name}.py "
                     f"| {ds}{extra} | {len(spec['seeds'])} | {kind}：{spec['desc']} |")
    path = os.path.join(REPO, "openi", "boots", "MANIFEST.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())

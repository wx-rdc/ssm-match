# -*- coding: utf-8 -*-
# 由 tools/gen_openi_boots.py 自动生成，勿手改。
# 任务：消融[cub5] State dim Ns=8，3 种子
# 平台配置：镜像 ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0；预训练模型挂 vim-tiny-midclstok + cu118-torch210-wheels；启动文件选本文件。
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from openi import boot_common  # noqa: E402

SPEC = {'name': 'abl_cub5_ns8', 'dataset': 'cub', 'k_shot': 5, 'config': 'configs/ablations/cub5_ns8.yaml', 'seeds': [42, 123, 456], 'n_classes': 200, 'method': 'abl_cub5_ns8', 'cross_eval': {}, 'desc': '消融[cub5] State dim Ns=8，3 种子'}


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

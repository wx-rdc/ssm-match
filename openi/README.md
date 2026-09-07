# 启智平台运行指南（SSM-Match 验证）

本目录存放启智 AI 协作平台（openi.pcl.ac.cn）的云脑任务适配文件。平台侧资源调研结论（2026-09）：

> 本目录 `boots/` 中的逐任务启动文件由 `tools/gen_openi_boots.py` 生成
> （清单见 `boots/MANIFEST.md`），可按实验矩阵需要重新生成或增删。

## 已就绪的平台资源

| 资源 | 内容 |
|------|------|
| 预训练模型 | `edwardzhu/vim-tiny-midclstok`（公开，231.7MB）：Vim-Tiny ImageNet-1K 权重，含 `vim_t_midclstok_ft_78p3acc.pth`（论文采用，Top-1 78.3%）与 `vim_t_midclstok_76p1acc.pth` |
| 预编译 wheel | `edwardzhu/cu118-torch210-wheels`（公开，约 427MB）：`causal_conv1d-1.5.0` 与 `mamba_ssm-2.2.2`，匹配 torch2.1+cu118+py310。PyPI 上这两个包只有源码包，编译需 nvcc 且慢，**建任务时把该模型一并挂载**，env_check.py 会自动探测并离线安装 |
| miniImageNet | 公开数据集 `HeartTo/mini-imagenet`（3.08GB，100 类 jpg，按 ImageNet 类号命名；**不含** 64/16/20 划分文件，需配套 Ravi & Larochelle 标准 split CSV，置于代码仓内） |
| CUB-200-2011 | 公开数据集 `NJartist/CUB_200_2011`（1.25GB，官方目录结构） |
| Stanford Cars | 公开数据集 `Epiphany_0304/StanfordCars`（1.99GB；备选 `shisan/standfordcars`，MIT） |
| CIFAR-100（跨域用） | 公开数据集 `Open_Dataset/cifar-100-python`（0.19GB）；CIFAR-FS 平台无现成版本，用 Bertinetto 标准划分列表从 CIFAR-100 重构 |

## 任务配置推荐

- **镜像**（英伟达GPU）：首选 `ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0`（调试+训练均可用，与 Vim 官方 cu118 环境一致，mamba-ssm 有预编译 wheel）；备选 `ubuntu22.04-cuda12.1.0-py310-torch2.1.2-tf2.14`。
- **算力**：RTX 3090 / V100 / A100 / T4 均满足（论文设定为单张 3090，峰值显存约 3.2GB）。勿选昇腾等国产卡（mamba-ssm CUDA kernel 不兼容）。
- **启动文件**：`openi/env_check.py`（先跑通环境自检，再替换为正式训练入口）。
- **预训练模型**：任务创建时挂载模型 `edwardzhu/vim-tiny-midclstok`。
- **数据集**：创建任务时在"选择数据集文件"中搜索挂载上表公开数据集；若账号下不可选他人数据集，用 CLI 搬运到自己仓库：

```bash
openi login --token <OPENI_TOKEN>
openi dataset download HeartTo/mini-imagenet mini-imagenet -s ./data
openi dataset upload ./data <owner>/<repo> mini-imagenet
```

## 平台训练流程（SSM-Match 已实现）

平台端到端流程：

1. **调试任务**（首次验证，启动文件留空，进 JupyterLab 后执行）：
   ```bash
   cd /tmp/code/ssm-match
   pip install -r requirements.txt --no-build-isolation \
     -i https://mirrors.aliyun.com/pypi/simple/   # mamba-ssm 已挂 wheel 模型时走离线安装
   python tools/smoke_test.py --model_dir /tmp/pretrainmodel/vim-tiny-midclstok
   ```
   预期打印 `SMOKE PASS`（真实骨干前向 + 3 次训练步 + 显存峰值）。
2. **训练任务**（正式训练）：
   - 镜像 `ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0`；
   - 数据集挂载 `mini-imagenet`（HeartTo）；模型同时挂载 `vim-tiny-midclstok` 与 `cu118-torch210-wheels`；
   - 启动文件 `tools/train.py`（自动读 c2net 路径与 configs/mini_1shot.yaml）。
3. **断点续训**：任务被停后新建任务并保持输出目录不变，启动文件会 `--resume auto`
   自动找 `latest.pt` 续训；平台结果 30 天有效，重要检查点及时从页面下载。
4. **生成 CIFAR-FS 划分**（跨域评测前一次性执行）：
   ```bash
   python scripts/prepare_cifar_fs.py --cifar_root /tmp/dataset/cifar-100-python \
     --out data/splits/cifar_fs
   ```

## 容器内路径（c2net 库统一）

不同镜像/任务类型的挂载根路径不同：**新版计算任务容器在 `/tmp` 下**（`/tmp/code`、`/tmp/dataset`、`/tmp/pretrainmodel`、`/tmp/output`），老版容器在 `/code`、`/dataset`、`/pretrainmodel`、`/output`。用 c2net 库则无需关心差异：

```python
from c2net.context import prepare, upload_output
ctx = prepare()
ctx.code_path + '/ssm-match/...'
ctx.dataset_path + '/<数据集名>/...'        # 挂载的数据集
ctx.pretrain_model_path + '/vim-tiny-midclstok/...'  # 挂载的 Vim 权重
ctx.output_path                             # 结果必须写这里
upload_output()                             # 结束前回传，否则页面取不到结果；结果仅保留 30 天
```

调试任务 Terminal 手动执行示例（新版容器）：

```bash
cd /tmp/code/ssm-match
python openi/env_check.py                          # 走 c2net
python openi/env_check.py --model_dir /tmp/pretrainmodel/vim-tiny-midclstok  # 兜底
```

注意：平台向启动脚本注入额外命令行参数，argparse 必须使用 `parse_known_args()`。

## 本地验证

```bash
python openi/env_check.py --model_dir /path/to/vim-tiny-midclstok
```

通过标准：报告中 `cuda_available=true`、`mamba_forward_ok=true`、checkpoint 参数量约 7.9M（SSM-Match 全框架冻结参数 8.7M）、`verdict=PASS`。

## openi CLI 要点（v3.0.1）

- 模型/数据集已与代码仓解耦：`openi model upload` 的 `repo_id` 参数是 `拥有者/模型名`（AI 模型实体），不是代码仓名；模型实体需先在网页端"新建模型"或经 API 创建。
- 下载 HuggingFace 权重：`HF_ENDPOINT=https://hf-mirror.com hf download hustvl/Vim-tiny-midclstok --local-dir <目录>`。

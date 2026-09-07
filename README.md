# SSM-Match

Official PyTorch implementation of **"SSM-Match: Bridging State Space Models and Metric
Learning for Few-Shot Fine-Grained Image Classification"**.

SSM-Match keeps a pretrained **Vision Mamba (Vim-Tiny)** backbone frozen and learns, on top
of it, a **cross-image state-space matcher**: multi-scale token features of the support and
query images are concatenated into a single scan sequence and processed by a shared Mamba
block with bidirectional scanning, a direction gate, and an evidence gate. Matching is done
by evidence-weighted cosine similarity, so the SSM acts as a learned *cross-image metric*
rather than a per-image encoder. Episodic training adds a contrastive (InfoNCE) auxiliary
loss to structure the metric space.

The paper evaluates 5-way 1-shot / 5-shot classification on **miniImageNet**,
**CUB-200-2011**, **Stanford Cars**, and the cross-domain transfer
**miniImageNet → CIFAR-FS**. See the paper for full results.

## Repository layout

```
ssm_match/        Core package: backbone, data, engine, models, utils
tools/            Entry points: train / eval / smoke test / analysis utilities
configs/          Per-dataset configs and factorized ablation configs (YAML)
scripts/          Dataset preparation (split generation, class-folder merge)
data/splits/      Class-level split files used in the paper
openi/            OpenI (openi.pcl.ac.cn) cloud-task adapters + environment check
tests/            CPU unit tests
```

## Requirements

- Python 3.10, PyTorch 2.1 + CUDA 11.8 (tested configuration); a single CUDA GPU with
  >= 6 GB memory is sufficient (paper runs used one RTX 3090).
- Python dependencies are listed in `requirements.txt`. Note that `mamba-ssm` and
  `causal-conv1d` are CUDA extensions: install wheels that match your torch/CUDA build,
  e.g.
  ```bash
  pip install -r requirements.txt --no-build-isolation
  ```

## Backbone checkpoint

The frozen backbone is Vim-Tiny (midclstok variant, ImageNet-1K fine-tuned, 78.3% top-1)
from the official [Vision Mamba](https://github.com/hustvl/Vim) release:

```bash
huggingface-cli download hustvl/Vim-tiny-midclstok --local-dir data/vim-tiny-midclstok
# file used: vim_t_midclstok_ft_78p3acc.pth
```

## Datasets

| Dataset | Source | Loader layout |
|---|---|---|
| miniImageNet | verified mirror: [edwardzhu/mini-imagenet-ravi](https://www.modelscope.cn/datasets/edwardzhu/mini-imagenet-ravi) (zip with standard Ravi filenames + 64/16/20 class-split CSVs) | flat `data/mini-imagenet/images/*.jpg` (100 classes x 600 images) |
| CUB-200-2011 | verified mirror: [OpenDataLab/CUB-200-2011](https://www.modelscope.cn/datasets/OpenDataLab/CUB-200-2011) (official archive; 200 classes, 11,788 images) | `<cub_root>/images/<class>/*.jpg` (200 classes) |
| Stanford Cars | verified mirror: [edwardzhu/stanford-cars-196cls](https://www.modelscope.cn/datasets/edwardzhu/stanford-cars-196cls) (merged 196-class folders, 16,185 images) | `<cars_root>/<class>/*.jpg` |
| CIFAR-100 (for CIFAR-FS) | verified mirror: [OpenDataLab/CIFAR-100](https://www.modelscope.cn/datasets/OpenDataLab/CIFAR-100) (`cifar-100-python.tar.gz`, byte-identical to the official release) | `data/cifar-100-python/` (batch files) |

Each mirror's README on ModelScope contains the exact extraction/flatten commands and the
verification results (file counts / class counts). Extract datasets under `data/` (git-ignored).

## Splits

| Dataset | Split | How to obtain |
|---|---|---|
| miniImageNet | Ravi & Larochelle 64/16/20 **class** split | shipped: `data/splits/mini/{train,val,test}.csv` |
| Stanford Cars | deterministic 130/17/49 class split (seed 42) | shipped: `data/splits/cars/{train,val,test}.txt` |
| CUB-200-2011 | deterministic 100/50/50 class split (seed 42) | shipped: `data/splits/cub/{train,val,test}.txt` |
| CIFAR-FS | Bertinetto standard 12/4/4 superclass split (60/20/20 classes) | shipped: `data/splits/cifar_fs/{train,val,test}.txt` |

CUB/Cars have no official few-shot class split; we use fixed-seed deterministic splits
(identical across all seeds/runs, so all reported numbers share one split).

## Quick start

```bash
# 1. smoke test (backbone forward + full matching path; no dataset needed)
python tools/smoke_test.py --model_dir data/vim-tiny-midclstok

# 2. train (5-way 1-shot miniImageNet, seed 42)
python tools/train.py --config configs/mini_1shot.yaml --dataset mini \
    --data_root data/mini-imagenet --model_dir data/vim-tiny-midclstok \
    --output_dir output/mini_1shot_seed42

# 3. evaluate (10,000 episodes, evaluation seed 12345, decoupled from training seeds)
python tools/eval.py --run_dir output/mini_1shot_seed42 --dataset mini \
    --data_root data/mini-imagenet --model_dir data/vim-tiny-midclstok
```

- Training runs 200 epochs of episodic learning (AdamW, cosine annealing; see the YAML
  configs). `--resume auto` continues from `latest.pt` in the output directory.
- `tools/eval.py` reports mean accuracy with 95% confidence intervals over 10,000 sampled
  episodes; `tools/stats_tests.py` adds paired significance tests across seeds.
- Ablations (scan direction, prior/attention separation, direction & evidence gates,
  state size, scale subsets, sigmoid scoring) are expressed as factorized switches in
  `configs/ablations/*.yaml`.

## Tests

```bash
python -m pytest tests/ -q
```

## Cloud execution (OpenI)

`openi/` contains adapters for the [OpenI](https://openi.pcl.ac.cn) cloud platform:
`env_check.py` verifies GPU/CUDA, the Mamba CUDA kernels, and the Vim checkpoint;
`tools/gen_openi_boots.py` regenerates the per-task launcher files in `openi/boots/`.
`ssm_match/platform.py` auto-detects the platform context and falls back to plain CLI
arguments locally.

## License

Released under the [MIT License](LICENSE).

## Citation

```bibtex
@article{zhu2026ssmmatch,
  title   = {SSM-Match: Bridging State Space Models and Metric Learning for
             Few-Shot Fine-Grained Image Classification},
  author  = {Zhu, Hongyu and Liu, Gui and Li, Rong},
  journal = {IEEE Access},
  year    = {2026}
}
```

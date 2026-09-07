# 启智训练任务清单（由 tools/gen_openi_boots.py 生成）

镜像统一选 `ubuntu22.04-cuda11.8.0-py310-torch2.1.0-tf2.14.0`；预训练模型统一挂 vim-tiny-midclstok + cu118-torch210-wheels。
「数据集挂载」含 cifar-100-python 的任务仅为 mini 主结果（跨域评测用）。

| 任务名（建议） | 启动文件 | 数据集挂载 | 种子数 | 内容 |
|---|---|---|---|---|
| ssm-main-mini-1shot | openi/boots/main_mini_1shot.py | mini-imagenet + cifar-100-python | 5 | 主结果：主结果 mini 1-shot，5 种子，训练+终评+CIFAR-FS 跨域评测 |
| ssm-main-mini-5shot | openi/boots/main_mini_5shot.py | mini-imagenet + cifar-100-python | 5 | 主结果：主结果 mini 5-shot，5 种子，训练+终评+CIFAR-FS 跨域评测 |
| ssm-main-cub-1shot | openi/boots/main_cub_1shot.py | CUB_200_2011 | 5 | 主结果：主结果 cub 1-shot，5 种子，训练+终评 |
| ssm-main-cub-5shot | openi/boots/main_cub_5shot.py | CUB_200_2011 | 5 | 主结果：主结果 cub 5-shot，5 种子，训练+终评 |
| ssm-main-cars-1shot | openi/boots/main_cars_1shot.py | StanfordCars | 5 | 主结果：主结果 cars 1-shot，5 种子，训练+终评 |
| ssm-main-cars-5shot | openi/boots/main_cars_5shot.py | StanfordCars | 5 | 主结果：主结果 cars 5-shot，5 种子，训练+终评 |
| ssm-abl-mini1-no-prior | openi/boots/abl_mini1_no_prior.py | mini-imagenet | 3 | 消融：消融[mini1] w/o class prior token，3 种子 |
| ssm-abl-mini1-no-sep | openi/boots/abl_mini1_no_sep.py | mini-imagenet | 3 | 消融：消融[mini1] w/o [SEP] separators，3 种子 |
| ssm-abl-mini1-fwd | openi/boots/abl_mini1_fwd.py | mini-imagenet | 3 | 消融：消融[mini1] Forward scan only，3 种子 |
| ssm-abl-mini1-bwd | openi/boots/abl_mini1_bwd.py | mini-imagenet | 3 | 消融：消融[mini1] Backward scan only，3 种子 |
| ssm-abl-mini1-no-dirgate | openi/boots/abl_mini1_no_dirgate.py | mini-imagenet | 3 | 消融：消融[mini1] Symmetric fusion，3 种子 |
| ssm-abl-mini1-no-evigate | openi/boots/abl_mini1_no_evigate.py | mini-imagenet | 3 | 消融：消融[mini1] w/o evidence gate，3 种子 |
| ssm-abl-mini1-sigmoid | openi/boots/abl_mini1_sigmoid.py | mini-imagenet | 3 | 消融：消融[mini1] Sigmoid score head，3 种子 |
| ssm-abl-mini1-scale-f1 | openi/boots/abl_mini1_scale_f1.py | mini-imagenet | 3 | 消融：消融[mini1] Single scale 28x28，3 种子 |
| ssm-abl-mini1-scale-f2 | openi/boots/abl_mini1_scale_f2.py | mini-imagenet | 3 | 消融：消融[mini1] Single scale 14x14，3 种子 |
| ssm-abl-mini1-scale-f3 | openi/boots/abl_mini1_scale_f3.py | mini-imagenet | 3 | 消融：消融[mini1] Single scale 7x7，3 种子 |
| ssm-abl-mini1-ns8 | openi/boots/abl_mini1_ns8.py | mini-imagenet | 3 | 消融：消融[mini1] State dim Ns=8，3 种子 |
| ssm-abl-mini1-ns32 | openi/boots/abl_mini1_ns32.py | mini-imagenet | 3 | 消融：消融[mini1] State dim Ns=32，3 种子 |
| ssm-abl-cub5-no-prior | openi/boots/abl_cub5_no_prior.py | CUB_200_2011 | 3 | 消融：消融[cub5] w/o class prior token，3 种子 |
| ssm-abl-cub5-no-sep | openi/boots/abl_cub5_no_sep.py | CUB_200_2011 | 3 | 消融：消融[cub5] w/o [SEP] separators，3 种子 |
| ssm-abl-cub5-fwd | openi/boots/abl_cub5_fwd.py | CUB_200_2011 | 3 | 消融：消融[cub5] Forward scan only，3 种子 |
| ssm-abl-cub5-bwd | openi/boots/abl_cub5_bwd.py | CUB_200_2011 | 3 | 消融：消融[cub5] Backward scan only，3 种子 |
| ssm-abl-cub5-no-dirgate | openi/boots/abl_cub5_no_dirgate.py | CUB_200_2011 | 3 | 消融：消融[cub5] Symmetric fusion，3 种子 |
| ssm-abl-cub5-no-evigate | openi/boots/abl_cub5_no_evigate.py | CUB_200_2011 | 3 | 消融：消融[cub5] w/o evidence gate，3 种子 |
| ssm-abl-cub5-sigmoid | openi/boots/abl_cub5_sigmoid.py | CUB_200_2011 | 3 | 消融：消融[cub5] Sigmoid score head，3 种子 |
| ssm-abl-cub5-scale-f1 | openi/boots/abl_cub5_scale_f1.py | CUB_200_2011 | 3 | 消融：消融[cub5] Single scale 28x28，3 种子 |
| ssm-abl-cub5-scale-f2 | openi/boots/abl_cub5_scale_f2.py | CUB_200_2011 | 3 | 消融：消融[cub5] Single scale 14x14，3 种子 |
| ssm-abl-cub5-scale-f3 | openi/boots/abl_cub5_scale_f3.py | CUB_200_2011 | 3 | 消融：消融[cub5] Single scale 7x7，3 种子 |
| ssm-abl-cub5-ns8 | openi/boots/abl_cub5_ns8.py | CUB_200_2011 | 3 | 消融：消融[cub5] State dim Ns=8，3 种子 |
| ssm-abl-cub5-ns32 | openi/boots/abl_cub5_ns32.py | CUB_200_2011 | 3 | 消融：消融[cub5] State dim Ns=32，3 种子 |

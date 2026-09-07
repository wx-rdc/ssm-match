"""episode 采样：n-way k-shot（每类 15 查询），batch 内多个 episode 并行。

类目采样用独立 RNG（numpy Generator），与模型训练 RNG 分离，保证断点续训后
epoch→episode 序列可完全重现。
"""

import torch
from PIL import Image
from torch.utils.data import Dataset


class EpisodeSampler:
    def __init__(self, class_images: dict, n_way: int, k_shot: int, n_query: int = 15,
                 seed: int = 0):
        self.classes = sorted(class_images)
        self.class_images = [class_images[c] for c in self.classes]
        self.n_way, self.k_shot, self.n_query = n_way, k_shot, n_query
        self.rng = __import__("numpy").random.default_rng(seed)

    def sample_episode(self) -> tuple:
        chosen = self.rng.choice(len(self.classes), self.n_way, replace=False)
        support, query = [], []
        per_class_query = []
        for c in chosen:
            imgs = self.class_images[c]
            idx = self.rng.choice(len(imgs), self.k_shot + self.n_query, replace=False)
            support.append([imgs[i] for i in idx[:self.k_shot]])
            query.append([imgs[i] for i in idx[self.k_shot:]])
            per_class_query.append(len(idx[self.k_shot:]))
        return support, query, per_class_query


class EpisodeDataset(Dataset):
    """把 sampler 采出的样本列表缓存为可索引样本，由 collate 组装 batch。

    每个元素：(episode_id, role, class_id, sample, transform)。
    sample 为图像路径（str，mini/class-folder）或内存 uint8 张量 (3,H,W)
    （CIFAR-FS，见 data/cifar_fs.py）。
    DataLoader shuffle=False，batch_size = 每批 episode 数 × 单集图像数。
    """

    def __init__(self, sampler: EpisodeSampler, episodes: int, transform):
        self.sampler = sampler
        self.episodes = episodes
        self.transform = transform
        self.index = []
        plans = []
        for e in range(episodes):
            support, query, n_q = sampler.sample_episode()
            plans.append((support, query))
            for ci, sup in enumerate(support):
                for p in sup:
                    self.index.append((e, "s", ci, p))
            for ci, qry in enumerate(query):
                for p in qry:
                    self.index.append((e, "q", ci, p))
        self.plans = plans

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, role, ci, sample = self.index[i]
        if isinstance(sample, str):
            img = self.transform(Image.open(sample).convert("RGB"))
        else:  # 内存 uint8 (3,H,W)（CIFAR-FS）
            from torchvision.transforms.functional import to_pil_image
            img = self.transform(to_pil_image(sample))
        return e, role, ci, img


class EpisodeBatcher:
    """把 Dataset 输出重组为模型输入：support/quer 多尺度特征之外的原始张量字典。"""

    def __init__(self, n_way: int):
        self.n_way = n_way

    def collate(self, batch):
        episodes = {}
        for e, role, ci, img in batch:
            d = episodes.setdefault(e, {"s": {}, "q": {}})
            d[role].setdefault(ci, []).append(img)
        out_s, out_q = [], []
        for e in sorted(episodes):
            d = episodes[e]
            sup = [torch.stack(d["s"][ci]) for ci in sorted(d["s"])]
            qry = [torch.stack(d["q"][ci]) for ci in sorted(d["q"])]
            out_s.append(torch.stack(sup))          # (n,K,3,H,W)
            out_q.append(torch.cat(qry))            # (n*nq,3,H,W)
        return torch.stack(out_s), out_q            # 每类查询数恒为 n_query

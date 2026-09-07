"""图像变换：论文协议——resize 224，训练加随机裁剪/翻转/颜色抖动。"""

import torchvision.transforms as T

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def build_transform(train: bool) -> T.Compose:
    if train:
        return T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=8),
            T.RandomHorizontalFlip(),
            T.ColorJitter(0.2, 0.2, 0.2),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

# 第1部分：读取数据和图像预处理
import sys, json, random, subprocess, gc, copy
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from IPython.display import display
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED, N_FOLDS, EPOCHS = 2026, 3, 160
SAMPLES_PER_EPOCH, BATCH, WORKERS = 768, 16, 2
LR, BASE, PATIENCE, MIN_EPOCHS = 8e-4, 24, 35, 80
LOG_EVERY = 5
# 三折统一使用192×192的图块。
PLANS = [{'patch': 192, 'seed': 2026}]
INFER_BATCH, EMA_DECAY = 24, 0.995
DATA_ROOT = None
OUT = Path('/kaggle/working')
RUN = OUT / 'vessel_native_3models'
RUN.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEVICE.type == 'cuda'
torch.backends.cudnn.benchmark = AMP
cv2.setNumThreads(0)

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if AMP:
        torch.cuda.manual_seed_all(seed)

seed_all(SEED)
if DATA_ROOT is None:
    roots = sorted({p.parent for p in Path('/kaggle/input').rglob('sample_submission.csv')
                    if all((p.parent / q).exists() for q in
                           ['train/image', 'train/label', 'test/image', 'segmentation_to_csv.py'])})
    assert len(roots) == 1, f'请设置 DATA_ROOT 为已挂载的 segmentation 目录：{roots}'
    ROOT = roots[0].resolve()
else:
    ROOT = Path(DATA_ROOT).resolve()

def index_images(folder):
    paths = [p for p in folder.iterdir()
             if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp', '.gif'}]
    assert len(paths) == len({p.stem for p in paths})
    return {p.stem: p for p in paths}

train_files = index_images(ROOT / 'train/image')
label_files = index_images(ROOT / 'train/label')
test_files = index_images(ROOT / 'test/image')
sample = pd.read_csv(ROOT / 'sample_submission.csv', dtype={'Id': str}, keep_default_na=False)
train_ids = sorted(train_files, key=int)
test_ids = sample['Id'].tolist()
assert len(train_ids) == len(label_files) == len(test_files) == len(test_ids) == 20
assert set(train_ids) == set(label_files)
assert len(set(test_ids)) == 20 and set(test_ids) == set(test_files)

def read_rgb(path):
    # 部分.jpg文件实际是TIFF或GIF，用Pillow按内容读取。
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
def make_features(rgb):
    # 输入为RGB、CLAHE增强绿通道和亮度校正绿通道，共5个通道。
    g = rgb[..., 1].astype(np.float32) / 255.0
    enhanced = clahe.apply(rgb[..., 1]).astype(np.float32) / 255.0
    background = cv2.GaussianBlur(g, (0, 0), sigmaX=9)
    corrected = np.clip(0.5 * g / (background + 0.02), 0, 1)
    fov = (rgb.max(axis=-1) > 15).astype(np.uint8)
    fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    features = np.concatenate([rgb.astype(np.float32) / 255.0,
                                enhanced[..., None], corrected[..., None]], axis=-1)
    return features, fov.astype(np.float32)

def official_resize(binary):
    # 按官方脚本用双线性插值缩放到512×512，再取非零像素。
    white = Image.fromarray(np.asarray(binary, dtype=np.uint8) * 255)
    return np.asarray(white.resize((512, 512), Image.Resampling.BILINEAR)) > 0

images, masks, fovs, metric_masks = [], [], [], []
for sid in train_ids:
    rgb = read_rgb(train_files[sid])
    with Image.open(label_files[sid]) as im:
        gray = np.asarray(im.convert('L'))
    assert gray.shape == rgb.shape[:2]
    assert set(np.unique(gray)).issubset({0, 255}), f'标签不是纯二值：{sid}'
    mask = (gray > 127).astype(np.float32)
    x, fov = make_features(rgb)
    images.append(x)
    masks.append(mask)
    fovs.append(fov)
    metric_masks.append(official_resize(mask > 0.5))
assert 0.01 < np.mean([m.mean() for m in masks]) < 0.5
metric_masks = np.stack(metric_masks)

def tensor_image(x):
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1) * 2 - 1))

class VesselPatches(Dataset):
    def __init__(self, indices, patch):
        self.indices, self.patch = np.asarray(indices), patch
        self.points = {int(i): np.argwhere(masks[i] > 0.5) for i in indices}
        self.retina = {int(i): np.argwhere(fovs[i] > 0.5) for i in indices}

    def __len__(self):
        return SAMPLES_PER_EPOCH

    def __getitem__(self, _):
        i, p = int(np.random.choice(self.indices)), self.patch
        h, w = masks[i].shape
        assert min(h, w) >= p
        r = np.random.rand()
        points = self.points[i] if r < 0.6 else self.retina[i]
        if r < 0.95 and len(points):
            cy, cx = points[np.random.randint(len(points))]
            y, x = int(np.clip(cy - p // 2, 0, h - p)), int(np.clip(cx - p // 2, 0, w - p))
        else:
            y, x = np.random.randint(h - p + 1), np.random.randint(w - p + 1)
        a, b = images[i][y:y+p, x:x+p].copy(), masks[i][y:y+p, x:x+p].copy()
        k = np.random.randint(4)
        a, b = np.rot90(a, k), np.rot90(b, k)
        if np.random.rand() < 0.5:
            a, b = a[:, ::-1], b[:, ::-1]
        a, b = np.ascontiguousarray(a), np.ascontiguousarray(b)
        if np.random.rand() < 0.5:
            matrix = cv2.getRotationMatrix2D((p / 2, p / 2),
                                             np.random.uniform(-20, 20),
                                             np.random.uniform(0.85, 1.15))
            a = cv2.warpAffine(a, matrix, (p, p), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
            b = cv2.warpAffine(b, matrix, (p, p), flags=cv2.INTER_NEAREST,
                               borderMode=cv2.BORDER_REFLECT_101)
        if np.random.rand() < 0.8:
            a[..., :3] = np.clip(a[..., :3] ** np.random.uniform(0.8, 1.2)
                                 * np.random.uniform(0.9, 1.1), 0, 1)
        return tensor_image(a), torch.from_numpy(np.ascontiguousarray(b[None]))

def seed_worker(_):
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)

folds = np.array_split(np.random.default_rng(SEED).permutation(20), N_FOLDS)
fold_table = pd.DataFrame({'Id': train_ids, 'fold': -1})
for fold, indices in enumerate(folds):
    fold_table.loc[indices, 'fold'] = fold + 1
fold_table.to_csv(RUN / 'fold_assignments.csv', index=False)
print('Device:', DEVICE, '| train=20 | test=20 | 原图尺寸：', masks[0].shape)
print(f'只用附件数据；无外部数据、无预训练权重。'
      f'{len(PLANS)} 种裁剪尺度 × {N_FOLDS} 折，共 {len(PLANS) * N_FOLDS} 个模型。')
if not AMP:
    print('请开启 GPU。')



# 第2部分：网络、损失函数和滑窗预测
# soft-clDice参考论文：https://arxiv.org/abs/2003.07311
class SE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.gate = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c, max(c // 8, 8), 1),
                                   nn.SiLU(), nn.Conv2d(max(c // 8, 8), c, 1), nn.Sigmoid())
    def forward(self, x):
        return x * self.gate(x)

class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(ci, co, 3, padding=1, bias=False),
                                   nn.GroupNorm(8, co), nn.SiLU(),
                                   nn.Conv2d(co, co, 3, padding=1, bias=False),
                                   nn.GroupNorm(8, co), SE(co))
        self.skip = nn.Conv2d(ci, co, 1, bias=False) if ci != co else nn.Identity()
    def forward(self, x):
        return F.silu(self.body(x) + self.skip(x))

class VesselNet(nn.Module):
    def __init__(self):
        super().__init__()
        b = BASE
        self.e0, self.e1 = Block(5, b), Block(b, b * 2)
        self.e2, self.e3 = Block(b * 2, b * 4), Block(b * 4, b * 8)
        self.d2, self.d1, self.d0 = Block(b * 12, b * 4), Block(b * 6, b * 2), Block(b * 3, b)
        self.dropout = nn.Dropout2d(0.1)
        self.head = nn.Conv2d(b, 1, 1)
        self.aux1, self.aux2 = nn.Conv2d(b * 2, 1, 1), nn.Conv2d(b * 4, 1, 1)

    def forward(self, x):
        e0 = self.e0(x)
        e1 = self.e1(F.max_pool2d(e0, 2))
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.dropout(self.e3(F.max_pool2d(e2, 2)))
        up = lambda a, b: F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.d2(torch.cat([up(e3, e2), e2], 1))
        d1 = self.d1(torch.cat([up(d2, e1), e1], 1))
        d0 = self.d0(torch.cat([up(d1, e0), e0], 1))
        main = self.head(d0)
        # 辅助输出只用于训练。
        return (main, self.aux1(d1), self.aux2(d2)) if self.training else main

def soft_erode(x):
    a = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    b = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.minimum(a, b)

def soft_skeleton(x, steps=4):
    opened = F.max_pool2d(soft_erode(x), 3, stride=1, padding=1)
    skeleton = F.relu(x - opened)
    for _ in range(steps):
        x = soft_erode(x)
        opened = F.max_pool2d(soft_erode(x), 3, stride=1, padding=1)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton

def single_loss(logits, y, topology=True):
    logits = logits.float()
    p = logits.sigmoid()
    dims = (1, 2, 3)
    dice = (2 * (p * y).sum(dims) + 1) / (p.sum(dims) + y.sum(dims) + 1)
    bce = F.binary_cross_entropy_with_logits(logits, y)
    if not topology:
        return 0.5 * bce + 0.5 * (1 - dice.mean())
    sp, sy = soft_skeleton(p), soft_skeleton(y)
    precision = ((sp * y).sum(dims) + 1) / (sp.sum(dims) + 1)
    sensitivity = ((sy * p).sum(dims) + 1) / (sy.sum(dims) + 1)
    cldice = 2 * precision * sensitivity / (precision + sensitivity + 1e-7)
    return 0.4 * bce + 0.5 * (1 - dice.mean()) + 0.1 * (1 - cldice.mean())

def criterion(outputs, y):
    if not isinstance(outputs, (tuple, list)):
        return single_loss(outputs, y)
    loss = single_loss(outputs[0], y)
    for aux in outputs[1:]:
        target = F.interpolate(y, size=aux.shape[-2:], mode='area')
        loss = loss + 0.15 * single_loss(aux, target, topology=False)
    return loss

class EMA:
    def __init__(self, model):
        self.model = copy.deepcopy(model).eval()
        self.steps = 0
        for p in self.model.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        self.steps += 1
        decay = min(EMA_DECAY, 1 - 1 / (self.steps + 1))
        for averaged, current in zip(self.model.parameters(), model.parameters()):
            averaged.lerp_(current.detach(), 1 - decay)

def window_starts(length, patch, stride):
    assert length >= patch
    positions = list(range(0, length - patch + 1, stride))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions

@torch.inference_mode()
def predict_native(model, features, patch, tta=False):
    model.eval()
    h, w = features.shape[:2]
    halo = patch // 4
    padded = np.pad(features, ((halo, halo), (halo, halo), (0, 0)), mode='reflect')
    hh, ww = padded.shape[:2]
    coords = [(y, x) for y in window_starts(hh, patch, patch // 2)
              for x in window_starts(ww, patch, patch // 2)]
    # 给图块中心更大的权重，减轻拼接边缘的影响。
    one = np.hanning(patch).astype(np.float32)
    weight = np.maximum(np.outer(one, one), 0.01)
    total, count = np.zeros((hh, ww), np.float32), np.zeros((hh, ww), np.float32)
    for start in range(0, len(coords), INFER_BATCH):
        chunk = coords[start:start + INFER_BATCH]
        x = torch.stack([tensor_image(padded[y:y+patch, xx:xx+patch])
                          for y, xx in chunk]).to(DEVICE)
        probability = torch.zeros((len(chunk), 1, patch, patch), device=DEVICE)
        for transform in range(8 if tta else 1):
            k, flip = transform % 4, transform >= 4
            z = torch.rot90(x, k, (-2, -1))
            if flip:
                z = torch.flip(z, (-1,))
            with torch.autocast(device_type=DEVICE.type, enabled=AMP):
                pred = model(z).float().sigmoid()
            if flip:
                pred = torch.flip(pred, (-1,))
            probability += torch.rot90(pred, -k, (-2, -1))
        probability = (probability / (8 if tta else 1))[:, 0].cpu().numpy()
        for j, (y, xx) in enumerate(chunk):
            total[y:y+patch, xx:xx+patch] += probability[j] * weight
            count[y:y+patch, xx:xx+patch] += weight
    result = total / np.maximum(count, 1e-7)
    return result[halo:halo+h, halo:halo+w]

def remove_small(binary, min_size=0):
    binary = np.asarray(binary, dtype=np.uint8)
    if min_size <= 0:
        return binary > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = stats[:, cv2.CC_STAT_AREA] >= min_size
    keep[0] = False
    return keep[labels]

def output_mask(probability, threshold=0.5, min_size=0):
    return official_resize(remove_small(probability > threshold, min_size))

def dice_one(p, y):
    p, y = np.asarray(p, dtype=bool), np.asarray(y, dtype=bool)
    return float((2 * (p & y).sum() + 1e-7) / (p.sum() + y.sum() + 1e-7))

def evaluate_probabilities(probabilities, indices, threshold=0.5, min_size=0):
    return float(np.mean([dice_one(output_mask(p, threshold, min_size), metric_masks[i])
                          for p, i in zip(probabilities, indices)]))

def probability_loss(p, target):
    # 验证损失取BCE和Dice损失的平均。
    p = np.clip(p, 1e-6, 1 - 1e-6)
    bce = -np.mean(target * np.log(p) + (1 - target) * np.log(1 - p))
    dice = (2 * np.sum(p * target) + 1) / (p.sum() + target.sum() + 1)
    return float(0.5 * bce + 0.5 * (1 - dice))

@torch.inference_mode()
def validate(model, indices, patch):
    ps = [predict_native(model, images[i], patch) * fovs[i] for i in indices]
    score = evaluate_probabilities(ps, indices)
    loss = float(np.mean([probability_loss(p, masks[i]) for p, i in zip(ps, indices)]))
    return score, loss

# 用随机输入检查输出尺寸和反向传播。
_check = VesselNet().to(DEVICE).train()
_x = torch.rand(2, 5, 64, 64, device=DEVICE)
_y = (torch.rand(2, 1, 64, 64, device=DEVICE) > 0.9).float()
with torch.autocast(device_type=DEVICE.type, enabled=AMP):
    _outputs = _check(_x)
    _loss = criterion(_outputs, _y)
assert _outputs[0].shape == _y.shape and torch.isfinite(_loss).item()
_loss.backward()
del _check, _x, _y, _outputs, _loss
if AMP:
    torch.cuda.empty_cache()



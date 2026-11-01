"""YOLO v1 paper architecture and VOC pipeline, implemented in PyTorch.

Output layout: [class probabilities (20), box1 (x,y,w,h,c), box2 (...)].
No anchors, batch normalization, sigmoid, softmax, or pretrained modern YOLO.
See README.md for reproduction boundaries. Python 3.10+, PyTorch 2.3+.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import random
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.transforms import functional as TF
from torchvision.ops import batched_nms
from torchvision.datasets.utils import download_and_extract_archive

CLASSES = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')
S, B, C = 7, 2, 20
ARCHITECTURE = 'yolov1-paper-24conv-4096fc-noBN-v1'
VOC_ARCHIVES = {
    '2007': ('voc2007/VOCtrainval_06-Nov-2007.tar', 'c52e279531787c972589f7e41ab4ae64'),
    '2007-test': ('voc2007/VOCtest_06-Nov-2007.tar', 'b6e924de25625d8de591ea690078ad9f'),
    '2012': ('voc2012/VOCtrainval_11-May-2012.tar', '6cd6e144f989b92b3379bac3b3de84fd'),
}


@dataclass
class Config:
    mode: str = 'smoke'             # smoke / dev / paper
    data_root: str = '/content/voc'
    output_dir: str = '/content/yolov1_run'
    backbone_checkpoint: str = ''   # from pretrain_imagenet(), NOT Darknet .weights
    resume: str = ''                # our own last.pt, epoch-boundary resume
    epochs: int = 0                 # 0 -> preset (1 / 30 / 135)
    micro_batch: int = 2            # reduce to 1 if GPU memory is insufficient
    effective_batch: int = 64       # gradient accumulation, paper batch size
    workers: int = 2
    seed: int = 42
    amp: bool = True
    download: bool = True
    eval_every: int = 5
    eval_score: float = 0.001        # low threshold for AP, not visualization threshold
    nms_iou: float = 0.5            # implementation choice (not specified in paper)
    device: str = 'cuda'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_worker(_):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def conv(cin, cout, kernel, stride=1):
    return nn.Sequential(nn.Conv2d(cin, cout, kernel, stride, kernel // 2),
                         nn.LeakyReLU(0.1, inplace=True))


def first20():
    # Figure 3: 224 -> 7 for classification; 448 -> 14 for detection.
    layers = [conv(3, 64, 7, 2), nn.MaxPool2d(2, 2),
              conv(64, 192, 3), nn.MaxPool2d(2, 2),
              conv(192, 128, 1), conv(128, 256, 3),
              conv(256, 256, 1), conv(256, 512, 3), nn.MaxPool2d(2, 2)]
    for _ in range(4):
        layers += [conv(512, 256, 1), conv(256, 512, 3)]
    layers += [conv(512, 512, 1), conv(512, 1024, 3), nn.MaxPool2d(2, 2)]
    for _ in range(2):
        layers += [conv(1024, 512, 1), conv(512, 1024, 3)]
    return nn.Sequential(*layers)


def initialize(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, a=0.1, nonlinearity='leaky_relu')
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, std=0.01)
        nn.init.zeros_(module.bias)


class YOLOv1(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = first20()
        self.extra = nn.Sequential(conv(1024, 1024, 3), conv(1024, 1024, 3, 2),
                                   conv(1024, 1024, 3), conv(1024, 1024, 3))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(1024*7*7, 4096),
                                  nn.LeakyReLU(0.1, inplace=True), nn.Dropout(0.5),
                                  nn.Linear(4096, S*S*(C+B*5)))
        self.apply(initialize)
        # Small final outputs keep random-start smoke tests numerically manageable.
        nn.init.normal_(self.head[-1].weight, std=0.001)

    def forward(self, x):
        if x.shape[-2:] != (448, 448):
            raise ValueError('YOLO v1 detection input must be 448 x 448.')
        return self.head(self.extra(self.backbone(x))).reshape(-1, S, S, C+B*5)


class ImageNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = first20()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(1024, 1000)
        self.apply(initialize)

    def forward(self, x):
        return self.fc(self.pool(self.backbone(x)).flatten(1))


def load_backbone(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != ARCHITECTURE or checkpoint.get('task') != 'imagenet1000':
        raise ValueError('Use an ImageNet backbone checkpoint exported by this implementation.')
    model.backbone.load_state_dict(checkpoint['backbone'], strict=True)
    return checkpoint.get('provenance', {})


def xywh_to_xyxy(box):
    return torch.cat((box[..., :2] - box[..., 2:]/2,
                      box[..., :2] + box[..., 2:]/2), dim=-1)


def aligned_iou(a, b):
    """Broadcastable IoU for continuous, 0-based xyxy coordinates."""
    inter = (torch.minimum(a[..., 2:], b[..., 2:]) -
             torch.maximum(a[..., :2], b[..., :2])).clamp(min=0).prod(-1)
    area_a = (a[..., 2:]-a[..., :2]).clamp(min=0).prod(-1)
    area_b = (b[..., 2:]-b[..., :2]).clamp(min=0).prod(-1)
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def grid_boxes_to_xyxy(boxes):
    # boxes: [N, row, col, B, 4]; xy are cell-relative; wh image-relative.
    row, col = torch.meshgrid(torch.arange(S, device=boxes.device),
                              torch.arange(S, device=boxes.device), indexing='ij')
    grid = torch.stack((col, row), -1).to(boxes).reshape(1, S, S, 1, 2)
    xy = (boxes[..., :2] + grid) / S
    return xywh_to_xyxy(torch.cat((xy, boxes[..., 2:].clamp(min=0)), -1))


def signed_sqrt(x, eps=1e-6):
    # Linear outputs may be negative. This finite-gradient extension agrees
    # with sqrt(w) for positive w (up to eps) and pushes negative widths upward.
    return x / torch.sqrt(x.abs() + eps)


class YoloLoss(nn.Module):
    def forward(self, pred, target):
        pred, target = pred.float(), target.float()
        boxes = pred[..., C:].reshape(-1, S, S, B, 5)
        gt = target[..., :4].unsqueeze(-2)
        obj = target[..., 4]
        with torch.no_grad():
            iou = aligned_iou(grid_boxes_to_xyxy(boxes[..., :4]),
                              grid_boxes_to_xyxy(gt))
            winner = iou.argmax(-1)
            responsible = torch.nn.functional.one_hot(winner, B).float() * obj[..., None]
        xy = ((boxes[..., :2]-gt[..., :2]).square().sum(-1)*responsible).sum()
        wh = ((signed_sqrt(boxes[..., 2:4])-signed_sqrt(gt[..., 2:4])).square()
              .sum(-1)*responsible).sum()
        confidence = ((boxes[..., 4]-iou.detach()).square()*responsible).sum()
        # Non-responsible predictors, including the second predictor in occupied
        # cells, receive a no-object target. Box/class targets are masked separately.
        noobj = (boxes[..., 4].square()*(1-responsible)).sum()
        cls = ((pred[..., :C]-target[..., 5:]).square().sum(-1)*obj).sum()
        n = pred.shape[0]
        terms = {'xy': 5*xy/n, 'wh': 5*wh/n, 'obj': confidence/n,
                 'noobj': 0.5*noobj/n, 'class': cls/n}
        return sum(terms.values()), {k: float(v.detach()) for k, v in terms.items()}


def encode_targets(boxes, labels):
    """One GT per cell. Deterministically retain largest box on collisions.

    Never use this lossy encoding as the evaluation ground truth.
    """
    result = torch.zeros(S, S, 5+C)
    if not len(boxes):
        return result, 0
    wh = boxes[:, 2:]-boxes[:, :2]
    center = ((boxes[:, :2]+boxes[:, 2:])/2).clamp(0, 1-1e-7)
    order = (wh[:, 0]*wh[:, 1]).argsort(descending=True, stable=True)
    collisions = 0
    for idx in order.tolist():
        x, y = center[idx]*S
        col, row = int(x), int(y)
        if result[row, col, 4]:
            collisions += 1
            continue
        result[row, col, :4] = torch.stack((x-col, y-row, wh[idx, 0], wh[idx, 1]))
        result[row, col, 4] = 1
        result[row, col, 5+int(labels[idx])] = 1
    return result, collisions


def augment(image, boxes, labels, difficult):
    # Random crop/pad margins within 20% -> joint scale and translation.
    # Box coordinates follow the same geometric transformation as pixels.
    w, h = image.size
    left, right = [round(random.uniform(-0.2, 0.2)*w) for _ in range(2)]
    top, bottom = [round(random.uniform(-0.2, 0.2)*h) for _ in range(2)]
    nw, nh = w-left-right, h-top-bottom
    image = image.crop((left, top, w-right, h-bottom))
    boxes = boxes.clone()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]]*w-left)/nw
    boxes[:, [1, 3]] = (boxes[:, [1, 3]]*h-top)/nh
    boxes.clamp_(0, 1)
    keep = (boxes[:, 2:] > boxes[:, :2]).all(-1)
    boxes, labels, difficult = boxes[keep], labels[keep], difficult[keep]
    if random.random() < 0.5:
        image = TF.hflip(image)
        boxes[:, [0, 2]] = 1-boxes[:, [2, 0]]
    hsv = np.array(image.convert('HSV'), dtype=np.float32)
    for channel in (1, 2):
        scale = random.uniform(1, 1.5)
        if random.random() < 0.5:
            scale = 1/scale
        hsv[..., channel] = np.clip(hsv[..., channel]*scale, 0, 255)
    image = Image.fromarray(hsv.astype(np.uint8), mode='HSV').convert('RGB')
    return image, boxes, labels, difficult


class VOCDataset(Dataset):
    def __init__(self, root, splits, training=False, download=True, limit=None):
        self.sources, self.index = [], []
        self.training = training
        for year, split in splits:
            split_file = Path(root)/'VOCdevkit'/f'VOC{year}'/'ImageSets'/'Main'/f'{split}.txt'
            # Test/trainval share a VOC2007 directory; check split file, not directory.
            # Use official HTTPS URLs even on older Colab torchvision releases.
            if download and not split_file.exists():
                key = '2007-test' if (year, split) == ('2007', 'test') else year
                remote, md5 = VOC_ARCHIVES[key]
                download_and_extract_archive('https://thor.robots.ox.ac.uk/pascal/VOC/'+remote,
                                             root, filename=Path(remote).name, md5=md5)
            src = datasets.VOCDetection(root, year=year, image_set=split, download=False)
            self.index.extend((len(self.sources), i) for i in range(len(src)))
            self.sources.append((year, src))
        if limit is not None:
            # Fixed subset, independent from augmentation RNG.
            rng = random.Random(123)
            self.index = rng.sample(self.index, min(limit, len(self.index)))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        src_idx, item_idx = self.index[index]
        year, src = self.sources[src_idx]
        image, ann = src[item_idx]
        ann = ann['annotation']
        w, h = image.size
        objects = ann.get('object', [])
        if isinstance(objects, dict):
            objects = [objects]
        boxes, labels, difficult = [], [], []
        for obj in objects:
            b = obj['bndbox']
            # VOC is 1-based inclusive. Convert to 0-based continuous edges.
            box = [(float(b['xmin'])-1)/w, (float(b['ymin'])-1)/h,
                   float(b['xmax'])/w, float(b['ymax'])/h]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            labels.append(CLASSES.index(obj['name']))
            difficult.append(bool(int(obj.get('difficult', 0))))
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4).clamp(0, 1)
        labels = torch.tensor(labels, dtype=torch.long)
        difficult = torch.tensor(difficult, dtype=torch.bool)
        if self.training:
            boxes, labels = boxes[~difficult], labels[~difficult]
            difficult = difficult[~difficult]
            image, boxes, labels, difficult = augment(image, boxes, labels, difficult)
        target, collisions = encode_targets(boxes[~difficult], labels[~difficult])
        image = TF.to_tensor(image.resize((448, 448), Image.Resampling.BILINEAR))
        metadata = {'id': f"{year}/{Path(ann['filename']).stem}", 'boxes': boxes,
                    'labels': labels, 'difficult': difficult, 'collisions': collisions}
        return image, target, metadata


def collate(items):
    images, targets, metadata = zip(*items)
    return torch.stack(images), torch.stack(targets), list(metadata)


def make_loader(dataset, cfg, training=False):
    return DataLoader(dataset, batch_size=cfg.micro_batch, shuffle=training,
                      num_workers=cfg.workers, pin_memory=cfg.device.startswith('cuda'),
                      collate_fn=collate, worker_init_fn=seed_worker,
                      persistent_workers=False)


@torch.no_grad()
def decode(pred, score_threshold=0.001, nms_iou=0.5):
    """Class-aware NMS. Scores use raw linear outputs, never sigmoid/softmax."""
    pred = pred.float()
    boxes = pred[..., C:].reshape(-1, S, S, B, 5)
    xyxy = grid_boxes_to_xyxy(boxes[..., :4]).clamp(0, 1).reshape(-1, S*S*B, 4)
    # Negative probabilities/confidences are invalid; do not multiply two
    # negatives into a spuriously positive detection score.
    scores = (pred[..., :C].clamp(min=0).unsqueeze(-2)*
              boxes[..., 4:].clamp(min=0)).reshape(-1, S*S*B, C)
    outputs = []
    for coords, score in zip(xyxy, scores):
        valid_box = (coords[:, 2:] > coords[:, :2]).all(-1)
        bi, ci = torch.where((score > score_threshold) & valid_box[:, None])
        b, s = coords[bi], score[bi, ci]
        keep = batched_nms(b, s, ci, nms_iou)
        outputs.append({'boxes': b[keep].cpu(), 'scores': s[keep].cpu(),
                        'labels': ci[keep].cpu()})
    return outputs


def voc2007_ap(predictions, ground_truth):
    """VOC2007 11-point AP@IoU>0.5; difficult matches ignored; duplicates FP.

    Inputs are dictionaries keyed by unique image id. Evaluation uses ALL GT,
    including objects whose centers collide in the same YOLO cell.
    """
    ap, counts = {}, {}
    for cls, name in enumerate(CLASSES):
        truth, positive = {}, 0
        for ident, gt in ground_truth.items():
            mask = gt['labels'] == cls
            b, d = gt['boxes'][mask], gt['difficult'][mask]
            truth[ident] = (b, d, np.zeros(len(b), dtype=bool))
            positive += int((~d).sum())
        counts[name] = positive
        if positive == 0:
            ap[name] = None
            continue
        entries = []
        for ident, pred in predictions.items():
            for idx in torch.where(pred['labels'] == cls)[0].tolist():
                entries.append((float(pred['scores'][idx]), ident, pred['boxes'][idx]))
        entries.sort(key=lambda x: -x[0])
        tp, fp = [], []
        for _, ident, box in entries:
            b, d, matched = truth.get(ident, (torch.empty(0, 4), torch.empty(0, dtype=torch.bool), []))
            if not len(b):
                tp.append(0); fp.append(1)
                continue
            overlaps = aligned_iou(box, b)
            best = int(overlaps.argmax())
            if float(overlaps[best]) > 0.5:
                if d[best]:
                    continue
                if not matched[best]:
                    matched[best] = True
                    tp.append(1); fp.append(0)
                else:
                    tp.append(0); fp.append(1)
            else:
                tp.append(0); fp.append(1)
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp/positive
        precision = tp/np.maximum(tp+fp, 1e-12)
        ap[name] = float(sum(np.max(precision[recall >= t], initial=0)
                             for t in np.linspace(0, 1, 11))/11)
    valid = [v for v in ap.values() if v is not None]
    return {'mAP50_VOC2007_11point': float(np.mean(valid)) if valid else 0.0,
            'AP_per_class': ap, 'GT_per_class': counts, 'classes_with_gt': len(valid)}


def autocast(cfg):
    return torch.autocast('cuda', dtype=torch.float16) if cfg.amp and cfg.device.startswith('cuda') else contextlib.nullcontext()


@torch.no_grad()
def evaluate(model, loader, cfg):
    model.eval()
    predictions, truth = {}, {}
    for images, _, metadata in loader:
        with autocast(cfg):
            raw = model(images.to(cfg.device, non_blocking=True))
        outputs = decode(raw, cfg.eval_score, cfg.nms_iou)
        for out, gt in zip(outputs, metadata):
            predictions[gt['id']], truth[gt['id']] = out, gt
    result = voc2007_ap(predictions, truth)
    result['images'] = len(truth)
    return result


def learning_rate(epoch, fraction, cfg, has_pretrain):
    if cfg.mode == 'paper':
        # Warm-up is inside first 75 epochs -> total 75+30+30 = 135.
        # The paper does not specify exact warm-up duration; choose 1 epoch.
        if epoch == 0:
            return 0.001 + 0.009*fraction
        return 0.01 if epoch < 75 else (0.001 if epoch < 105 else 0.0001)
    # Practical short-run setting, explicitly different from paper schedule.
    base = 0.001 if has_pretrain else 0.0001
    progress = (epoch+fraction)/cfg.epochs
    return base*(1 if progress < 0.6 else (0.1 if progress < 0.85 else 0.01))


def train_epoch(model, loader, optimizer, scaler, cfg, epoch, has_pretrain):
    model.train()
    loss_fn = YoloLoss()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    seen = collisions = pending = optimizer_steps = skipped_steps = 0
    start = time.perf_counter()
    for step, (images, target, metadata) in enumerate(loader):
        images, target = images.to(cfg.device), target.to(cfg.device)
        with autocast(cfg):
            output = model(images)
        # Float32 loss outside autocast: sqrt, IoU and squared sums are sensitive.
        loss, terms = loss_fn(output, target)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at epoch={epoch+1}, step={step}: {terms}')
        n = len(images)
        scaler.scale(loss*n/cfg.effective_batch).backward()
        pending += n
        total += float(loss.detach())*n
        seen += n
        collisions += sum(m['collisions'] for m in metadata)
        if pending == cfg.effective_batch or step+1 == len(loader):
            scaler.unscale_(optimizer)
            # Last partial effective batch must also be an image-average loss.
            if pending != cfg.effective_batch:
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(cfg.effective_batch/pending)
            lr = learning_rate(epoch, (step+1)/len(loader), cfg, has_pretrain)
            for group in optimizer.param_groups:
                group['lr'] = lr
            previous_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() < previous_scale:
                skipped_steps += 1
            else:
                optimizer_steps += 1
            optimizer.zero_grad(set_to_none=True)
            pending = 0
        if step % 25 == 0 or step+1 == len(loader):
            print(f'epoch {epoch+1}/{cfg.epochs} batch {step+1}/{len(loader)} '
                  f'loss={total/seen:.4f} lr={optimizer.param_groups[0]["lr"]:.6g}', flush=True)
    return {'train_loss': total/seen, 'collision_objects_skipped': collisions,
            'optimizer_steps': optimizer_steps, 'amp_skipped_steps': skipped_steps,
            'seconds': time.perf_counter()-start}


def rng_state():
    return {'python': random.getstate(), 'torch': torch.get_rng_state(),
            'numpy': np.random.get_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python'])
    torch.set_rng_state(state['torch'])
    np.random.set_state(state['numpy'])
    if torch.cuda.is_available() and state['cuda']:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(obj, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(obj, temporary)
    os.replace(temporary, path)


def write_json(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def run(cfg):
    if cfg.mode not in ('smoke', 'dev', 'paper'):
        raise ValueError('mode must be smoke, dev or paper')
    if not cfg.epochs:
        cfg.epochs = {'smoke': 1, 'dev': 30, 'paper': 135}[cfg.mode]
    if cfg.epochs < 1 or cfg.eval_every < 1:
        raise ValueError('epochs and eval_every must be positive')
    if cfg.mode == 'paper' and cfg.epochs != 135:
        raise ValueError('paper mode requires 135 epochs (75+30+30)')
    if cfg.micro_batch < 1 or cfg.effective_batch % cfg.micro_batch:
        raise ValueError('effective_batch must be divisible by micro_batch')
    if cfg.mode == 'paper' and cfg.effective_batch != 64:
        raise ValueError('paper mode requires effective_batch=64')
    if cfg.mode == 'paper' and not (cfg.backbone_checkpoint or cfg.resume):
        raise ValueError('paper mode requires ImageNet pretraining: supply backbone_checkpoint or resume.')
    if cfg.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('Select a GPU runtime in Colab. CPU is only for explicit local tests.')
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir/'last.pt').exists() and not cfg.resume:
        raise FileExistsError('Existing run found: set resume to last.pt or choose a new output_dir.')
    seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = False
    train_splits = [('2007', 'train')] if cfg.mode == 'smoke' else [('2007', 'train'), ('2012', 'trainval')]
    if cfg.mode == 'paper':
        train_splits = [('2007', 'trainval'), ('2012', 'trainval')]
    train = VOCDataset(cfg.data_root, train_splits, True, cfg.download,
                       limit=128 if cfg.mode == 'smoke' else None)
    val = None if cfg.mode == 'paper' else VOCDataset(
        cfg.data_root, [('2007', 'val')], False, cfg.download,
        limit=64 if cfg.mode == 'smoke' else None)
    loader = make_loader(train, cfg, True)
    model = YOLOv1()
    provenance = {}
    if cfg.backbone_checkpoint and not cfg.resume:
        provenance = load_backbone(model, cfg.backbone_checkpoint)
    has_pretrain = bool(cfg.backbone_checkpoint)
    # Load on CPU first to avoid duplicating full detector/optimizer GPU tensors.
    checkpoint = None
    start_epoch, best = 0, -1.0
    if cfg.resume:
        # Only load checkpoints made by you: full resume includes Python RNG state.
        checkpoint = torch.load(cfg.resume, map_location='cpu', weights_only=False)
        if checkpoint['architecture'] != ARCHITECTURE:
            raise ValueError('Checkpoint architecture mismatch')
        for key in ('mode', 'seed', 'epochs', 'effective_batch', 'micro_batch', 'amp', 'eval_score', 'nms_iou'):
            if checkpoint['config'][key] != asdict(cfg)[key]:
                raise ValueError(f'Resume config mismatch: {key}')
        model.load_state_dict(checkpoint['model'])
        start_epoch, best = checkpoint['epoch']+1, checkpoint['best']
        has_pretrain = checkpoint['has_pretrain']
        provenance = checkpoint['pretrain_provenance']
    if cfg.mode == 'paper' and not has_pretrain:
        raise ValueError('Cannot resume paper mode without ImageNet pretraining provenance')
    model.to(cfg.device)
    # foreach=False avoids extra parameter-sized temporary buffers on small GPUs.
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9,
                                weight_decay=0.0005, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=cfg.amp and cfg.device.startswith('cuda'), init_scale=128)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        restore_rng(checkpoint['rng'])
        del checkpoint
    manifest = {'config': asdict(cfg), 'architecture': ARCHITECTURE,
                'train_splits': train_splits, 'train_images': len(train),
                'validation_images': len(val) if val is not None else 0,
                'pretrain_provenance': provenance, 'has_imagenet_pretraining': has_pretrain,
                'torch': str(torch.__version__), 'gpu': torch.cuda.get_device_name() if torch.cuda.is_available() else 'CPU',
                'parameters': sum(p.numel() for p in model.parameters()),
                'metric': 'VOC2007 11-point mAP at IoU > 0.5'}
    write_json(manifest, output_dir/'manifest.json')
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    if not has_pretrain:
        warnings.warn('Random initialization: this run verifies implementation; it does not reproduce paper accuracy.')
    history_path = output_dir/'history.json'
    history = json.loads(history_path.read_text()) if cfg.resume and history_path.exists() else []
    history = [row for row in history if row['epoch'] <= start_epoch]
    for epoch in range(start_epoch, cfg.epochs):
        stats = {'epoch': epoch+1, **train_epoch(model, loader, optimizer, scaler, cfg, epoch, has_pretrain)}
        if val is not None and ((epoch+1) % cfg.eval_every == 0 or epoch+1 == cfg.epochs):
            metrics = evaluate(model, make_loader(val, cfg), cfg)
            stats['validation'] = metrics
            score = metrics['mAP50_VOC2007_11point']
            print(f'Validation mAP50 (VOC07 11-point): {score*100:.2f}', flush=True)
            if score > best:
                best = score
                atomic_save({'architecture': ARCHITECTURE, 'model': model.state_dict(),
                             'epoch': epoch, 'config': asdict(cfg)}, output_dir/'best_model.pt')
        history.append(stats)
        atomic_save({'architecture': ARCHITECTURE, 'model': model.state_dict(),
                     'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
                     'epoch': epoch, 'best': best, 'config': asdict(cfg), 'rng': rng_state(),
                     'has_pretrain': has_pretrain, 'pretrain_provenance': provenance}, output_dir/'last.pt')
        write_json(history, history_path)
    # Paper protocol: fixed final-epoch checkpoint, test once, no test-based selection.
    if cfg.mode == 'paper':
        test = VOCDataset(cfg.data_root, [('2007', 'test')], False, cfg.download)
        metrics = evaluate(model, make_loader(test, cfg), cfg)
        write_json(metrics, output_dir/'voc2007_test_metrics.json')
        print('Final VOC2007 TEST:', metrics, flush=True)
    atomic_save({'architecture': ARCHITECTURE, 'model': model.state_dict(),
                 'epoch': cfg.epochs-1, 'config': asdict(cfg)}, output_dir/'final_model.pt')
    return model, history


@torch.no_grad()
def predict_image(model, image_path, device='cuda', score_threshold=0.2, nms_iou=0.5):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    image = Image.open(image_path).convert('RGB')
    x = TF.to_tensor(image.resize((448, 448), Image.Resampling.BILINEAR))[None].to(device)
    model.eval()
    pred = decode(model(x), score_threshold, nms_iou)[0]
    w, h = image.size
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(image)
    for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
        x1, y1, x2, y2 = (box*torch.tensor([w, h, w, h])).tolist()
        color = plt.get_cmap('tab20')(int(label))
        ax.add_patch(patches.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2))
        ax.text(x1, y1, f'{CLASSES[int(label)]} {float(score):.2f}', color='white',
                bbox={'facecolor': color, 'alpha': 0.8, 'pad': 1})
    ax.axis('off')
    ax.set_title(f'YOLO v1 | detections: {len(pred["boxes"])}')
    plt.show()
    return pred


@torch.no_grad()
def benchmark(model, device='cuda', repeats=50, warmup=10):
    """Batch=1 FP32: network-only and network+decode+NMS, excluding file IO."""
    model.eval()
    x = torch.rand(1, 3, 448, 448, device=device)
    def sync():
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    for _ in range(warmup):
        decode(model(x), 0.2, 0.5)
    result = {}
    for include_nms in (False, True):
        sync()
        start = time.perf_counter()
        for _ in range(repeats):
            raw = model(x)
            if include_nms:
                decode(raw, 0.2, 0.5)
        sync()
        result['network_plus_nms_fps' if include_nms else 'network_only_fps'] = repeats/(time.perf_counter()-start)
    result['note'] = 'Batch 1, FP32, synthetic input, no image IO/resize; hardware-dependent, not original paper FPS.'
    return result


def pretrain_imagenet(root, output_dir, epochs=90, micro_batch=16, effective_batch=64,
                      workers=2, device='cuda', resume=''):
    """Optional 1000-class ImageNet first-20-conv pretraining.

    Supply licensed/local ImageNet train/<class> and val/<class> folders.
    90 epochs, step LR and transforms below are explicit implementation choices:
    the paper does not give a complete classification pretraining recipe.
    """
    if not torch.cuda.is_available() and device.startswith('cuda'):
        raise RuntimeError('ImageNet pretraining requires a GPU runtime.')
    if micro_batch < 1 or effective_batch < 1 or effective_batch % micro_batch:
        raise ValueError('effective_batch must be divisible by micro_batch')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir/'pretrain_last.pt').exists() and not resume:
        raise FileExistsError('Use resume or a new pretraining output directory')
    seed_all(42)
    train = datasets.ImageFolder(Path(root)/'train', transforms.Compose([
        transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()]))
    val = datasets.ImageFolder(Path(root)/'val', transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]))
    if len(train.classes) != 1000 or train.class_to_idx != val.class_to_idx:
        raise ValueError('ImageNet must have the same 1000 class folders in train and val.')
    loader = DataLoader(train, batch_size=micro_batch, shuffle=True, num_workers=workers,
                        pin_memory=True, worker_init_fn=seed_worker)
    vloader = DataLoader(val, batch_size=micro_batch, num_workers=workers, pin_memory=True)
    model = ImageNetClassifier().to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005, foreach=False)
    scaler = torch.amp.GradScaler('cuda', enabled=device.startswith('cuda'))
    start_epoch = 0
    if resume:
        state = torch.load(resume, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scaler.load_state_dict(state['scaler'])
        restore_rng(state['rng'])
        start_epoch = state['epoch']+1
        del state
    for epoch in range(start_epoch, epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        for group in optimizer.param_groups:
            group['lr'] = 0.01 * 0.1**(epoch//30)
        for step, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            with torch.autocast(device_type='cuda', enabled=device.startswith('cuda')):
                loss = nn.functional.cross_entropy(model(x), y)
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite ImageNet loss')
            scaler.scale(loss*len(x)/effective_batch).backward()
            pending += len(x)
            if pending == effective_batch or step+1 == len(loader):
                scaler.unscale_(optimizer)
                if pending != effective_batch:
                    for p in model.parameters():
                        if p.grad is not None:
                            p.grad.mul_(effective_batch/pending)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
            if step % 100 == 0:
                print(f'ImageNet epoch={epoch+1} step={step}/{len(loader)} loss={float(loss):.4f}', flush=True)
        model.eval()
        correct = count = 0
        with torch.no_grad():
            for x, y in vloader:
                top5 = model(x.to(device)).topk(5, dim=1).indices.cpu()
                correct += int((top5 == y[:, None]).any(dim=1).sum())
                count += len(y)
        print(f'ImageNet top-5: {100*correct/count:.2f}%', flush=True)
        atomic_save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                     'scaler': scaler.state_dict(), 'epoch': epoch, 'rng': rng_state()},
                    output_dir/'pretrain_last.pt')
        atomic_save({'architecture': ARCHITECTURE, 'task': 'imagenet1000',
                     'backbone': model.backbone.state_dict(),
                     'provenance': {'epochs': epoch+1, 'val_top5': correct/count,
                                    'recipe': '90-epoch configurable PyTorch recipe; not exact original pretrain schedule'}},
                    output_dir/'imagenet_backbone.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Config JSON for detector training')
    parser.add_argument('--pretrain-imagenet', help='ImageNet folder containing train/ and val/')
    parser.add_argument('--output-dir', default='/content/yolov1_imagenet')
    parser.add_argument('--resume', default='')
    args = parser.parse_args()
    if args.pretrain_imagenet:
        pretrain_imagenet(args.pretrain_imagenet, args.output_dir, resume=args.resume)
    elif args.config:
        run(Config(**json.loads(Path(args.config).read_text(encoding='utf-8'))))
    else:
        parser.error('Use --config config.json or --pretrain-imagenet /path/to/imagenet')

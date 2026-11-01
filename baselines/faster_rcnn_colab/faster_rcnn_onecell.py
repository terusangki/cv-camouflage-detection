# Colab: 이 파일 전체를 코드 셀 하나에 붙여넣으세요.
# 바꿀 설정은 여기입니다. 먼저 GPU 런타임을 선택하세요.
PROFILE = "quick"       # "quick": 동작 확인 / "paper07": 논문 규모 학습
QUICK_STEPS = 300        # quick의 각 단계: 300 + 100 iterations
USE_DRIVE = True         # 런타임 중단 후 재개하려면 True 권장
RESUME = False           # 같은 실험을 이어갈 때 True
RUN_NAME = PROFILE + "_seed42"  # 새 실험은 이름을 바꾸세요.
EVAL_LIMIT = 100 if PROFILE == "quick" else 0  # 0=전체. 100장 결과는 정식 mAP가 아님.
DATA_ROOT = "/content/voc"

"""Faster R-CNN (VGG16, RPN, RoI Pooling), educational PyTorch reimplementation.

Paper: https://arxiv.org/abs/1506.01497
Four-step alternating training; no pretrained detection model is used.
See README.md for deliberate differences from the Caffe reference implementation.
Coordinates throughout: zero-based, half-open XYXY, not Caffe inclusive XYXY.
"""
import argparse
import copy
import gc
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision.models import vgg16, VGG16_Weights
from torchvision.ops import box_iou, nms, roi_pool
from torchvision.transforms.functional import to_tensor
from PIL import Image

CLASSES = ('__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
           'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor')


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode_boxes(boxes, reference):
    wh = (reference[:, 2:] - reference[:, :2]).clamp(min=1e-6)
    ctr = (reference[:, 2:] + reference[:, :2]) / 2
    target_wh = (boxes[:, 2:] - boxes[:, :2]).clamp(min=1e-6)
    target_ctr = (boxes[:, 2:] + boxes[:, :2]) / 2
    return torch.cat(((target_ctr - ctr) / wh, torch.log(target_wh / wh)), 1)


def decode_boxes(delta, reference):
    wh = reference[:, 2:] - reference[:, :2]
    ctr = (reference[:, 2:] + reference[:, :2]) / 2
    pred_ctr = delta[:, :2] * wh + ctr
    pred_wh = delta[:, 2:].clamp(-10, np.log(1000 / 16)).exp() * wh
    return torch.cat((pred_ctr - pred_wh / 2, pred_ctr + pred_wh / 2), 1)


def clip_boxes(boxes, height, width):
    return torch.stack((boxes[:, 0].clamp(0, width), boxes[:, 1].clamp(0, height),
                        boxes[:, 2].clamp(0, width), boxes[:, 3].clamp(0, height)), 1)


def choose(indices, count):
    return indices[torch.randperm(len(indices), device=indices.device)[:count]]


def make_anchors(height, width, device):
    # Nine references at each stride-16 feature location. Ratio = height / width.
    shapes = []
    for ratio in (0.5, 1., 2.):
        for scale in (128., 256., 512.):
            w, h = scale / np.sqrt(ratio), scale * np.sqrt(ratio)
            shapes.append((-w / 2, -h / 2, w / 2, h / 2))
    base = torch.tensor(shapes, dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(torch.arange(height, device=device),
                            torch.arange(width, device=device), indexing='ij')
    centers = torch.stack((xx, yy, xx, yy), -1).float().reshape(-1, 4) * 16 + 8
    return (centers[:, None] + base[None]).reshape(-1, 4)


class RPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(512, 512, 3, padding=1)
        self.cls = nn.Conv2d(512, 18, 1)
        self.reg = nn.Conv2d(512, 36, 1)
        for layer in (self.conv, self.cls, self.reg):
            nn.init.normal_(layer.weight, std=0.01)
            nn.init.zeros_(layer.bias)

    def forward(self, feature):
        hidden = F.relu(self.conv(feature))
        # Channels are paired per anchor; flatten order matches make_anchors.
        logits = self.cls(hidden).permute(0, 2, 3, 1).reshape(-1, 2)
        delta = self.reg(hidden).permute(0, 2, 3, 1).reshape(-1, 4)
        anchors = make_anchors(*feature.shape[-2:], feature.device)
        return logits, delta, anchors


def rpn_loss(logits, delta, anchors, gt, image_hw, feature_locations):
    h, w = image_hw
    inside = ((anchors[:, :2] >= 0).all(1) &
              (anchors[:, 2] <= w) & (anchors[:, 3] <= h))
    ids = torch.where(inside)[0]
    a = anchors[ids]
    if len(a) == 0:
        raise ValueError('No inside anchors. Use images resized with short side >= 256.')
    labels = torch.full((len(a),), -1, dtype=torch.long, device=a.device)
    if len(gt):
        iou = box_iou(a, gt)
        best, match = iou.max(1)
        labels[best < 0.3] = 0
        labels[best >= 0.7] = 1
        # Keep ALL ties for each GT, as in the reference implementation.
        labels[(iou == iou.max(0).values[None]).any(1)] = 1
    else:
        labels[:] = 0
        match = torch.zeros(len(a), dtype=torch.long, device=a.device)
    pos = choose(torch.where(labels == 1)[0], 128)
    neg = choose(torch.where(labels == 0)[0], 256 - len(pos))
    sampled = torch.cat((pos, neg))
    cls_loss = F.cross_entropy(logits[ids[sampled]], labels[sampled])
    # Paper Eq.1: Ncls = sampled anchors, Nreg = feature locations, lambda = 10.
    reg_loss = delta.sum() * 0
    if len(pos):
        target = encode_boxes(gt[match[pos]], a[pos])
        reg_loss = 10 * F.smooth_l1_loss(delta[ids[pos]], target,
                                       reduction='sum', beta=1.) / feature_locations
    return {'rpn_cls': cls_loss, 'rpn_box': reg_loss}


@torch.no_grad()
def proposals(logits, delta, anchors, image_hw, scale, training):
    h, w = image_hw
    boxes = clip_boxes(decode_boxes(delta, anchors), h, w)
    scores = logits.softmax(1)[:, 1]
    wh = boxes[:, 2:] - boxes[:, :2]
    valid = ((wh >= 16 * scale).all(1) & torch.isfinite(boxes).all(1)
             & torch.isfinite(scores))
    boxes, scores = boxes[valid], scores[valid]
    order = scores.argsort(descending=True)[:12000 if training else 6000]
    boxes, scores = boxes[order], scores[order]
    keep = nms(boxes, scores, 0.7)[:2000 if training else 300]
    return boxes[keep]


def sample_rois(boxes, gt, labels, count=128):
    if len(gt):
        boxes = torch.cat((boxes, gt))  # Guarantees foreground training examples.
        overlap = box_iou(boxes, gt)
        best, match = overlap.max(1)
        pos = choose(torch.where(best >= 0.5)[0], count // 4)
        neg = choose(torch.where((best >= 0.1) & (best < 0.5))[0], count - len(pos))
    else:
        match = torch.zeros(len(boxes), dtype=torch.long, device=boxes.device)
        pos = match[:0]
        neg = choose(torch.arange(len(boxes), device=boxes.device), count)
    ids = torch.cat((pos, neg))
    if len(ids) == 0:
        return boxes[:0], labels[:0], boxes[:0]
    target_labels = torch.zeros(len(ids), dtype=torch.long, device=boxes.device)
    target_delta = boxes.new_zeros(len(ids), 4)
    if len(pos):
        target_labels[:len(pos)] = labels[match[pos]]
        target_delta[:len(pos)] = encode_boxes(gt[match[pos]], boxes[pos])
    return boxes[ids], target_labels, target_delta


class FasterRCNN(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Remove pool5: conv5_3 feature map has stride 16.
        self.backbone = nn.Sequential(*list(vgg.features.children())[:-1])
        # fc6 and fc7 (4096 each), including ReLU + dropout.
        self.head = nn.Sequential(*list(vgg.classifier.children())[:-1])
        self.rpn = RPN()
        self.cls = nn.Linear(4096, len(CLASSES))
        self.reg = nn.Linear(4096, len(CLASSES) * 4)
        nn.init.normal_(self.cls.weight, std=0.01)
        nn.init.normal_(self.reg.weight, std=0.001)
        nn.init.zeros_(self.cls.bias)
        nn.init.zeros_(self.reg.bias)
        self.register_buffer('box_std', torch.tensor([0.1, 0.1, 0.2, 0.2]))

    def configure_phase(self, phase):
        self.train()
        for p in self.parameters():
            p.requires_grad_(False)
        if phase in (1, 2):
            for index, layer in enumerate(self.backbone):
                # Freeze conv1_x and conv2_x; conv3_1 begins at index 10.
                for p in layer.parameters():
                    p.requires_grad_(index >= 10)
        modules = (self.rpn,) if phase in (1, 3) else (self.head, self.cls, self.reg)
        for module in modules:
            for p in module.parameters():
                p.requires_grad_(True)

    def detect_head(self, feature, boxes):
        pooled = roi_pool(feature, [boxes], output_size=(7, 7), spatial_scale=1/16)
        hidden = self.head(pooled.flatten(1))
        return self.cls(hidden), self.reg(hidden).reshape(-1, len(CLASSES), 4)

    def training_losses(self, image, target, scale, phase, teacher=None, roi_count=128):
        feature = self.backbone(image[None])
        gt, labels = target['boxes'], target['labels']
        if phase in (1, 3):
            logits, delta, anchors = self.rpn(feature)
            return rpn_loss(logits, delta, anchors, gt, image.shape[-2:],
                            feature.shape[-2] * feature.shape[-1])
        with torch.no_grad():
            if teacher is None:  # Phase 4 uses frozen shared backbone and final RPN.
                outputs = self.rpn(feature)
            else:  # Phase 2 uses frozen phase-1 proposal network.
                outputs = teacher(image[None])
            boxes = proposals(*outputs, image.shape[-2:], scale, True)
            boxes, y, target_delta = sample_rois(boxes, gt, labels, roi_count)
        if not len(boxes):
            zero = self.cls.weight.sum() * 0
            return {'roi_cls': zero, 'roi_box': zero}
        logits, delta = self.detect_head(feature, boxes)
        pos = torch.where(y > 0)[0]
        loss_box = delta.sum() * 0
        if len(pos):
            loss_box = F.smooth_l1_loss(delta[pos, y[pos]],
                        target_delta[pos] / self.box_std, reduction='sum', beta=1.) / len(y)
        return {'roi_cls': F.cross_entropy(logits, y), 'roi_box': loss_box}

    @torch.no_grad()
    def predict(self, image, scale, threshold=0.05):
        self.eval()
        feature = self.backbone(image[None])
        boxes = proposals(*self.rpn(feature), image.shape[-2:], scale, False)
        empty = {'boxes': image.new_zeros((0, 4)), 'scores': image.new_zeros(0),
                 'labels': torch.zeros(0, dtype=torch.long, device=image.device)}
        if not len(boxes):
            return empty
        logits, delta = self.detect_head(feature, boxes)
        scores = logits.softmax(1)
        all_boxes, all_scores, all_labels = [], [], []
        for c in range(1, len(CLASSES)):
            mask = scores[:, c] >= threshold
            b = decode_boxes(delta[mask, c] * self.box_std, boxes[mask])
            b = clip_boxes(b, *image.shape[-2:])
            s = scores[mask, c]
            valid = ((b[:, 2:] - b[:, :2]) > 0).all(1) & torch.isfinite(b).all(1)
            b, s = b[valid], s[valid]
            keep = nms(b, s, 0.3)
            all_boxes.append(b[keep])
            all_scores.append(s[keep])
            all_labels.append(torch.full((len(keep),), c, device=image.device, dtype=torch.long))
        b, s, y = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
        keep = s.argsort(descending=True)[:100]
        return {'boxes': b[keep], 'scores': s[keep], 'labels': y[keep]}


class ProposalTeacher(nn.Module):
    def __init__(self, backbone, rpn):
        super().__init__()
        self.backbone = backbone
        self.rpn = rpn
        self.eval()
        self.requires_grad_(False)

    def forward(self, images):
        return self.rpn(self.backbone(images))


def preprocess(pil, target=None, augment=False, short_side=600):
    image = to_tensor(pil.convert('RGB'))
    h, w = image.shape[-2:]
    nominal_scale = min(short_side / min(h, w), 1000 / max(h, w))
    nh, nw = round(h * nominal_scale), round(w * nominal_scale)
    image = F.interpolate(image[None], size=(nh, nw), mode='bilinear',
                          align_corners=False)[0]
    factors = image.new_tensor([nw/w, nh/h, nw/w, nh/h])
    if target is not None:
        target = {k: v.clone() if torch.is_tensor(v) else v for k, v in target.items()}
        target['boxes'] *= factors
    if augment and random.random() < 0.5:
        image = image.flip(-1)
        if target is not None:
            b = target['boxes']
            target['boxes'] = torch.stack((nw-b[:, 2], b[:, 1], nw-b[:, 0], b[:, 3]), 1)
    # RGB/ImageNet normalization matches torchvision's classification weights.
    mean = image.new_tensor([0.485, 0.456, 0.406])[:, None, None]
    std = image.new_tensor([0.229, 0.224, 0.225])[:, None, None]
    return (image - mean) / std, target, nominal_scale, factors


class VOC:
    def __init__(self, root, split, download=True):
        self.data = torchvision.datasets.VOCDetection(root, year='2007',
                    image_set=split, download=download)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, annotation = self.data[index]
        objects = annotation['annotation'].get('object', [])
        if isinstance(objects, dict):
            objects = [objects]
        boxes, labels, difficult = [], [], []
        for obj in objects:
            b = obj['bndbox']
            # VOC 1-based inclusive -> zero-based half-open.
            box = [float(b['xmin'])-1, float(b['ymin'])-1,
                   float(b['xmax']), float(b['ymax'])]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            boxes.append(box)
            labels.append(CLASSES.index(obj['name']))
            difficult.append(int(obj.get('difficult', 0)))
        return image, {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                       'labels': torch.tensor(labels, dtype=torch.long),
                       'difficult': torch.tensor(difficult, dtype=torch.bool)}


def checkpoint_load(path):
    # Only load checkpoints produced by this script / trusted by the user.
    return torch.load(path, map_location='cpu', weights_only=False)


def atomic_save(value, path):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def optimizer_for(model, lr):
    # Original convention: biases use double learning rate and no weight decay.
    regular, biases = [], []
    for name, p in model.named_parameters():
        if p.requires_grad:
            (biases if name.endswith('bias') else regular).append(p)
    return torch.optim.SGD([{'params': regular, 'lr': lr, 'multiplier': 1., 'weight_decay': 5e-4},
                            {'params': biases, 'lr': 2*lr, 'multiplier': 2., 'weight_decay': 0.}],
                           momentum=0.9)


def train(args):
    device = torch.device(args.device)
    if device.type == 'cpu' and not args.allow_cpu:
        raise RuntimeError('Select a GPU runtime in Colab. CPU training requires --allow-cpu.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    # Tutorial deliberately keeps VOC val separate; paper07 uses trainval/test.
    split = 'trainval' if args.profile == 'paper07' else 'train'
    data = VOC(args.data, split, not args.no_download)
    phases = [(60000, 20000), (30000, 10000), (60000, 20000), (30000, 10000)]
    if args.profile == 'quick':
        phases = [(args.quick_steps, max(1, args.quick_steps // 3))] * 4
    run_config = dict(vars(args), train_split=split, phases=phases,
                      torch=torch.__version__, torchvision=torchvision.__version__,
                      classes=CLASSES)
    latest = out / 'latest.pt'
    if latest.exists() and not args.resume:
        raise FileExistsError(f'{latest} exists. Use --resume or a different --out directory.')
    resume = checkpoint_load(latest) if args.resume and latest.exists() else None
    if args.resume and resume is None:
        raise FileNotFoundError(f'No checkpoint at {latest}')
    if resume:
        old = resume['config']
        for key in ('profile', 'quick_steps', 'short_side', 'seed'):
            if old[key] != run_config[key]:
                raise ValueError(f'Resume configuration differs: {key}')
    (out / 'config.json').write_text(json.dumps(run_config, indent=2, ensure_ascii=False), encoding='utf-8')
    start_phase = resume['phase'] if resume else 1
    model = None
    for phase in range(start_phase, 5):
        teacher = None
        if resume is not None:
            model = FasterRCNN(pretrained=False).to(device)
            model.load_state_dict(resume['model'])
            if phase == 2:
                teacher = ProposalTeacher(copy.deepcopy(model.backbone), copy.deepcopy(model.rpn))
                teacher.load_state_dict(resume['teacher'])
        elif phase == 1:
            model = FasterRCNN(pretrained=True).to(device)
        elif phase == 2:
            # Move stage-1 modules to a frozen teacher, release its unused FC head.
            teacher = ProposalTeacher(model.backbone, model.rpn)
            del model
            gc.collect()
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            model = FasterRCNN(pretrained=True).to(device)
        elif phase == 3:
            # Keep detector backbone from step 2; restore learned RPN from step 1.
            model.rpn.load_state_dict(checkpoint_load(out / 'stage1_rpn.pt'))
        # Phase 4 keeps step-3 backbone + RPN and step-2 detector head.
        model.configure_phase(phase)
        optimizer = optimizer_for(model, 0.001)
        start = 0
        if resume is not None:
            optimizer.load_state_dict(resume['optimizer'])
            start = resume['step']
            random.setstate(resume['rng_python'])
            np.random.set_state(resume['rng_numpy'])
            torch.set_rng_state(resume['rng_torch'])
            if device.type == 'cuda' and resume['rng_cuda'] is not None:
                torch.cuda.set_rng_state_all(resume['rng_cuda'])
            del resume
            resume = None
        high, low = phases[phase-1]
        total = high + low
        print(f'Phase {phase}/4: steps {start}/{total}; split={split}', flush=True)
        log_path = out / 'losses.jsonl'
        for step in range(start, total):
            lr = 0.001 if step < high else 0.0001
            for group in optimizer.param_groups:
                group['lr'] = lr * group['multiplier']
            optimizer.zero_grad(set_to_none=True)
            # Detector: two images x 64 RoIs; RPN: one image x 256 anchors.
            image_count = 1 if phase in (1, 3) else 2
            recorded = {}
            for _ in range(image_count):
                pil, target = data[random.randrange(len(data))]
                keep = ~target['difficult']
                target = {k: v[keep] for k, v in target.items()}
                image, target, scale, _ = preprocess(pil, target, True, args.short_side)
                image = image.to(device)
                target = {k: v.to(device) for k, v in target.items()}
                losses = model.training_losses(image, target, scale, phase, teacher, roi_count=64)
                loss = sum(losses.values()) / image_count
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Non-finite loss in phase {phase}, step {step}')
                loss.backward()
                for key, value in losses.items():
                    recorded[key] = recorded.get(key, 0.) + float(value.detach()) / image_count
            optimizer.step()
            if (step + 1) % 20 == 0 or step == start:
                row = dict(phase=phase, step=step+1, lr=lr, **recorded)
                print(row, flush=True)
                with log_path.open('a', encoding='utf-8') as handle:
                    handle.write(json.dumps(row) + '\n')
            if (step + 1) % args.save_every == 0 or step + 1 == total:
                state = {'phase': phase, 'step': step+1, 'model': model.state_dict(),
                         'optimizer': optimizer.state_dict(), 'config': run_config,
                         'teacher': teacher.state_dict() if teacher is not None else None,
                         'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                         'rng_torch': torch.get_rng_state(),
                         'rng_cuda': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None}
                atomic_save(state, latest)
                del state
        if phase == 1:
            atomic_save(model.rpn.state_dict(), out / 'stage1_rpn.pt')
        atomic_save({'model': model.state_dict(), 'config': run_config, 'phase': phase},
                    out / f'stage{phase}.pt')
        del optimizer, teacher
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    atomic_save({'model': model.state_dict(), 'config': run_config, 'phase': 4}, out / 'final.pt')
    print(f'Finished: {out / "final.pt"}', flush=True)


def voc07_ap(predictions, targets):
    """VOC2007 11-point AP, IoU > .5; difficult GT ignored, duplicates are FP."""
    result = {}
    for c in range(1, len(CLASSES)):
        truth, npos = {}, 0
        for image_id, target in enumerate(targets):
            mask = target['labels'] == c
            b, difficult = target['boxes'][mask], target['difficult'][mask]
            truth[image_id] = (b, difficult, torch.zeros(len(b), dtype=torch.bool))
            npos += int((~difficult).sum())
        detections = []
        for image_id, pred in enumerate(predictions):
            for j in torch.where(pred['labels'] == c)[0].tolist():
                detections.append((float(pred['scores'][j]), image_id, pred['boxes'][j]))
        detections.sort(key=lambda item: item[0], reverse=True)
        tp, fp = [], []
        for _, image_id, box in detections:
            boxes, difficult, used = truth[image_id]
            if not len(boxes):
                tp.append(0); fp.append(1)
                continue
            overlaps = box_iou(box[None], boxes)[0]
            j = int(overlaps.argmax())
            if float(overlaps[j]) > 0.5:
                if difficult[j]:
                    continue
                tp.append(int(not used[j])); fp.append(int(used[j]))
                used[j] = True
            else:
                tp.append(0); fp.append(1)
        if npos == 0:
            result[CLASSES[c]] = None
            continue
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        recall = tp / npos
        precision = tp / np.maximum(tp + fp, 1)
        result[CLASSES[c]] = float(np.mean([
            precision[recall >= threshold].max() if (recall >= threshold).any() else 0.
            for threshold in np.linspace(0, 1, 11)]))
    valid = [v for v in result.values() if v is not None]
    return {'mAP': float(np.mean(valid)) if valid else None, 'AP': result,
            'metric': 'VOC2007 11-point AP, IoU > 0.5', 'classes_with_gt': len(valid)}


def load_detector(path, device):
    checkpoint = checkpoint_load(path)
    if checkpoint.get('phase') != 4:
        raise ValueError('For final evaluation/prediction, use stage4.pt or final.pt.')
    model = FasterRCNN(pretrained=False)
    model.load_state_dict(checkpoint['model'])
    model.to(device).eval()
    return model, checkpoint['config']


@torch.no_grad()
def evaluate(args):
    device = torch.device(args.device)
    model, config = load_detector(args.checkpoint, device)
    if config['train_split'] == 'trainval' and args.split != 'test':
        raise ValueError('trainval-trained model must be evaluated on test, not val (leakage).')
    data = VOC(args.data, args.split, not args.no_download)
    count = min(len(data), args.limit) if args.limit else len(data)
    predictions, targets = [], []
    for i in range(count):
        pil, target = data[i]
        image, _, scale, factors = preprocess(pil, short_side=config['short_side'])
        pred = model.predict(image.to(device), scale, threshold=0.001)
        pred = {k: v.cpu() for k, v in pred.items()}
        pred['boxes'] /= factors
        predictions.append(pred)
        targets.append(target)
        if (i+1) % 100 == 0:
            print(f'Evaluated {i+1}/{count}', flush=True)
    result = voc07_ap(predictions, targets)
    result['mAP_percent'] = result['mAP'] * 100 if result['mAP'] is not None else None
    result.update(images=count, split=args.split, subset=bool(args.limit),
                  checkpoint=str(args.checkpoint), profile=config['profile'])
    output = Path(args.checkpoint).parent / f'metrics_{args.split}_{count}.json'
    output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


@torch.no_grad()
def predict_image(args):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    device = torch.device(args.device)
    model, config = load_detector(args.checkpoint, device)
    pil = Image.open(args.image).convert('RGB')
    image, _, scale, factors = preprocess(pil, short_side=config['short_side'])
    pred = model.predict(image.to(device), scale, args.threshold)
    boxes = pred['boxes'].cpu() / factors
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(pil)
    records = []
    for box, score, label in zip(boxes.tolist(), pred['scores'].tolist(), pred['labels'].tolist()):
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor='lime', linewidth=2))
        ax.text(x1, y1, f'{CLASSES[label]} {score:.2f}', color='black', backgroundcolor='lime')
        records.append(dict(box=box, score=score, label=CLASSES[label]))
    ax.axis('off')
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches='tight', dpi=150)
    plt.close(fig)
    output.with_suffix('.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'Saved {output}; detections={len(records)}')


def self_test(args):
    seed_all(7)
    device = torch.device(args.device)
    reference = torch.tensor([[0., 0., 100., 100.], [10., 30., 80., 90.]], device=device)
    target = reference + 5
    assert torch.allclose(decode_boxes(encode_boxes(target, reference), reference), target, atol=1e-5)
    anchors = make_anchors(32, 40, device)
    assert anchors.shape == (32*40*9, 4)
    rpn = RPN().to(device)
    feature = torch.randn(1, 512, 32, 40, device=device, requires_grad=True)
    logits, delta, anchors = rpn(feature)
    gt = torch.tensor([[100., 100., 240., 260.]], device=device)
    losses = rpn_loss(logits, delta, anchors, gt, (512, 640), 32*40)
    sum(losses.values()).backward()
    assert feature.grad is not None and torch.isfinite(feature.grad).all()
    boxes = proposals(logits, delta, anchors, (512, 640), 1., False)
    assert 0 < len(boxes) <= 300
    assert not boxes.requires_grad
    empty_gt_losses = rpn_loss(logits.detach(), delta.detach(), anchors, gt[:0], (512, 640), 32*40)
    assert all(torch.isfinite(value) for value in empty_gt_losses.values())
    assert empty_gt_losses['rpn_box'] == 0
    rois, labels, offsets = sample_rois(boxes, gt, torch.tensor([1], device=device), 64)
    assert len(rois) <= 64 and 0 < int((labels > 0).sum()) <= 16
    assert torch.isfinite(offsets).all()
    pooled = roi_pool(feature.detach(), [boxes[:2]], (7, 7), 1/16)
    assert pooled.shape == (min(2, len(boxes)), 512, 7, 7)
    target_cpu = {'boxes': torch.tensor([[0., 0., 10., 10.], [20., 20., 30., 30.]]),
                  'labels': torch.tensor([1, 1]), 'difficult': torch.tensor([False, True])}
    pred = {'boxes': target_cpu['boxes'].clone(), 'labels': torch.tensor([1, 1]),
            'scores': torch.tensor([0.8, 0.9])}
    assert abs(voc07_ap([pred], [target_cpu])['mAP'] - 1) < 1e-8
    pred['boxes'] = torch.tensor([[50., 50., 60., 60.], [70., 70., 80., 80.]])
    assert voc07_ap([pred], [target_cpu])['mAP'] == 0
    # A duplicate before the second true positive must lower AP.
    two_gt = dict(target_cpu, difficult=torch.tensor([False, False]))
    duplicate = {'boxes': torch.stack((two_gt['boxes'][0], two_gt['boxes'][0], two_gt['boxes'][1])),
                 'labels': torch.tensor([1, 1, 1]), 'scores': torch.tensor([0.9, 0.8, 0.7])}
    assert 0 < voc07_ap([duplicate], [two_gt])['mAP'] < 1
    print('PASS: box codec, anchors, RPN gradients, NMS proposals, RoI Pooling, VOC AP/difficult.')
    if args.full:
        del rpn, feature, logits, delta, anchors, pooled, losses
        gc.collect()
        model = FasterRCNN(pretrained=False).to(device)
        image = torch.randn(3, 400, 500, device=device)
        target = {'boxes': gt, 'labels': torch.tensor([1], device=device)}
        teacher = ProposalTeacher(copy.deepcopy(model.backbone), copy.deepcopy(model.rpn))
        for phase in range(1, 5):
            model.configure_phase(phase)
            model.zero_grad(set_to_none=True)
            losses = model.training_losses(image, target, 1., phase,
                                            teacher if phase == 2 else None, roi_count=8)
            sum(losses.values()).backward()
            assert all(torch.isfinite(v) for v in losses.values())
            assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
            assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
            assert all(p.grad is None for p in teacher.parameters())
            print(f'PASS: full VGG16 phase {phase} forward/backward + freeze checks')
        model.zero_grad(set_to_none=True)
        prediction = model.predict(image, 1.)
        assert len(prediction['boxes']) <= 100
        print('PASS: full model inference')


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    p = sub.add_parser('train')
    p.add_argument('--data', default='/content/voc')
    p.add_argument('--out', default='/content/faster_rcnn_run')
    p.add_argument('--profile', choices=['quick', 'paper07'], default='quick')
    p.add_argument('--quick-steps', type=int, default=300)
    p.add_argument('--short-side', type=int, default=600)
    p.add_argument('--save-every', type=int, default=200)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-cpu', action='store_true')
    p.add_argument('--no-download', action='store_true')
    p.add_argument('--device', default=device)
    p = sub.add_parser('eval')
    p.add_argument('--data', default='/content/voc')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--split', choices=['val', 'test'], default='val')
    p.add_argument('--limit', type=int, default=0)
    p.add_argument('--device', default=device)
    p.add_argument('--no-download', action='store_true')
    p = sub.add_parser('predict')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--image', required=True)
    p.add_argument('--output', default='/content/prediction.png')
    p.add_argument('--threshold', type=float, default=0.5)
    p.add_argument('--device', default=device)
    p = sub.add_parser('self-test')
    p.add_argument('--full', action='store_true')
    p.add_argument('--device', default=device)
    args = parser.parse_args()
    if args.command == 'train' and (args.quick_steps < 1 or args.save_every < 1 or args.short_side < 256):
        parser.error('quick-steps/save-every must be positive; short-side must be >= 256')
    {'train': train, 'eval': evaluate, 'predict': predict_image, 'self-test': self_test}[args.command](args)



# GPU 및 torchvision C++ 연산 확인
import sys, platform
print("Python:", platform.python_version(), "torch:", torch.__version__, "torchvision:", torchvision.__version__)
if not torch.cuda.is_available():
    raise RuntimeError("코랩: 런타임 > 런타임 유형 변경 > GPU를 선택하세요.")
print("GPU:", torch.cuda.get_device_name(0))
if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RUN_DIR = '/content/drive/MyDrive/faster_rcnn/' + RUN_NAME
else:
    RUN_DIR = '/content/faster_rcnn/' + RUN_NAME

def run_command(*arguments):
    old_argv = sys.argv
    sys.argv = ['faster_rcnn.py', *map(str, arguments)]
    try:
        main()
    finally:
        sys.argv = old_argv
        gc.collect()
        torch.cuda.empty_cache()

run_command('self-test')
arguments = ['train', '--data', DATA_ROOT, '--out', RUN_DIR,
             '--profile', PROFILE, '--quick-steps', QUICK_STEPS]
if RESUME:
    arguments.append('--resume')
run_command(*arguments)

CHECKPOINT = RUN_DIR + '/final.pt'
EVAL_SPLIT = 'test' if PROFILE == 'paper07' else 'val'
run_command('eval', '--data', DATA_ROOT, '--checkpoint', CHECKPOINT,
            '--split', EVAL_SPLIT, '--limit', EVAL_LIMIT)

# 같은 평가 split의 첫 사진을 시각화합니다. 짧은 학습에서는 검출이 없을 수 있습니다.
example_data = torchvision.datasets.VOCDetection(DATA_ROOT, year='2007',
                                                image_set=EVAL_SPLIT, download=False)
example_data[0][0].save('/content/example.jpg')
run_command('predict', '--checkpoint', CHECKPOINT, '--image', '/content/example.jpg',
            '--output', RUN_DIR + '/prediction.png', '--threshold', '0.5')
from IPython.display import display, Image as DisplayImage
display(DisplayImage(filename=RUN_DIR + '/prediction.png'))
print('결과 폴더:', RUN_DIR)

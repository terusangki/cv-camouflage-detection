"""Local regression tests: python test_yolov1.py [--full-model]. No downloads."""
import argparse
import gc
import math
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image
import torch
from torch import nn

import yolov1 as y


class CoreTests(unittest.TestCase):
    def test_iou(self):
        a = torch.tensor([0., 0., 2., 2.])
        b = torch.tensor([[0., 0., 2., 2.], [1., 1., 3., 3.], [3., 3., 4., 4.]])
        torch.testing.assert_close(y.aligned_iou(a, b), torch.tensor([1., 1/7, 0.]))

    def test_target_collision_and_edge(self):
        boxes = torch.tensor([[.01, .01, .1, .1], [0., 0., .13, .13], [.95, .95, 1., 1.]])
        labels = torch.tensor([0, 1, 2])
        target, collisions = y.encode_targets(boxes, labels)
        self.assertEqual(collisions, 1)
        self.assertEqual(target[..., 4].sum().item(), 2)
        self.assertEqual(target[0, 0, 6].item(), 1)
        recovered = y.grid_boxes_to_xyxy(target[None, ..., :4].unsqueeze(-2))
        torch.testing.assert_close(recovered[0, 6, 6, 0], boxes[2])

    def test_loss_responsibility_and_gradient(self):
        target = torch.zeros(1, 7, 7, 25)
        target[0, 2, 3, :5] = torch.tensor([.4, .6, .2, .3, 1.])
        target[0, 2, 3, 5+11] = 1
        pred = torch.zeros(1, 7, 7, 30)
        pred[0, 2, 3, 11] = 1
        pred[0, 2, 3, 25:30] = torch.tensor([.4, .6, .2, .3, 1.])
        loss, terms = y.YoloLoss()(pred, target)
        self.assertLess(float(loss), 1e-10)
        # Nonresponsible box confidence=0.5 adds 0.5*(0.5^2)=0.125 only.
        pred[0, 2, 3, 24] = .5
        loss, terms = y.YoloLoss()(pred, target)
        self.assertAlmostEqual(float(loss), .125, places=6)
        pred[0, 2, 3, 22] = -.01
        pred[0, 2, 3, 27] = -.01  # Both IoUs=0 -> first predictor wins tie.
        pred.requires_grad_()
        loss, _ = y.YoloLoss()(pred, target)
        loss.backward()
        self.assertTrue(torch.isfinite(pred.grad).all())
        self.assertLess(float(pred.grad[0, 2, 3, 22]), 0)
        # Empty-image confidence loss trains both box predictors; no class/xy loss.
        pred = torch.zeros(1, 7, 7, 30, requires_grad=True)
        with torch.no_grad():
            pred[..., 24] = .2
            pred[..., 29] = .3
            pred[..., :20] = .7
        loss, terms = y.YoloLoss()(pred, torch.zeros(1, 7, 7, 25))
        self.assertAlmostEqual(float(loss), .5*49*(.04+.09), places=5)
        loss.backward()
        self.assertEqual(float(pred.grad[..., :20].abs().sum()), 0)

    def test_class_aware_nms(self):
        pred = torch.zeros(1, 7, 7, 30)
        pred[0, 3, 3, 0] = .8
        pred[0, 3, 3, 1] = .7
        pred[0, 3, 3, 20:25] = torch.tensor([.5, .5, .2, .2, .9])
        pred[0, 3, 3, 25:30] = torch.tensor([.5, .5, .2, .2, .8])
        out = y.decode(pred, .1)[0]
        self.assertEqual(len(out['boxes']), 2)
        self.assertEqual(set(out['labels'].tolist()), {0, 1})
        torch.testing.assert_close(out['scores'], torch.tensor([.72, .63]))

    def test_voc_ap_difficult_duplicates_and_collisions(self):
        # Two GT boxes can share a cell. Both must count in recall denominator.
        gt = {'a': {'boxes': torch.tensor([[0., 0., .1, .1], [.01, .01, .11, .11], [.6, .6, .8, .8]]),
                    'labels': torch.zeros(3, dtype=torch.long),
                    'difficult': torch.tensor([False, False, True])}}
        pred = {'a': {'boxes': gt['a']['boxes'][[2, 0, 0, 1]],
                      'scores': torch.tensor([.99, .9, .8, .7]),
                      'labels': torch.zeros(4, dtype=torch.long)}}
        result = y.voc2007_ap(pred, gt)
        # Difficult is ignored; precision at recall<=.5=1; >.5=2/3.
        self.assertAlmostEqual(result['AP_per_class']['aeroplane'], (6+5*2/3)/11)
        self.assertEqual(result['GT_per_class']['aeroplane'], 2)
        self.assertEqual(result['classes_with_gt'], 1)
        self.assertEqual(y.voc2007_ap({}, gt)['mAP50_VOC2007_11point'], 0)

    def test_schedule(self):
        cfg = y.Config(mode='paper', epochs=135)
        self.assertEqual(y.learning_rate(0, 0, cfg, True), .001)
        self.assertAlmostEqual(y.learning_rate(0, 1, cfg, True), .01)
        self.assertEqual(y.learning_rate(74, 1, cfg, True), .01)
        self.assertEqual(y.learning_rate(75, 0, cfg, True), .001)
        self.assertEqual(y.learning_rate(105, 0, cfg, True), .0001)

    def test_voc_xml_image_and_augmentation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'VOCdevkit'/'VOC2007'
            for folder in ('ImageSets/Main', 'Annotations', 'JPEGImages'):
                (root/folder).mkdir(parents=True)
            (root/'ImageSets/Main/val.txt').write_text('tiny\n')
            Image.new('RGB', (100, 80), 'green').save(root/'JPEGImages/tiny.jpg')
            (root/'Annotations/tiny.xml').write_text('''<annotation><filename>tiny.jpg</filename>
            <object><name>dog</name><difficult>0</difficult><bndbox><xmin>1</xmin><ymin>1</ymin>
            <xmax>50</xmax><ymax>40</ymax></bndbox></object>
            <object><name>cat</name><difficult>1</difficult><bndbox><xmin>51</xmin><ymin>41</ymin>
            <xmax>100</xmax><ymax>80</ymax></bndbox></object></annotation>''')
            ds = y.VOCDataset(temp, [('2007', 'val')], False, False)
            image, target, meta = ds[0]
            self.assertEqual(image.shape, (3, 448, 448))
            self.assertEqual(meta['id'], '2007/tiny')
            self.assertEqual(len(meta['boxes']), 2)
            self.assertEqual(float(target[..., 4].sum()), 1)
            torch.testing.assert_close(meta['boxes'][0], torch.tensor([0., 0., .5, .5]))
            ds.training = True
            for seed in range(5):
                y.seed_all(seed)
                image, target, meta = ds[0]
                self.assertTrue(torch.isfinite(image).all())
                self.assertTrue((meta['boxes'] >= 0).all() and (meta['boxes'] <= 1).all())
                self.assertFalse(meta['difficult'].any())

    def test_accumulation_partial_batch_matches_large_batch(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.zeros(7, 7, 30))
            def forward(self, x):
                return self.bias[None].expand(len(x), -1, -1, -1)
        samples = []
        for _ in range(3):
            target = torch.zeros(7, 7, 25)
            target[0, 0, :5] = torch.tensor([.5, .5, .2, .2, 1.])
            target[0, 0, 5] = 1
            samples.append((torch.zeros(3, 1, 1), target, {'collisions': 0}))
        a, b = Tiny(), Tiny()
        optimizer_a = torch.optim.SGD(a.parameters(), lr=.0001)
        optimizer_b = torch.optim.SGD(b.parameters(), lr=.0001)
        cfg = y.Config(epochs=10, micro_batch=1, effective_batch=4, device='cpu', workers=0, amp=False)
        loader = y.make_loader(samples, cfg)
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        y.train_epoch(a, loader, optimizer_a, scaler, cfg, 0, False)
        target = torch.stack([s[1] for s in samples])
        loss, _ = y.YoloLoss()(b(torch.zeros(3, 3, 1, 1)), target)
        loss.backward(); optimizer_b.step()
        torch.testing.assert_close(a.bias, b.bias)

    def test_checkpoint_resume_matches_uninterrupted_run(self):
        class Tiny(nn.Module):
            def __init__(self):
                super().__init__()
                self.bias = nn.Parameter(torch.randn(7, 7, 30)*.01)
            def forward(self, x):
                return self.bias[None].expand(len(x), -1, -1, -1)
        samples = []
        for i in range(3):
            boxes = torch.tensor([[.1, .1, .2, .2]])
            labels = torch.tensor([i])
            target, collisions = y.encode_targets(boxes, labels)
            samples.append((torch.zeros(3, 1, 1), target,
                            {'id': str(i), 'boxes': boxes, 'labels': labels,
                             'difficult': torch.tensor([False]), 'collisions': collisions}))
        original_train = y.train_epoch
        def interrupt_second(*args, **kwargs):
            if args[5] == 1:
                raise RuntimeError('simulated runtime interruption')
            return original_train(*args, **kwargs)
        with tempfile.TemporaryDirectory() as temp, patch.object(y, 'YOLOv1', Tiny), \
             patch.object(y, 'VOCDataset', return_value=samples):
            def config(folder):
                return y.Config(mode='smoke', output_dir=str(Path(temp)/folder), epochs=2,
                                device='cpu', workers=0, amp=False, micro_batch=1,
                                effective_batch=2, eval_every=1)
            a, _ = y.run(config('continuous'))
            cfg = config('resumed')
            with patch.object(y, 'train_epoch', side_effect=interrupt_second):
                with self.assertRaisesRegex(RuntimeError, 'simulated'):
                    y.run(cfg)
            cfg.resume = str(Path(cfg.output_dir)/'last.pt')
            b, history = y.run(cfg)
            torch.testing.assert_close(a.bias, b.bias, rtol=0, atol=0)
            self.assertEqual([h['epoch'] for h in history], [1, 2])
            self.assertTrue((Path(cfg.output_dir)/'final_model.pt').exists())


def full_model_test():
    torch.set_num_threads(4)
    # Actual 448 forward and backward, all layers trainable, full FC4096.
    model = y.YOLOv1()
    assert sum(isinstance(m, nn.Conv2d) for m in model.modules()) == 24
    assert sum(isinstance(m, nn.Linear) for m in model.modules()) == 2
    assert not any(isinstance(m, nn.BatchNorm2d) for m in model.modules())
    x = torch.rand(1, 3, 448, 448)
    target = torch.zeros(1, 7, 7, 25)
    target[0, 3, 3, :5] = torch.tensor([.5, .5, .3, .3, 1.])
    target[0, 3, 3, 16] = 1
    pred = model(x)
    assert pred.shape == (1, 7, 7, 30)
    loss, _ = y.YoloLoss()(pred, target)
    loss.backward()
    assert torch.isfinite(model.backbone[0][0].weight.grad).all()
    assert torch.isfinite(model.head[-1].weight.grad).all()
    assert all(p.grad is not None for p in model.parameters())
    print('FULL_MODEL_PASS', sum(p.numel() for p in model.parameters()), 'parameters', 'loss', float(loss.detach()))
    del model, x, pred, loss
    gc.collect()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--full-model', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests))
    if not result.wasSuccessful():
        raise SystemExit(1)
    if args.full_model:
        full_model_test()

"""Exercise actual train/checkpoint/resume control flow with a tiny synthetic model.

This test does not claim VOC accuracy or substitute for the VGG forward/backward test.
It checks interrupted runs against an uninterrupted run without multi-GB snapshots.
"""
import argparse
import contextlib
import io
import tempfile
from pathlib import Path
import torch
from torch import nn
from PIL import Image
import faster_rcnn as impl


class Tiny(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.backbone = nn.Linear(1, 1)
        self.rpn = nn.Linear(1, 1)
        self.head = nn.Linear(1, 1)
        self.cls = nn.Linear(1, 1)
        self.reg = nn.Linear(1, 1)

    def configure_phase(self, phase):
        self.requires_grad_(False)
        if phase in (1, 2):
            self.backbone.requires_grad_(True)
        for module in ((self.rpn,) if phase in (1, 3) else (self.head, self.cls, self.reg)):
            module.requires_grad_(True)

    def training_losses(self, image, target, scale, phase, teacher=None, roi_count=64):
        # Randomness tests that RNG restoration occurs after model initialization.
        factor = torch.rand(()) + 0.5
        return {'synthetic': sum(p.square().sum() for p in self.parameters() if p.requires_grad) * factor}


class SyntheticVOC:
    def __init__(self, *args, **kwargs):
        pass

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return Image.new('RGB', (320, 256)), {
            'boxes': torch.tensor([[10., 10., 100., 150.]]),
            'labels': torch.tensor([1]), 'difficult': torch.tensor([False])}


def run_test():
    impl.FasterRCNN, impl.VOC = Tiny, SyntheticVOC
    original_save = impl.atomic_save
    with tempfile.TemporaryDirectory(prefix='faster_rcnn_checkpoint_') as directory:
        root = Path(directory)
        args = argparse.Namespace(device='cpu', allow_cpu=True, out=str(root / 'baseline'),
            seed=42, profile='quick', quick_steps=1, data='synthetic', no_download=True,
            resume=False, short_side=256, save_every=1)
        with contextlib.redirect_stdout(io.StringIO()):
            impl.train(args)
        expected = impl.checkpoint_load(root / 'baseline' / 'final.pt')['model']
        for interrupted_phase in (1, 2, 3, 4):
            args.out = str(root / f'interrupted_{interrupted_phase}')
            args.resume = False

            def interrupted_save(state, path):
                original_save(state, path)
                if Path(path).name == 'latest.pt' and state['phase'] == interrupted_phase and state['step'] == 1:
                    raise InterruptedError('Synthetic interruption AFTER atomic save')

            impl.atomic_save = interrupted_save
            with contextlib.redirect_stdout(io.StringIO()):
                try:
                    impl.train(args)
                except InterruptedError:
                    pass
                else:
                    raise AssertionError('Did not interrupt')
                impl.atomic_save = original_save
                args.resume = True
                impl.train(args)
            actual = impl.checkpoint_load(Path(args.out) / 'final.pt')['model']
            assert actual.keys() == expected.keys()
            assert all(torch.equal(actual[key], expected[key]) for key in expected)
            print(f'PASS: phase {interrupted_phase} interruption/resume exactly matches uninterrupted weights')
    impl.atomic_save = original_save


if __name__ == '__main__':
    run_test()

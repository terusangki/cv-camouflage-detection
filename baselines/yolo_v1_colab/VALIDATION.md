# Validation — 2026-10-04

Environment: Windows, Python 3.12 bundled runtime, PyTorch 2.14.1+cpu,
torchvision 0.29.1+cpu. No pretrained weights or VOC downloads used in tests.

## Passed

- Python AST parsing of every source file and every notebook code cell.
- Notebook embedded module matches yolov1.py exactly.
- Nine regression tests (`python test_yolov1.py --full-model`):
  1. Partial effective-batch gradient accumulation matches large-batch gradient.
  2. Epoch-boundary checkpoint resume matches uninterrupted training exactly on a tiny test model.
  3. Class-aware NMS removes same-class duplicates but preserves distinct classes.
  4. Analytic IoU values.
  5. Responsible predictor, no-object masking, negative width gradient, perfect-target loss.
  6. Paper learning-rate boundaries.
  7. Grid collision rule and coordinate conversion at image edges.
  8. VOC2007 AP with difficult objects, duplicate detections and colliding GT centers.
  9. Actual synthetic VOC XML/image loading and joint augmentation.
- Full production model: 24 Conv, 2 Linear, no BatchNorm, 271,703,550 parameters.
- Full production model forward on [1,3,448,448] -> [1,7,7,30].
- Full production model backward: every parameter receives a gradient;
  first convolution and last linear gradients are finite. Synthetic loss: 6.607242584228516.

## Not executed

- Colab GPU/CUDA/AMP execution and GPU memory profiling.
- Remote VOC archive downloads and full VOC training/test evaluation.
- ImageNet pretraining, original 63.4 mAP/45 FPS reproduction.
- Real-image quality assessment using a trained detector.

The tests validate implementation mechanics, not trained detection quality.

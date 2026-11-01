"""Build a self-contained Colab notebook. No external repo or uploads required."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
cells = []


def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.strip()+'\n'})


def code(text):
    cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                  'outputs': [], 'source': text.strip()+'\n'})


md('''# YOLO v1 논문 직접 구현 — Colab

사진 → 24개 convolution → FC 4096 → FC 1470 → 7×7×30 → NMS.

**처음에는 `MODE="smoke"` 그대로 위에서 아래로 실행하세요.** 모델을 축소하지 않고
VOC2007의 학습 128장/검증 64장으로 데이터·학습·평가·저장을 확인합니다.
이 한 epoch는 성능 실험이 아니며, 무작위 초기화이므로 정상 검출이 안 나와도 이상하지 않습니다.

1. Colab에서 **파일 → 노트북 업로드**로 이 파일을 엽니다.
2. **런타임 → 런타임 유형 변경 → GPU**를 선택합니다.
3. 설정 셀에서 Drive 저장 여부를 정하고 나머지 셀을 순서대로 실행합니다.
4. `smoke`, `dev`, `paper`는 서로 다른 출력 폴더를 사용합니다.

입력 448×448, S=7/B=2/C=20, 24 Conv + 2 FC, LeakyReLU(0.1), Dropout(0.5),
선형 출력, IoU 책임 박스, λcoord=5/λnoobj=0.5를 직접 구현했습니다.
ResNet/YOLOv8 등으로 교체하지 않습니다. 약 2.7억 파라미터이므로 무거운 모델입니다.

**논문 성능과 구현은 구분해야 합니다.** 논문은 ImageNet 1000-class로 첫 20개 Conv를
사전학습했습니다. 그 과정을 생략한 짧은 학습에서 논문의 63.4 mAP가 나오지는 않습니다.
아래에는 ImageNet 사전학습 코드도 있지만 전체 ImageNet 데이터는 별도로 준비해야 합니다.
논문 수치 재현을 완료한 코드는 아니며, 동일한 핵심 구조/목적함수를 구현한 실험 출발점입니다.
''')
md('''## 1. 환경 확인

Colab 기본 PyTorch/torchvision을 사용합니다. CUDA 패키지를 무조건 재설치하지 않습니다.
호환되지 않는 이전 패키지가 설치된 런타임이면 새 런타임에서 시작하세요.
''')
code('''import sys, torch, torchvision
print("Python:", sys.version)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
assert torch.cuda.is_available(), "런타임 유형을 GPU로 바꿔 주세요."
print("GPU:", torch.cuda.get_device_name(0))
print("VRAM GiB:", round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
''')
md('''## 2. 전체 구현 코드

다음 셀 안에 데이터셋, 모델, loss, 학습, mAP, NMS, 추론, FPS, ImageNet 사전학습이
모두 들어 있습니다. 이 셀은 코드 파일만 만들며 아직 데이터 다운로드나 학습은 시작하지 않습니다.
''')
source = (ROOT/'yolov1.py').read_text(encoding='utf-8')
code('%%writefile /content/yolov1.py\n'+source)
md('''## 3. 실행 설정

| 모드 | 학습 | 검증/평가 | 목적 |
|---|---|---|---|
| smoke | VOC07 train 128장, 1 epoch | VOC07 val 64장 | 실행 점검; 성능 보고 금지 |
| dev | VOC07 train + VOC12 trainval, 기본 30 epochs | VOC07 val | 개발·설정 조정 |
| paper | VOC07 trainval + VOC12 trainval, 135 epochs | 마지막에 VOC07 test 1회 | 논문 데이터/검출 스케줄 |

`paper`는 ImageNet 사전학습 체크포인트를 요구합니다. 임의로 이 제한만 제거하면
논문 설정 실험이 아닙니다. `dev`도 사전학습 가중치를 쓰는 것이 좋습니다.

**저장 공간:** 전체 모델의 가중치 하나가 약 1.1GB이고 optimizer까지 포함한 `last.pt`는
약 2.2GB입니다. atomic save 임시 파일, final/best 가중치까지 고려해 출력 공간 약 8GB 이상,
VOC 전체 다운로드/압축 해제에는 별도 여유 공간이 필요합니다.
smoke도 다운로드는 VOC07 전체 trainval 묶음을 받고, 학습에만 128장을 씁니다.

Drive를 사용하면 매 epoch 종료 시 저장되어 런타임 종료 후에도 남습니다.
기본 `/content` 데이터는 런타임 종료 시 사라질 수 있어 다시 다운로드될 수 있습니다.
''')
code('''from pathlib import Path
import importlib, yolov1
importlib.reload(yolov1)
from yolov1 import Config, run, predict_image, benchmark

MODE = "smoke"                     # "smoke", "dev", "paper"
USE_DRIVE = True
RESUME = False                      # 같은 모드/설정의 last.pt가 있으면 True
BACKBONE_CHECKPOINT = ""           # 예: /content/drive/MyDrive/yolov1_imagenet/imagenet_backbone.pt
MICRO_BATCH = 2                     # CUDA OOM이면 1로 낮추기 (effective batch는 64 유지)

if USE_DRIVE:
    from google.colab import drive
    drive.mount('/content/drive')
    RUN_ROOT = Path('/content/drive/MyDrive/computer_vision_project')
else:
    RUN_ROOT = Path('/content')

RUN_DIR = RUN_ROOT / f'yolov1_{MODE}'
cfg = Config(
    mode=MODE,
    output_dir=str(RUN_DIR),
    data_root='/content/voc',
    backbone_checkpoint=BACKBONE_CHECKPOINT,
    resume=str(RUN_DIR / 'last.pt') if RESUME else '',
    micro_batch=MICRO_BATCH,
    effective_batch=64,
    workers=2,
    amp=True,
    epochs=0,                       # 0은 모드 기본값; paper는 반드시 135
)
print(cfg)
''')
md('''## 4. 학습과 평가

일반적인 `dev` 실행에서는 validation mAP가 좋아질 때 `best_model.pt`를 저장합니다.
`paper` 모드는 test 결과로 모델을 고르지 않으며 고정된 135번째 epoch 모델을 평가합니다.
중단 후에는 설정 셀의 `RESUME=True`로 같은 폴더를 지정하고 다시 실행하세요.
학습 도중 중단하면 **마지막 완료 epoch부터** 재개합니다.

오래 걸리는 작업입니다. 출력되는 loss, 현재 batch/epoch로 진행 상황을 확인하세요.
loss가 감소해도 mAP가 바로 상승하는 것은 아닙니다.
''')
code('''model, history = run(cfg)
''')
md('''## 5. 학습 곡선과 결과 파일

mAP 값은 코드에서는 0~1, 아래 그래프에서는 ×100으로 표시합니다.
smoke 검증 64장은 VOC07 test가 아니므로 논문 결과와 비교하면 안 됩니다.
''')
code('''import json
import matplotlib.pyplot as plt

history = json.loads((RUN_DIR/'history.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot([h['epoch'] for h in history], [h['train_loss'] for h in history], marker='o')
axes[0].set(xlabel='epoch', ylabel='loss', title='Training loss')
evaluated = [h for h in history if 'validation' in h]
if evaluated:
    axes[1].plot([h['epoch'] for h in evaluated],
                 [h['validation']['mAP50_VOC2007_11point']*100 for h in evaluated], marker='o')
axes[1].set(xlabel='epoch', ylabel='mAP50 (%)', title='Validation: VOC2007 11-point AP')
plt.tight_layout(); plt.show()
for p in sorted(RUN_DIR.iterdir()):
    print(p.name, f'{p.stat().st_size/2**20:.1f} MiB')
''')
md('''## 6. 사진 한 장 추론

학습된 현재 `model`을 사용합니다. `smoke`에서는 박스가 없거나 틀릴 수 있습니다.
학습을 완료한 모델로 자신의 사진을 시험하려면 `IMAGE_PATH`를 업로드한 파일 경로로 바꾸세요.
주석 처리한 업로드 두 줄을 실행하면 Colab에서 파일 선택창이 열립니다.
''')
code('''# from google.colab import files
# uploaded = files.upload()
# IMAGE_PATH = next(iter(uploaded))

split = 'test' if MODE == 'paper' else 'val'
sample_id = (Path(cfg.data_root)/'VOCdevkit/VOC2007/ImageSets/Main'/f'{split}.txt').read_text().split()[0]
IMAGE_PATH = str(Path(cfg.data_root)/'VOCdevkit/VOC2007/JPEGImages'/f'{sample_id}.jpg')
prediction = predict_image(model, IMAGE_PATH, device=cfg.device, score_threshold=0.2)
''')
md('''## 7. 속도 측정 (선택)

batch 1, FP32, warm-up 후 CUDA 동기화하여 측정합니다.
파일 읽기·이미지 resize는 제외하고 network만 / network+NMS를 나누어 출력합니다.
입력은 고정 크기의 합성 이미지이며 원문 Titan X의 45 FPS와 직접 비교할 수 없습니다.
두 모델을 비교할 때 GPU, dtype, batch, IO/NMS 포함 범위를 반드시 같게 하세요.
''')
code('''RUN_SPEED_TEST = False
if RUN_SPEED_TEST:
    speed = benchmark(model, device=cfg.device)
    print(speed)
    (RUN_DIR/'speed.json').write_text(json.dumps(speed, indent=2))
''')
md('''## 8. ImageNet 사전학습 (선택, 별도 장시간 실행)

논문은 첫 20개 Conv + global average pooling + 1000-class FC를 224×224 입력으로 학습합니다.
아래 코드는 그 모델을 제공합니다. ImageNet은 사용자가 이용 조건에 따라 준비해야 하므로
자동으로 받지 않습니다. `train/<class>/*.JPEG`, `val/<class>/*.JPEG` 구조가 필요합니다.

**분류 사전학습의 정확한 스케줄은 원문에 충분히 기술되어 있지 않습니다.** 여기서는
90 epochs, SGD, 30 epochs마다 LR×0.1의 명시적인 구현 선택을 사용합니다.
원문의 top-5 88% 달성을 보장하지 않으며 실제 top-5를 저장합니다.
이 단계는 새 GPU 런타임에서 1~2번 셀 실행 후 아래 셀을 실행하는 것이 좋습니다.
학습이 끝나면 생성된 `imagenet_backbone.pt`를 3번 셀에 넣고 `MODE="paper"`로 실행합니다.
Darknet 최신 .weights는 구조가 다를 수 있어 이 구현에 직접 넣을 수 없습니다.
''')
code('''RUN_IMAGENET_PRETRAIN = False
if RUN_IMAGENET_PRETRAIN:
    # 별도 런타임 권장. 현재 detector가 있으면 GPU 메모리를 비웁니다.
    import gc
    if 'model' in globals():
        del model
    gc.collect(); torch.cuda.empty_cache()
    from yolov1 import pretrain_imagenet
    pretrain_imagenet(
        root='/content/imagenet',
        output_dir='/content/drive/MyDrive/yolov1_imagenet',
        epochs=90,
        micro_batch=8,
        effective_batch=64,
        resume='',    # 재개: .../pretrain_last.pt
    )
''')
md('''## 구현 선택과 보고서에 적을 점

- 원문: 24 Conv + 2 FC, 448 input, grid 7×7, B=2, C=20, linear output,
  SSE loss, IoU 책임 할당, confidence target=IoU, coord=5/noobj=0.5.
- BatchNorm/anchor/CE classification loss를 추가하지 않았습니다.
- 좌표는 x/y=cell 내부 상대좌표, w/h=전체 이미지 상대크기입니다.
- 음수 w/h를 낼 수 있는 선형 출력의 sqrt를 유한한 gradient로 처리하기 위해
  `x/sqrt(abs(x)+eps)` 확장을 사용합니다. 이는 원문이 명시하지 않은 수치 처리입니다.
- 같은 칸에 GT가 여러 개면 학습에서는 가장 큰 물체 하나를 남깁니다. 이 선택은 원문에
  명시되어 있지 않습니다. 평가에서는 빠진 GT까지 **전부** 사용합니다.
- VOC difficult는 학습에서 제외하고, 평가에서 그 물체와 겹친 검출은 무시합니다.
- 원본 VOC 1-based inclusive 좌표를 continuous 0-based 좌표로 바꿔 IoU를 계산합니다.
- no-object confidence loss는 물체 칸의 비담당 predictor에도 적용합니다.
- 원문에는 warm-up 길이/NMS threshold가 명확하지 않아 각각 1 epoch / 0.5를 선택했습니다.
- 증강은 ±20% crop/pad에 따른 scale/translation, 좌우 반전, HSV S/V 1.5 범위입니다.
  원본 Darknet의 픽셀 단위 보간·증강 구현과 완전히 같지는 않습니다.
- AMP와 gradient accumulation은 Colab 메모리를 위한 선택입니다. AMP off 가능.
- ImageNet 분류 학습의 세부 recipe와 초기화는 본 구현의 선택입니다.
- mAP: **VOC2007 11-point AP @ IoU>0.5**, COCO mAP@[.5:.95]와 다릅니다.
- 이 노트북은 VOC2007 평가 경로를 구현합니다. VOC2012 test 서버 평가,
  Fast YOLO, Fast R-CNN ensemble, artwork 실험까지 재현하지 않습니다.

### 출처

- [YOLO v1 원문](https://arxiv.org/abs/1506.02640), Figure 3 및 §2.1–2.4.
- [torchvision VOCDetection](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)
- [Darknet 저장소의 yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg)는
  현재 B=3/BatchNorm/local layer를 포함하여 첨부 논문의 Figure 3과 다릅니다.
  따라서 그 파일을 그대로 쓰지 않고 논문 구조를 직접 구현했습니다.
''')

for index, cell in enumerate(cells):
    cell['id'] = f'yolov1-{index:02d}'
    if cell['cell_type'] == 'code':
        src = cell['source']
        ast.parse(src.split('\n', 1)[1] if src.startswith('%%writefile') else src)

notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
            'metadata': {'accelerator': 'GPU',
                         'colab': {'name': 'YOLO_v1_paper_colab.ipynb', 'provenance': []},
                         'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                         'language_info': {'name': 'python', 'version': '3.12'}}}
(ROOT/'YOLO_v1_paper_colab.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')

# All-in-one cell version: no local module upload/import required.
implementation = source.split("if __name__ == '__main__':")[0]
driver = '''
# ===== Colab: 아래 설정을 바꾸고 전체 코드를 한 셀에 붙여 실행 =====
if __name__ == '__main__':
    MODE = 'smoke'  # dev / paper
    USE_DRIVE = True
    RESUME = False
    BACKBONE_CHECKPOINT = ''
    if USE_DRIVE:
        from google.colab import drive
        drive.mount('/content/drive')
        run_root = Path('/content/drive/MyDrive/computer_vision_project')
    else:
        run_root = Path('/content')
    run_dir = run_root / f'yolov1_{MODE}'
    cfg = Config(mode=MODE, output_dir=str(run_dir),
                 backbone_checkpoint=BACKBONE_CHECKPOINT,
                 resume=str(run_dir/'last.pt') if RESUME else '')
    model, history = run(cfg)
    print('완료. 결과 폴더:', run_dir)
    # 내 사진 확인: predict_image(model, '/content/my_photo.jpg')
'''
ast.parse(implementation+driver)
(ROOT/'YOLO_v1_onecell.py').write_text(implementation+driver, encoding='utf-8')
print(f'Built notebook ({len(cells)} cells) and one-cell script.')

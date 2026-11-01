"""Generate a self-contained Colab notebook and single-cell entrypoint."""
import json
from pathlib import Path

root = Path(__file__).resolve().parent
source = (root / 'faster_rcnn.py').read_text(encoding='utf-8')
source = source.rsplit("if __name__ == '__main__':", 1)[0]
source = source.replace('from __future__ import annotations\n', '')
settings = '''# 바꿀 설정은 여기입니다. 먼저 GPU 런타임을 선택하세요.
PROFILE = "quick"       # "quick": 동작 확인 / "paper07": 논문 규모 학습
QUICK_STEPS = 300        # quick의 각 단계: 300 + 100 iterations
USE_DRIVE = True         # 런타임 중단 후 재개하려면 True 권장
RESUME = False           # 같은 실험을 이어갈 때 True
RUN_NAME = PROFILE + "_seed42"  # 새 실험은 이름을 바꾸세요.
EVAL_LIMIT = 100 if PROFILE == "quick" else 0  # 0=전체. 100장 결과는 정식 mAP가 아님.
DATA_ROOT = "/content/voc"
'''
runner = '''# GPU 및 torchvision C++ 연산 확인
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
Path('/content/faster_rcnn.py').write_text(SOURCE_CODE, encoding='utf-8')

def run_command(*arguments):
    # 실패 시 셀을 즉시 중단하여 실패한 학습 뒤 평가가 진행되지 않게 합니다.
    import subprocess
    subprocess.run([sys.executable, '/content/faster_rcnn.py', *map(str, arguments)], check=True)

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
'''
# Store the complete CLI source as an embedded string, then define it for inspection.
# Training/evaluation run in subprocesses so GPU memory is released between commands.
payload = 'SOURCE_CODE = ' + repr((root / 'faster_rcnn.py').read_text(encoding='utf-8')) + '\n'
imports = 'from pathlib import Path\nimport torch, torchvision\n'
onecell_runner = runner.replace("Path('/content/faster_rcnn.py').write_text(SOURCE_CODE, encoding='utf-8')\n", '')
start = onecell_runner.index('def run_command(')
end = onecell_runner.index("run_command('self-test')", start)
onecell_runner = onecell_runner[:start] + '''def run_command(*arguments):
    old_argv = sys.argv
    sys.argv = ['faster_rcnn.py', *map(str, arguments)]
    try:
        main()
    finally:
        sys.argv = old_argv
        gc.collect()
        torch.cuda.empty_cache()

''' + onecell_runner[end:]
onecell = '# Colab: 이 파일 전체를 코드 셀 하나에 붙여넣으세요.\n' + settings + '\n' + source + '\n' + onecell_runner
(root / 'faster_rcnn_onecell.py').write_text(onecell, encoding='utf-8')
cells = []
def md(text):
    cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(True)})
def code(text):
    cells.append({'cell_type': 'code', 'execution_count': None, 'outputs': [],
                  'metadata': {}, 'source': text.splitlines(True)})
md('''# Faster R-CNN 논문 구현 (VGG16 + RPN + RoI Pooling)
**2015 Faster R-CNN을 구현합니다. 2014 R-CNN과는 다릅니다.**

GPU 런타임을 선택하고 위에서부터 실행하세요. 외부 `.py` 파일 업로드는 필요 없습니다.
기본 quick은 전체 파이프라인 실행 확인용이며 논문 정확도 재현 실험이 아닙니다.
VOC 다운로드, ImageNet VGG16 초기화, 4단계 학습, 체크포인트, VOC AP, 예측 그림을 포함합니다.
''')
code(settings)
md('''## 원 논문과의 관계
4단계 교대 학습, VGG16 stride16, 9 anchors, RPN 2k/4k, 7×7 RoI Pooling, fc6/fc7,
RPN 256 samples, detector 2 images×64 RoIs, SGD를 구현합니다.

PyTorch RGB 정규화·half-open 좌표·고정 box target std·샘플링/손실 구현 세부는
원본 Caffe와 다릅니다. 따라서 **원본 수치의 완전 재현을 보장하지 않습니다.**
paper07은 VOC2007 trainval→test와 총 240k step, quick은 train→val과 기본 총 1600 step입니다.
Google Drive의 run 폴더 전체를 유지하고 중단 뒤 동일 설정 + `RESUME=True`로 실행하세요.
''')
md('## 전체 구현 코드\n이 셀은 구현을 파일에 저장합니다. 모델·학습·평가 코드를 펼쳐 읽을 수 있습니다.')
code('%%writefile /content/faster_rcnn.py\n' + (root / 'faster_rcnn.py').read_text(encoding='utf-8'))
md('''## 실행
이 셀은 체크→다운로드→학습→평가→그림을 순서대로 실행합니다.
처음 VGG 분류 가중치와 VOC 다운로드에 시간이 걸립니다. quick도 4단계를 모두 학습합니다.
EVAL_LIMIT=100은 부분 평가이므로 팀 비교에서는 0으로 설정해 전체를 평가하세요.
''')
code(imports + "SOURCE_CODE = Path('/content/faster_rcnn.py').read_text(encoding='utf-8')\n" + runner)
md('''## 내 사진에 적용 (선택)
아래 셀은 자동 실행하지 않도록 주석 처리했습니다. 학습이 끝난 뒤 주석을 풀어 사용하세요.
''')
code('''# from google.colab import files
# uploaded = files.upload()
# image_path = next(iter(uploaded))
# run_command('predict', '--checkpoint', CHECKPOINT, '--image', image_path,
#             '--output', RUN_DIR + '/my_prediction.png', '--threshold', '0.5')
# display(DisplayImage(filename=RUN_DIR + '/my_prediction.png'))
''')
md('''## 기록할 결과
- 실제 실행한 프로필, 단계별 iteration, 데이터 split, seed, torch/torchvision 버전
- `losses.jsonl`: 단계별 loss (RPN과 detector loss는 단계가 다르므로 동일 곡선으로 혼합하지 않기)
- `metrics_*.json`: 클래스별 AP, VOC2007 11-point mAP@0.5, 평가 이미지 수
- `prediction.png/.json`: 예측 예시
- 논문 수치와 차이 및 구현 변경점

[논문](https://arxiv.org/abs/1506.01497) · [저자 코드](https://github.com/rbgirshick/py-faster-rcnn)
''')
nb = {'cells': cells, 'metadata': {'accelerator': 'GPU',
      'colab': {'name': 'faster_rcnn_colab.ipynb', 'provenance': []},
      'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
      'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
for index, cell in enumerate(cells):
    cell['id'] = f'faster-rcnn-{index:02d}'
(root / 'faster_rcnn_colab.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding='utf-8')
print('Created self-contained notebook and single-cell code.')

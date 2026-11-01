# Faster R-CNN 논문 구현 — Google Colab

**Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks**를 구현합니다. 2014년 R-CNN(Selective Search + 개별 CNN + SVM) 구현과는 다릅니다.

## 실행 파일

- `faster_rcnn_colab.ipynb`: 코랩에 업로드하고 위에서부터 실행. 소스 코드가 내장되어 다른 파일 업로드가 필요 없습니다.
- `faster_rcnn_onecell.py`: 파일 전체를 코랩 코드 셀 하나에 붙여넣어 실행. 모델 정의부터 학습·평가·그림까지 포함합니다.
- `faster_rcnn.py`: import 가능한 구현 및 CLI. 완성된 torchvision detection 모델을 불러오지 않습니다.

코랩의 런타임 유형을 **GPU**로 선택하세요. PyTorch/torchvision은 코랩에 설치된 호환 버전을 사용하고 시작할 때 NMS/RoI Pool 연산을 검사합니다. 임의로 torch만 업그레이드하면 torchvision과 충돌할 수 있습니다.

기본 `quick`은 4단계 각각 300 + 100 iteration, 총 1,600 optimizer step입니다. 이것은 실행 흐름을 확인하는 짧은 학습이며, 좋은 검출 결과나 논문 mAP를 보장하지 않습니다. CPU로는 본 학습을 권하지 않습니다. VOC 데이터는 수 GB, VGG 사전학습 가중치는 약 0.5 GB이며, optimizer를 포함한 체크포인트와 각 단계 모델을 모두 저장하므로 저장소 여유를 10 GB 이상 확보하는 것이 좋습니다. 실행 시간은 할당 GPU에 따라 달라집니다.

## 구현 범위

| 항목 | 구현 |
|---|---|
| Backbone | ImageNet 분류 사전학습 VGG16, pool5 제거, stride 16 |
| RPN | 3×3 conv, 물체/배경 2k 출력, 4k 박스 보정 |
| Anchor | 128/256/512 크기 × 0.5/1/2 비율, 9개 |
| RPN 학습 | 이미지 경계 내부 앵커, IoU 0.7/0.3, GT별 최고 IoU, 256개 샘플 |
| RPN loss | CE + Smooth L1, 논문 식 (1)의 lambda=10 / feature-location 수 정규화 |
| 후보 | NMS 0.7, 학습 최대 2000 / 추론 최대 300 |
| 검출기 | RoI **Pooling** 7×7, fc6/fc7 4096, 21종 분류 + 클래스별 박스 회귀 |
| 검출기 학습 | 2 images × 64 RoIs, positive 최대 25%, background IoU [0.1,0.5) |
| 학습 | 논문의 4단계 교대 학습, SGD/momentum 0.9/weight decay 0.0005 |
| 검출 후처리 | 클래스별 NMS 0.3, 이미지당 최대 100 detection |
| 평가 | VOC2007 11-point mAP@0.5, difficult 제외, 중복 검출 FP 처리 |
| 결과물 | 단계별 모델, 저장/재개용 latest.pt, loss JSONL, AP JSON, 예측 PNG/JSON |

## 4단계가 실제 코드에 대응되는 방식

1. VGG의 conv3_1 이상과 RPN을 학습하고 RPN 가중치를 별도로 저장합니다.
2. 새로운 ImageNet VGG와 Fast R-CNN head를 학습합니다. 후보는 1단계 모델을 고정한 `ProposalTeacher`가 생성합니다.
3. 2단계에서 학습한 backbone을 고정합니다. 1단계 RPN 가중치를 불러온 뒤 RPN만 학습합니다.
4. 공유 backbone과 3단계 RPN을 고정하고, 2단계 검출 head만 학습합니다.

2단계에서 teacher가 별도로 존재하므로 GPU 메모리가 추가로 필요합니다. 저장된 후보를 디스크에 미리 쌓는 대신 고정 teacher가 필요할 때 생성합니다. 최종 추론에서는 backbone 하나만 사용합니다.

## 실험 프로필과 데이터 누수 방지

| 프로필 | 학습 데이터 | 평가 데이터 | 단계별 iteration |
|---|---|---|---|
| quick | VOC2007 train | VOC2007 val | 각각 300@1e-3 + 100@1e-4 |
| paper07 | VOC2007 trainval | VOC2007 test | RPN 단계 60k+20k, 검출 단계 30k+10k |

`paper07`은 학습량을 원래 설정에 가깝게 늘린 프로필로 총 240,000 optimizer step입니다. **정확한 수치 재현 완료를 뜻하지 않습니다.** 짧은 코랩 세션 한 번에 끝난다고 가정하지 마세요. 기본 코드가 저장되는 Google Drive 경로를 확인하고, 중단 후 `RESUME=True`로 재개하세요. Google Drive를 끄면 `/content`의 체크포인트는 런타임 초기화 시 사라질 수 있습니다.

`latest.pt`는 optimizer와 Python/NumPy/PyTorch 난수 상태, 2단계 teacher까지 저장합니다. `stage1_rpn.pt`도 이후 단계에 필요하므로 **run 폴더 전체를 유지**하세요. 체크포인트 저장 중 중단되면 기존 latest가 유지되도록 임시 파일 교체를 사용합니다. 같은 출력 폴더에 새 학습을 실행하면 기존 파일을 덮어쓰지 않고 중단합니다. 재개는 동일 프로필·해상도·seed·quick_steps를 요구합니다.

YOLO와 비교할 때는 **같은 train/eval split, VOC 클래스 순서, difficult 처리, VOC2007 AP, 학습량, 입력 조건, GPU**를 기록하세요. quick의 앞 100장 평가는 동작 점검용이며 정식 mAP 비교에 사용하면 안 됩니다. 정식 평가에서는 `EVAL_LIMIT=0`으로 전체 split을 평가하세요. test 결과로 하이퍼파라미터나 모델을 고르지 마세요.

평가 JSON의 `mAP`와 클래스별 `AP`는 0~1 범위이며, `mAP_percent`는 논문 표와 비교하기 편한 0~100 범위입니다. 예를 들어 0.699는 69.9%입니다.

## 논문 및 원본 Caffe 코드와의 차이 — 보고서에 명시

원 논문의 구조와 교대 학습을 따르는 PyTorch 재구현입니다. 아래 차이 때문에 69.9 mAP/198 ms 등 논문 수치를 그대로 재현했다고 주장할 수 없습니다.

1. Caffe BGR 평균 차감과 원래 VGG 가중치 대신, torchvision ImageNet VGG16의 RGB mean/std 정규화를 사용합니다.
2. 좌표는 일관되게 0-based half-open XYXY를 사용합니다. Caffe의 inclusive 좌표와 앵커 반올림 방식은 완전히 동일하지 않습니다.
3. RoI 연산은 torchvision `roi_pool`을 사용합니다. RoIAlign/FPN/ResNet으로 교체하지 않았습니다.
4. 검출기 회귀 target에 고정 std `(0.1,0.1,0.2,0.2)`를 적용합니다. 원 코드의 일부 교대 학습 경로에서 사용한 데이터 기반 target 통계와 다릅니다.
5. RPN Smooth L1은 논문 정의에 따라 beta=1, 회귀항은 식 (1) 기준입니다. 공개 Caffe 코드의 sigma 및 정규화 경로와 수치적으로 같지 않습니다.
6. 두 이미지의 detector gradient를 순차 누적합니다. 수학적으로 두 이미지 평균 손실에 해당하며 한 번에 두 이미지를 저장하는 부담을 줄입니다. RoI가 부족하면 있는 샘플만 사용하고 background 조건을 임의로 확장하지 않습니다.
7. 이미지 샘플링은 복원추출이며 horizontal flip은 매번 확률적으로 적용합니다. 원 코드의 epoch 순회·미리 복제한 flipped 데이터 순서와 다릅니다.
8. teacher 후보는 고정 모델에서 온라인 생성합니다. 원본의 offline proposal 저장 흐름과 차이가 있으나 2단계 detector gradient는 teacher에 전달되지 않습니다.

## 명령줄 사용

```bash
python faster_rcnn.py self-test --full
python faster_rcnn.py train --profile quick --out /content/drive/MyDrive/faster_rcnn/quick
python faster_rcnn.py train --profile quick --out /content/drive/MyDrive/faster_rcnn/quick --resume
python faster_rcnn.py eval --checkpoint /content/drive/MyDrive/faster_rcnn/quick/final.pt --split val
python faster_rcnn.py predict --checkpoint /content/drive/MyDrive/faster_rcnn/quick/final.pt --image photo.jpg
```

`--profile paper07`로 바꾸면 평가 `--split test`를 사용해야 합니다. `paper07` 실행 중 `quick`으로 바꾸어 재개할 수 없습니다. 새로운 실험에는 다른 `--out`을 사용하세요.

데이터 다운로드가 실패하면 이미 확보한 VOC 데이터를 `/content/voc/VOCdevkit/VOC2007`에 풀어놓고 `--no-download`를 사용하세요. 임의의 다른 데이터셋으로 자동 대체하지 않습니다.

## 근거

- 논문 참조 범위: Sections 3.1–3.3, Tables 3/5
- 논문: https://arxiv.org/abs/1506.01497
- 저자 공개 구현: https://github.com/rbgirshick/py-faster-rcnn
- 연산 문서: https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_pool.html
- 데이터 문서: https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html

실제 검증 범위는 함께 제공되는 `VALIDATION.md`에 기록합니다. 학습된 detector 가중치나 실제 VOC mAP 측정값은 배포 파일에 포함되어 있지 않습니다.

# YOLO v1 — Colab 논문 구현

**시작:** `YOLO_v1_paper_colab.ipynb`를 Colab의 **파일 → 노트북 업로드**로 열고 GPU 런타임에서 위부터 실행하세요.
코드를 한 셀에 통째로 붙여 넣고 싶으면 `YOLO_v1_onecell.py` 전체를 복사하세요.
별도 저장소 clone이나 모듈 파일 업로드가 필요하지 않습니다.

## 구성

- `YOLO_v1_paper_colab.ipynb`: 설명과 전체 코드가 내장된 노트북.
- `YOLO_v1_onecell.py`: 한 셀 복붙용 전체 코드. 맨 아래 실행 설정 수정.
- `yolov1.py`: 동일한 구현을 담은 모듈/CLI.
- `test_yolov1.py`: 다운로드 없는 합성 데이터 회귀 테스트.
- `build_notebook.py`: 모듈 수정 후 노트북/통코드 재생성.
- `VALIDATION.md`: 실제 수행한 검증 및 미실행 항목. 9개 테스트와 전체 모델 forward/backward 통과.

## 모드 선택

| 모드 | 데이터 | 학습 | 평가 |
|---|---|---|---|
| smoke, 기본 | VOC07 train 128장 | scratch 1 epoch | VOC07 val 64장 |
| dev | VOC07 train + VOC12 trainval | 기본 30 epochs | VOC07 val, best 저장 |
| paper | VOC07 trainval + VOC12 trainval | 사전학습 필수, 135 epochs | 고정 final 모델로 VOC07 test |

`smoke`는 실행 확인용입니다. mAP가 0이거나 검출이 부정확할 수 있습니다.
무작위 초기화한 이 모드의 수치를 논문 성능으로 보고하지 마세요.
`paper`도 실행만 하면 63.4 mAP가 보장되는 것은 아닙니다. 이 작업에서 전체 VOC/ImageNet 학습은 수행하지 않았습니다.

## 논문에서 구현한 부분

- Figure 3의 24 convolution + FC4096 + FC1470, 입력 448×448.
- S=7, B=2, C=20; output 7×7×30.
- LeakyReLU 0.1, dropout 0.5, 마지막 층 linear. BN/anchors/sigmoid/softmax 없음.
- class SSE, xy/√wh SSE, confidence SSE, 책임 박스의 IoU 기반 할당과 target.
- λcoord=5, λnoobj=0.5, SGD momentum 0.9, weight decay 0.0005.
- paper 모드 effective batch 64와 75/30/30 epochs의 LR 스케줄.
- 이미지와 박스를 함께 변환하는 증강, class-wise NMS, 원본 GT 기반 VOC07 mAP.
- first20+GAP+FC1000 ImageNet 분류 사전학습 및 backbone 이전 경로.

## 명시적인 구현 선택/재현 한계

1. **사전학습:** 첫 20개 Conv의 ImageNet 사전학습이 없으면 원문 조건이 아닙니다.
   제공된 `pretrain_imagenet()`로 생성한 `imagenet_backbone.pt`를 사용합니다.
   ImageNet 데이터는 사용자가 별도로 준비합니다. 90-epoch 분류 recipe는 원문에 상세 스케줄이 없어 선택한 설정입니다.
   사전학습 top-5는 실제 측정값을 기록하며 원문의 88%를 주장하지 않습니다.
2. **같은 셀의 여러 GT:** 가장 큰 물체 하나만 학습 target으로 사용합니다. 원문이 tie/collision 선택을 명시하지 않아 정한 규칙입니다.
   `collision_objects_skipped`를 기록하며 평가에서는 모든 GT를 유지합니다.
3. **sqrt 수치 처리:** linear w/h가 음수인 경우 `x/sqrt(abs(x)+1e-6)` 확장을 사용합니다.
   원문 수식의 양수 영역을 근사하면서 음수 예측에서도 gradient를 유지하는 선택입니다.
4. **warm-up/NMS:** warm-up은 첫 75 epochs에 포함된 1 epoch, NMS IoU=0.5로 설정했습니다.
   원문에서 구체적으로 지정되지 않은 값입니다.
5. **데이터 증강/초기화:** ±20% crop/pad, flip, HSV S/V 증강. 보간·padding·초기화는 원본 Darknet과 완전히 같지 않습니다.
6. **정밀도:** AMP를 기본 사용합니다. `amp=False`이면 FP32입니다. BN이 없으므로 작은 microbatch의 accumulation이 편리합니다.
7. **no-object 항:** 물체가 있는 셀에서도 비담당 박스 predictor의 confidence를 0으로 학습합니다.
8. **출력 저장 순서:** 클래스 20개 뒤에 박스별 xywh/confidence를 저장합니다. 원본 Darknet 바이너리와 직접 호환하지 않습니다.
9. **평가:** VOC2007 11-point AP at IoU>0.5. difficult는 ignore하며 중복 예측은 FP입니다.
   bbox를 1-based inclusive에서 0-based continuous edge로 바꾸므로 pixel +1 규칙을 정규화 좌표에 잘못 추가하지 않습니다.
   AP threshold는 0.001, 시각화 threshold는 0.2로 분리했습니다. score/NMS 임계값은 test를 보고 튜닝하지 마세요.
10. **범위:** VOC2012 test 서버, artwork, Fast YOLO, Fast R-CNN ensemble 실험은 포함하지 않습니다.

## 실행과 재개

노트북은 `USE_DRIVE=True`가 기본이며 `MyDrive/computer_vision_project/yolov1_<mode>`에 저장합니다.

- `last.pt`: 마지막 완료 epoch의 model/optimizer/scaler/RNG/config. 동일 설정에서 `RESUME=True`로 재개.
- `best_model.pt`: dev/smoke validation에서 선택한 model. 재개용 optimizer는 없음.
- `final_model.pt`: 마지막 모델만 저장.
- `manifest.json`: 데이터 split/설정/장비/라이브러리/사전학습 정보.
- `history.json`: epoch별 loss, validation mAP, collision 수, 시간.
- `voc2007_test_metrics.json`: paper 완료 후 test mAP와 class별 AP.

파일 하나에 가중치 약 1.1GB, last 약 2.2GB가 필요합니다. atomic-save 임시 파일까지 포함해 출력 폴더는 약 8GB 이상 여유를 권장합니다.
기본 데이터 경로 `/content/voc`는 임시 공간입니다. Drive 용량과 Colab 세션 시간은 사용 환경에 따라 확인하세요.
중단한 epoch의 중간 batch까지 이어지는 것은 아닙니다. 마지막 완료 epoch부터 재개합니다.
자신이 만든 체크포인트만 사용하세요. resume는 optimizer/RNG 복구를 위해 `weights_only=False`를 사용합니다.

GPU OOM이면 microbatch를 2에서 1로 줄이고 새 실행 폴더를 사용하세요. 기존 run 재개 중 microbatch를 바꾸려면
정확한 학습 조건이 달라지므로 해당 변경을 별도 실험으로 기록해야 합니다.
결과가 이미 있는 폴더에 새 학습을 시작하면 덮어쓰지 않고 오류를 냅니다. 새 폴더를 선택하거나 RESUME=True를 쓰세요.

CLI 예:

```bash
python yolov1.py --config config.json
python yolov1.py --pretrain-imagenet /content/imagenet --output-dir /content/drive/MyDrive/yolov1_imagenet
python test_yolov1.py --full-model
```

## 팀원과 비교할 때

같은 VOC split, 같은 AP 정의(VOC07 11-point), 같은 test set을 사용하세요.
FPS는 GPU, batch, precision, warm-up, 이미지 입출력 및 NMS 포함 범위를 맞추세요.
`benchmark()`는 synthetic 448 입력, batch1/FP32, 네트워크만/네트워크+NMS를 나누며 이미지 IO/resize는 제외합니다.
이 FPS를 원문의 Titan X 45 FPS와 동등한 조건의 재현값으로 보고하지 마세요.

## 출처

- 프로젝트 첨부 `paper/yolo v1.pdf`, §2 및 Figure 3.
- [YOLO v1 paper](https://arxiv.org/abs/1506.02640)
- [torchvision VOCDetection](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html)
- [VOC download URLs and checksums](https://github.com/pytorch/vision/blob/main/torchvision/datasets/voc.py)
- [PyTorch AMP](https://docs.pytorch.org/docs/stable/amp.html)

주의: [현재 Darknet yolov1.cfg](https://github.com/pjreddie/darknet/blob/master/cfg/yolov1.cfg)는
BatchNorm/local layer/B=3을 포함합니다. 이 코드에서는 첨부 논문 Figure 3과 §2의 구조를 따랐습니다.

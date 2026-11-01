# 배경과 비슷한 물체를 검출기는 얼마나 잘 찾을까?

컴퓨터비전프로젝트 수업용 저장소입니다. YOLOv1과 Faster R-CNN을 바탕으로, 물체와 배경의 색이 비슷해질 때 검출 결과가 어떻게 달라지는지 비교할 계획입니다.

## 현재 상태

현재는 두 모델의 PyTorch 구현과 Colab 실행 파일을 정리한 단계입니다. **카모플라주 실험, 학습된 검출 가중치, 정량 실험 결과는 아직 포함되어 있지 않습니다.** 아래 코드는 논문을 참고한 교육용 재구현이며 저자 공식 구현 자체는 아닙니다.

기존 VALIDATION.md에는 당시 실시한 합성 데이터 기반 검증 기록이 있습니다. 실제 VOC 전체 학습, Colab GPU 실행, 논문 성능 재현을 완료했다는 의미는 아닙니다. 이번 공개 준비에서는 Python 문법과 노트북 구조만 추가 확인했습니다.

## 파일 구성

| 폴더 | 내용 |
|---|---|
| baselines/yolo_v1_colab | YOLOv1 구현, 학습·평가 코드, Colab 노트북 |
| baselines/faster_rcnn_colab | VGG16 기반 Faster R-CNN 구현, 학습·평가 코드, Colab 노트북 |

각 폴더의 README.md에 실행 방법과 원 논문 구현과의 차이를 정리했습니다.

## 실행 방법

1. Google Colab에서 GPU 런타임을 선택합니다.
2. YOLO는 `baselines/yolo_v1_colab/YOLO_v1_paper_colab.ipynb`, Faster R-CNN은 `baselines/faster_rcnn_colab/faster_rcnn_colab.ipynb`를 엽니다.
3. 노트북의 데이터·저장 경로와 실행 설정을 확인한 뒤 위에서부터 실행합니다.

기본 짧은 학습은 동작 확인용입니다. 짧게 실행했다고 좋은 검출 결과가 나오는 것은 아닙니다. 모델이 크므로 GPU 메모리와 저장 공간을 확인해야 합니다.

로컬 실행에는 Python 3.10 이상, 호환되는 PyTorch·torchvision, NumPy, Pillow가 필요합니다. CUDA 환경에 맞게 PyTorch와 torchvision을 함께 설치하고 각 폴더의 안내를 따릅니다. 실행 환경을 확정하면 버전을 기록할 예정입니다.

## 앞으로 할 실험

- 기존 데이터셋에서 정답 박스와 객체 마스크를 사용할 수 있는 이미지 선정
- 두 모델의 학습된 검출 가중치와 공통 평가 클래스 확인
- 물체의 색을 주변 배경과 비슷하게 바꾸는 이미지 생성
- 원본과 변형 이미지의 검출 성공률·오검출 비교
- 성능이 떨어지는 사례와 개선 방법 분석

촬영 대신 기존 데이터셋을 활용할 예정입니다. 데이터 선정 기준과 변형 강도는 예비 실험 후 확정합니다. 사전학습 가중치를 활용하는 실험 경로는 별도로 확인할 예정이며, 현재 구현용 가중치와 호환된다고 가정하지 않습니다.

## 데이터와 가중치

데이터셋, 모델 가중치, 체크포인트는 저장소에 포함하지 않습니다. 각 제공처의 이용 조건을 따르고, 실제 채택한 다운로드 경로와 평가 조건을 추후 기록합니다.

## 참고 자료와 작성 기록

- YOLO 논문: https://arxiv.org/abs/1506.02640
- YOLO 저자 코드: https://github.com/pjreddie/darknet
- Faster R-CNN 논문: https://arxiv.org/abs/1506.01497
- Faster R-CNN 저자 코드: https://github.com/rbgirshick/py-faster-rcnn
- PASCAL VOC: https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/

코드 작성 및 정리에 AI 도구의 도움을 받았습니다. 팀에서 실행 결과와 구현 내용을 확인하며 수정할 예정입니다. 외부 코드·데이터·가중치는 각 원저작자의 이용 조건을 따릅니다.

# 검증 기록

검증일: 2026-10-04. Windows CPU, Python 3.12, torch 2.14.1+cpu / torchvision 0.29.1+cpu.

## 실제 실행하여 통과

`faster_rcnn.py self-test --full --device cpu`

- Box encode/decode round trip
- Anchor 개수와 RPN 출력 대응
- RPN loss 및 feature까지의 유한한 gradient
- 정답 물체가 없는 이미지의 RPN loss
- NMS 및 최대 300 proposal, proposal gradient 차단
- RoI sampling 및 RoI Pooling shape
- VOC2007 AP의 정답 검출 / 완전 오검출 / difficult 무시 / 중복 FP
- 전체 VGG16 모델의 1~4단계 각각 forward/backward
- 각 단계에서 학습 대상 parameter gradient 존재, 고정 parameter gradient 없음
- 2단계 teacher로 gradient가 전달되지 않음
- 전체 검출 모델 inference

`test_checkpoint_flow.py`

- 실제 train/save/resume 제어 흐름을 작은 합성 모델과 합성 데이터로 실행
- 각 단계의 첫 iteration 직후 체크포인트를 저장하고 의도적으로 중단
- 1/2/3/4단계 각각에서 재개한 최종 가중치가 연속 실행과 bitwise 일치
- 이 테스트는 제어 흐름 검증이며 VGG 학습 성능 측정이 아님

정적 검증: 모든 Python 파일의 AST 구문 분석, notebook JSON 및 코드 셀 구문,
셀 ID 중복 없음, 내장 구현과 배포 Python 소스의 동일성.

## 검증하지 않은 범위

- 실제 Google Colab GPU 세션 실행, Google Drive mount
- 실제 VOC 다운로드와 전체 데이터 학습
- 실제 ImageNet 가중치 다운로드 (모델 구조 검증은 pretrained=False)
- 실제 VOC mAP, 논문 정확도 재현, GPU 속도/메모리 사용량

따라서 동작 테스트 통과는 논문 성능 재현 완료를 뜻하지 않습니다.
Colab 노트북은 시작할 때 설치된 torch/torchvision 버전과 GPU를 출력하고
기본 self-test를 실행합니다. GPU 연산 또는 버전 호환 문제 발생 시 즉시 중단합니다.

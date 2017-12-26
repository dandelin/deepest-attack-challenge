# deepest-attack-challenge
Perform an adversarial attack on a pre-trained neural network

## 문제 설명
이미 학습되어있는 뉴럴 네트워크를 공격하는 입력을 만들고자 합니다. 여기서 *공격*은 뉴럴 네트워크의 학습 의도와 상관 없는 결과가 나오는 입력을 주는 행위입니다. 저희가 제공해드린 Deepest의 [로고 이미지](https://github.com/dandelin/deepest-attack-challenge/blob/master/deepest.png)를 변조하여 뉴럴 네트워크를 공격하여 봅시다.

### 대상 네트워크
이번 챌린지에서 공격의 대상이 되는 뉴럴 네트워크는 이미지를 천 개 클래스 중 하나로 분류하는 ImageNet 챌린지의 데이터셋으로 이미 학습된 ResNet-18입니다. 만약 pytorch를 사용하신다면 [링크](http://pytorch.org/docs/0.3.0/torchvision/models.html)에서 간단히 사용법을 알아보실 수 있습니다.

### 목표
본 챌린지의 목표는 앞서 설명한 Deepest의 로고를 [다음](https://github.com/dandelin/deepest-attack-challenge/blob/master/deepest_brain.png)과 같이 변조하여 대상 네트워크에서 변조된 이미지가 99% 이상의 확률로 "brain coral"인 것으로 확신하게 해야 합니다. (=109번째 인덱스의 softmax값이 0.99가 넘어야 합니다.) 또한 기존의 deepest 로고를 알아볼 수 있어야 합니다. (=기존의 이미지와 픽셀 값의 차이가 적어야 합니다.)

## 예제 설명
이 레포지토리를 클론하여 pytorch 0.3 이상을 설치하신 뒤 inference.py를 실행하면, deepest.png와 deepest\_brain.png를 pretrained resnet-18로 inference한 결과를 출력합니다. 직접 변조한 이미지의 이름을 deepest\_brain.png로 변조하여 성공적으로 챌린지를 수행했는지 확인해 볼 수 있습니다.
